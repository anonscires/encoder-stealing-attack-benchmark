# %% Defining the architecture
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from urllib.request import urlopen

# -------------------------------
# Preprocessing: Resize + Normalize
# -------------------------------
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0), image  # Add batch dimension


# -------------------------------
# Grad-CAM Class
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)  # Forward pass

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()  # Pick predicted class

        # Backward pass for target class
        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Global average pooling over gradients (N, C, H, W) -> (N, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # shape: [N, 1, H, W]

        # Apply ReLU
        cam = F.relu(cam)

        # Resize CAM to input size (usually 224x224)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


# -------------------------------
# ImageNet class name mapping
# -------------------------------
def load_imagenet_classes():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with urlopen(url) as f:
        class_idx = json.load(f)
    return {int(k): v[1] for k, v in class_idx.items()}


# -------------------------------
# Visualization
# -------------------------------
def show_cam_on_image(original_img, cam, label=None):
    img_np = np.array(original_img.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    cam_overlay = heatmap * 0.5 + img_np * 0.5
    cam_overlay = np.clip(cam_overlay, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(cam_overlay)
    plt.axis('off')
    if label:
        plt.title(f"Predicted Class: {label}", fontsize=14)
    plt.show()


def show_masked_cam(original_img, cam, threshold=0.5):
    """
    Show the Grad-CAM mask, and overlay the mask on the original image,
    only keeping the portion with more than `threshold` confidence visible.
    """
    # Resize cam to match original image size
    cam_resized = cv2.resize(cam, original_img.size, interpolation=cv2.INTER_LINEAR)
    mask = cam_resized > threshold

    # Show the Grad-CAM mask
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(cam_resized, cmap='jet')
    plt.title('Grad-CAM Mask')
    plt.axis('off')

    # Overlay: keep only > threshold
    img_np = np.array(original_img).astype(np.float32) / 255.0
    masked_img = img_np.copy()
    masked_img[~mask] = 0  # Set pixels below threshold to black

    plt.subplot(1, 2, 2)
    plt.imshow(masked_img)
    plt.title(f'Overlay (>{int(threshold*100)}% confidence)')
    plt.axis('off')
    plt.show()



def blur_with_mask(img, cam, max_blur=15, min_blur=1):
    """
    Blurs the image with a spatially-varying blur: higher Grad-CAM confidence = more blur.
    Args:
        input_tensor: torch.Tensor, shape [1, 3, H, W]
        cam: np.ndarray, shape [H, W], values in [0, 1]
        max_blur: int, maximum kernel size for blurring (should be odd)
        min_blur: int, minimum kernel size for blurring (should be odd)
    Returns:
        np.ndarray: Blurred image (H, W, 3) in [0, 1] range
    """
    # Convert tensor to numpy image
    img = np.transpose(img, (1, 2, 0))  # [H, W, C]
    # Unnormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    # Resize cam to match image size
    cam_resized = cv2.resize(cam, (img_uint8.shape[1], img_uint8.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Prepare output
    result = np.zeros_like(img_uint8, dtype=np.float32)

    # For efficiency, precompute blurred images for a set of blur levels
    blur_levels = np.linspace(min_blur, max_blur, num=5)
    blur_levels = [int(round(k)) | 1 for k in blur_levels]  # Ensure odd
    blurred_imgs = []
    for k in blur_levels:
        blurred_imgs.append(cv2.GaussianBlur(img_uint8, (k, k), 0))

    # For each pixel, interpolate between the blurred images according to cam confidence
    cam_flat = cam_resized.flatten()
    indices = np.searchsorted(np.linspace(0, 1, num=len(blur_levels)), cam_flat, side='right') - 1
    indices = np.clip(indices, 0, len(blur_levels) - 1)
    h, w = cam_resized.shape
    for i, blurred in enumerate(blurred_imgs):
        mask = (indices.reshape(h, w) == i)
        result[mask] = blurred[mask]

    # Convert back to float [0, 1]
    result = result.astype(np.float32) / 255.0
    return result


# -------------------------------
# %% Main
# -------------------------------
if __name__ == "__main__":
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)
    target_layer = model.layer4[-1]

    gradcam = GradCAM(model, target_layer)

    # Load image and preprocess
    input_tensor, original_image = preprocess_image("image.png")  # Replace with your image path

    # Generate CAM
    cam, class_idx = gradcam.generate_cam(input_tensor)

    print(f"CAM shape: {cam.shape}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Original image shape: {original_image.size}")
    print(f"Class index: {class_idx}")

    # Class label
    class_labels = load_imagenet_classes()
    label = class_labels.get(class_idx, f"Class {class_idx}")
    print(f"Predicted Class: {label}")
    print(f"Class Index: {class_idx}")

    # Show visualization
    show_cam_on_image(original_image, cam, label)
    show_masked_cam(original_image, cam, threshold=0.5)
    blurred_image = blur_with_mask(input_tensor, cam, max_blur=15)
    plt.figure(figsize=(6, 6))
    plt.imshow(blurred_image)
    plt.axis('off')
    plt.title('Blurred Image with Grad-CAM Mask')
    plt.show()

    # Clean up
    gradcam.remove_hooks()

# %%
