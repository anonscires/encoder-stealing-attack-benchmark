# %% define the DinoEncoder class
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

from src.models.resnet import blur_with_mask

import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np

from src.models.custom_transforms import LaplaceSubstitute

class DinoEncoder(nn.Module):
    def __init__(self, base_model='dino-vitb16', out_dim=768, loss=None, include_mlp = True, entropy=False, pretrained=True):
        

        if "dino" in base_model:
            base_model_name = base_model.replace("-", "_")
            if pretrained == False:
                print("Warning: pretrained=False is not supported for DINO models. Using pretrained=True.")
                pretrained = True
        elif base_model in ['vitb16', 'vits16']:
            base_model_name = "dino_" + base_model
            if pretrained == True:
                print("Warning: pretrained=True is not supported for ViT models. Using pretrained=False.")
                pretrained = False
        else:
            raise ValueError(f"Unsupported base model {base_model}. Choose either 'dino-vitb16' or 'dino-vits16'.")

        print("_" * 20)
        print("DinoEncoder")
        print("_" * 20)
        print(f"base_model: {base_model}")
        print(f"base_model_name: {base_model_name}")
        print(f"out_dim: {out_dim}")
        print(f"loss: {loss}")
        print(f"include_mlp: {include_mlp}")
        print(f"entropy: {entropy}")
        print(f"pretrained: {pretrained}")
        print("_" * 20)

        super(DinoEncoder, self).__init__()

        self.backbone = torch.hub.load('facebookresearch/dino:main', base_model_name, pretrained=pretrained)

        self.loss = loss
        self.entropy = entropy

        dim_mlp = self.backbone.blocks[-1].mlp.fc2.out_features # 512 for ViT CLIP

        if include_mlp:
            print(f"MLP Included: From {dim_mlp} dim to {out_dim} dim")
            # add mlp projection head
            if self.loss == "symmetrized":
                self.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                 nn.BatchNorm1d(dim_mlp),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(dim_mlp, out_dim))
            else:
                self.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
        else:
            self.fc = nn.Identity() # no head used

        print("Backbone:", self.backbone)
        print("FC:", self.fc)
        print("_" * 20)

    def forward(self, images):
        """
        Forward pass for the CLIP image encoder.
        Args:
            images: A batch of images (PIL images or tensors).
        Returns:
            image_features: Encoded image features.
        """
        # inputs = self.processor(images=images, return_tensors="pt", padding=True)
        # outputs = self.backbone.get_image_features(**inputs)
        outputs = self.backbone(images)
        outputs = self.fc(outputs)
        return outputs
    
    @staticmethod
    def get_transform():
        """
        Returns the transformation function for preprocessing images.
        """
        transform = transforms.Compose([
                        transforms.Resize((224, 224)),  # ViT expects 224x224 images
                        # transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        return transform

from pytorch_grad_cam import GradCAM

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class DinoGradCamEncoder(DinoEncoder):

    def __init__(self, base_model='dino-vitb16', out_dim=768, loss=None, include_mlp=True, entropy=False, pretrained=True, batch_size=128):
        """
        Initializes the DinoGradCamEncoder with the specified parameters.
        """
        super(DinoGradCamEncoder, self).__init__(base_model, out_dim, loss, include_mlp, entropy, pretrained)
        self.model = torch.nn.Sequential(self.backbone)
        self.cam = GradCAM(model=self.model, target_layers=[self.model[0].blocks[-1].norm1], reshape_transform=reshape_transform)
        print("DinoGradCamEncoder initialized with base model:", base_model, "and batch size:", batch_size)
        self.cam.batch_size = batch_size

    def forward(self, x):
        """
        Forward pass for the DinoGradCamEncoder.
        Args:
            images: A batch of images (PIL images or tensors).
        Returns:
            image_features: Encoded image features.
        """

        targets = None
        cam = self.cam(input_tensor=x, targets=targets)

        # Convert to numpy array 
        x_np = x.cpu().numpy()

        blurred_x = []
        for x_np_i, cam_i in zip(x_np, cam):
            blurred_x_i = blur_with_mask(x_np_i, cam_i)
            blurred_x.append(blurred_x_i)
        blurred_x = np.stack(blurred_x, axis=0)
        blurred_x = torch.from_numpy(blurred_x).to(x.device)

        blurred_x = blurred_x.permute(0, 3, 1, 2)

        return self.model(blurred_x)
    
class DinoGradCamEncoderWithLogits(torch.nn.Module):

    def __init__(self, base_model='dino-vitb16', out_dim=768, loss=None, include_mlp=True, entropy=False, pretrained=True, ckpt_path=None, batch_size=128, reshape_transform=reshape_transform):
        """
        Initializes the DinoGradCamEncoder with the specified parameters.
        """
        super(DinoGradCamEncoderWithLogits, self).__init__()
        dino = DinoEncoder(base_model=base_model, out_dim=out_dim, loss=loss, include_mlp=include_mlp, entropy=entropy, pretrained=pretrained)
        self.model = torch.nn.Sequential(dino, torch.nn.Linear(out_dim, 10))
        checkpoint = torch.load(ckpt_path)
        print("Loading checkpoint from:", ckpt_path)
        self.model.load_state_dict(checkpoint, strict=True)
        print("Checkpoint loaded successfully.")
        self.cam = GradCAM(model=self.model, target_layers=[self.model[0].backbone.blocks[-1].norm1], reshape_transform=reshape_transform)
        print("DinoGradCamEncoder initialized with base model:", base_model, "and batch size:", batch_size)
        self.cam.batch_size = batch_size

    def forward(self, x):
        """
        Forward pass for the DinoGradCamEncoder.
        Args:
            images: A batch of images (PIL images or tensors).
        Returns:
            image_features: Encoded image features.
        """

        targets = None
        cam = self.cam(input_tensor=x, targets=targets)
        
        # Convert to numpy array 
        x_np = x.cpu().numpy()

        blurred_x = []
        for x_np_i, cam_i in zip(x_np, cam):
            blurred_x_i = blur_with_mask(x_np_i, cam_i)
            blurred_x.append(blurred_x_i)
        blurred_x = np.stack(blurred_x, axis=0)
        blurred_x = torch.from_numpy(blurred_x).to(x.device)

        blurred_x = blurred_x.permute(0, 3, 1, 2)

        return self.model[0](blurred_x)
    
    @staticmethod
    def get_transform():
        """
        Returns the transformation function for preprocessing images.
        """
        return DinoGradCamEncoder.get_transform()

import torch
from torchvision import transforms

class MinMaxScale(object):
    """
    Custom torchvision transform for min-max scaling a tensor to [0, 1].
    """
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)

class DinoGuardCamEncoder(DinoEncoder):

    def __init__(self, base_model='dino-vitb16', out_dim=768, loss=None, include_mlp=True, entropy=False, pretrained=True):
        """
        Initializes the DinoGuardCamEncoder with the specified parameters.
        """
        super(DinoGuardCamEncoder, self).__init__(base_model, out_dim, loss, include_mlp, entropy, pretrained)
        print("DinoGuardCamEncoder initialized with base model:", base_model)
        # gaussian blur fileter
        print("Using Gaussian Blur with kernel size (23, 23) and sigma 5")
        self.gb = transforms.GaussianBlur(kernel_size=(23, 23), sigma=5)
        self.minmaxTransform = transforms.Compose([MinMaxScale(),])


    def forward(self, images):
        """
        Forward pass for the DinoGuardCamEncoder.
        Args:
            images: A batch of images (PIL images or tensors).
        Returns:
            image_features: Encoded image features.
        """

        # Forward pass
        img_features = self.backbone(images)

        targets = img_features.norm(p=2, dim=1)
        targets.backward(torch.ones_like(targets))

        # Compute saliency maps for the batch
        saliency_maps = images.grad.abs()

        saliency_maps = saliency_maps.sum(axis=1).unsqueeze(1).expand(-1, 3, -1, -1)
        saliency_maps = self.minmaxTransform(saliency_maps)
        saliency_maps = self.gb(saliency_maps)
        # threshold = torch.quantile(saliency_maps.flatten(), 0.75)
        threshold = torch.mean(saliency_maps)
        mask = saliency_maps >= threshold
        # mask.requires_grad_(False)  # Ensure mask is not differentiable

        blur_imgs = self.gb(images)
        # saliencies_stack = torch.stack(saliencies, dim=0)
        final_imgs = torch.where(mask, blur_imgs, images)

        self.backbone.zero_grad()
        outputs = self.backbone(final_imgs)

        return outputs
    
    @staticmethod
    def get_transform():
        """
        Returns the transformation function for preprocessing images.
        """
        # substitute = transforms.GaussianBlur(kernel_size=15, sigma=3)
        substitute = transforms.Lambda(lambda x: x + torch.from_numpy(np.random.laplace(loc=0.0, scale=0.05, size=x.shape)).type_as(x))

        transform = transforms.Compose([
                        transforms.Resize((224, 224)),  # ViT expects 224x224 images
                        transforms.ToTensor(),
                        LaplaceSubstitute(substitute=substitute, kernel_size=29),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        return transform

# %% Code to test the DinoEncoder
if __name__ == "__main__":
    # Create a random tensor simulating a batch of 10 images with 3 channels (RGB) and size 244x244
    tensor_image = torch.rand(10, 3, 244, 244)

    # Get the transformation function
    transform = DinoEncoder.get_transform()

    # Apply the transformation to the tensor_image
    # Since the transform expects PIL images, convert the tensor to PIL images first
    tensor_image = [transforms.ToPILImage()(img) for img in tensor_image]
    transformed_images = [transform(img) for img in tensor_image]

    # Stack the transformed images back into a tensor
    tensor_image = torch.stack(transformed_images)

    # Initialize the CLIP image encoder
    encoder = DinoEncoder('dino', out_dim=128)

    print(f"Encoder: {encoder}")

    # Freeze the parameters of the CLIP model to prevent them from being updated during training
    for param in encoder.parameters():
        param.requires_grad = False

    # Print the shape and trainable status of each parameter in the encoder
    for param in encoder.parameters():
        print(f"Parameter Shape: {param.shape}, Trainable: {param.requires_grad}")

    # Pass the tensor image batch through the encoder to get encoded image features
    image_features = encoder(tensor_image)
    print(f"Encoded image features: {image_features.shape}", image_features)

# %%
