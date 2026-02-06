import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch.nn as nn

class CLIPVisionEncoder(nn.Module):
    def __init__(self, base_model, out_dim, loss=None, include_mlp = False, entropy=False, pretrained=True):
        

        if base_model == "clip-vitb32":
            base_model_name = "openai/clip-vit-base-patch32"
        elif base_model == "clip-vitb16"  or base_model == "clip":
            base_model_name = "openai/clip-vit-base-patch16"
        else:
            raise ValueError(f"Unsupported base_model: {base_model}. Choose 'clip-vitb32' or 'clip-vitb16'.")
        
        print("_" * 20)
        print("ResNetEncoder")
        print("_" * 20)
        print(f"base_model arch: {base_model}")
        print(f"base_model name: {base_model_name}")
        print(f"out_dim: {out_dim}")
        print(f"loss: {loss}")
        print(f"include_mlp: {include_mlp}")
        print(f"entropy: {entropy}")
        print(f"pretrained: {pretrained}")
        print("_" * 20)

        super(CLIPVisionEncoder, self).__init__()

        if not pretrained:
            raise NotImplementedError(f'CLIP Model not implemented for pretrained = {pretrained} yet.')

        self.backbone = CLIPModel.from_pretrained(base_model_name)
        self.processor = CLIPProcessor.from_pretrained(base_model_name)

        self.loss = loss
        self.entropy = entropy

        dim_mlp = self.backbone.visual_projection.out_features # 512 for ViT CLIP

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
        print("FC", self.fc)
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
        outputs = self.backbone.get_image_features(images)
        outputs = self.fc(outputs)
        return outputs
    
    @staticmethod
    def get_transform():
        """
        Returns the transformation function for preprocessing images.
        """
        transform = transforms.Compose([
                                transforms.Resize((224, 224)),  # CLIP expects 224x224 images
                                # SSL needs tensor input but ContSteal does not
                                # transforms.ToTensor(),
                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                            ])
        return transform

# Code to test the CLIPImageEncoder
if __name__ == "__main__":
    # Create a random tensor simulating a batch of 10 images with 3 channels (RGB) and size 244x244
    tensor_image = torch.rand(10, 3, 244, 244)

    # Get the transformation function
    transform = CLIPVisionEncoder.get_transform()

    # Apply the transformation to the tensor_image
    # Since the transform expects PIL images, convert the tensor to PIL images first
    tensor_image = [transforms.ToPILImage()(img) for img in tensor_image]
    transformed_images = [transform(img) for img in tensor_image]

    # Stack the transformed images back into a tensor
    tensor_image = torch.stack(transformed_images)

    # Initialize the CLIP image encoder
    encoder = CLIPVisionEncoder(out_dim=128)

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
