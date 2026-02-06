import cv2
import numpy as np
import torch

class LaplaceSubstitute:
    def __init__(self, substitute=lambda x: x, kernel_size=3):
        self.substitute = substitute
        self.kernel_size = kernel_size

    def __call__(self, img: torch.Tensor):
        
        if isinstance(img, torch.Tensor) :
            # Convert tensor to NumPy array
            img_np = img.detach().cpu().numpy()
        else:
            raise TypeError(f"Input should be a Tensory or Numpy, got {type(img)}")
        
        # If image is grayscale, cv2.Laplacian works directly
        if len(img_np.shape) == 2:
            laplacian = cv2.Laplacian(img_np, cv2.CV_64F, ksize=self.kernel_size)
        else:
            # Apply Laplacian to each channel and merge
            laplacian = np.zeros_like(img_np, dtype=np.float64)
            for c in range(img_np.shape[0]): # Assuming img_np is in (C, H, W) format
                laplacian[c, :, :] = cv2.Laplacian(img_np[c, :, :], cv2.CV_64F, ksize=self.kernel_size)

        # Normalize result and clip to valid range
        # laplacian = np.clip(laplacian, 0, 1).astype(np.uint8)
        laplacian = torch.tensor(laplacian).to(img.device)  # Convert back to tensor and keep on the same device

        substitute = self.substitute(img)
        final_img = torch.where(laplacian > laplacian.median(), substitute, img)

        return final_img