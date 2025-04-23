import torch
import torch.nn as nn
import torch.nn.functional as F
from Scripts.MultiAttention_Unet.MultiImage_Unet import MultiImageUNet

class UnetModel(nn.Module):
    def __init__(self, in_channels=1):
        super(UnetModel, self).__init__()
        
        # Use the MultiImageUNet as the base architecture
        self.base_unet = MultiImageUNet(in_channels=in_channels)
        
        # Add a final upsampling layer to restore original resolution (1551, 868)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x_images, reference_images):
        """
        Args:
            x_images: [B, k, 1, 800, 800] - Batch of input images (resized)
            reference_images: [B, k, 1, 800, 800] - Used for reference (resized)
        
        Returns:
            outputs: [B, k, 1, 1551, 868] - Output images upsampled to original resolution
        """
        # Pass through the base UNet model
        base_output = self.base_unet(x_images, reference_images)  # [B, k, 1, H, W]
        
        # Get dimensions
        B, k, C, H, W = base_output.shape
        
        # Reshape for processing each image separately
        base_output_flat = base_output.view(B * k, C, H, W)
        
        # Upsample to original resolution (1551, 868)
        resized_output = F.interpolate(base_output_flat, size=(1551, 868), mode='bilinear', align_corners=False)
        
        # Apply final convolution layers for better quality
        final_output = self.final_upsample(resized_output)
        
        # Reshape back to [B, k, 1, 1551, 868]
        outputs = final_output.view(B, k, 1, 1551, 868)
        
        return outputs
