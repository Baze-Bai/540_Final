import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        """
        Decoder layer: Upsampling + Convolution
        Args:
            in_channels: Number of input channels (from deeper layers).
            out_channels: Number of output channels.
            skip_channels: Number of channels in skip features to be concatenated (default 0 means no skip connection).
        """
        super(DecoderLayer, self).__init__() 
        
        # Upsampling (transposed convolution, doubles the spatial dimensions)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Input channels after concatenation = out_channels + skip_channels (if skip exists)
        # Two convolution layers restore out_channels
        in_conv1 = out_channels + skip_channels if skip_channels > 0 else out_channels
        self.conv1 = nn.Conv2d(in_conv1, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, skip_feat=None):
        """
        x: [B, k, C, H, W] Feature map from the next level, B is batch size, k is number of images
        skip_feat: [B, k, C_skip, H*2, W*2] Corresponding encoder layer skip features
        Returns: Upsampled and convolved fused features [B, k, out_channels, H*2, W*2]
        """
        B, k, C, H, W = x.shape
        x = x.view(B*k, C, H, W)  # Reshape to [B*k, C, H, W] to fit 2D convolution
        
        x_up = self.upconv(x)  # Upsampling (H,W -> 2H,2W)
        
        if skip_feat is not None:
            # Reshape skip_feat for concatenation
            _, _, C_skip, H2, W2 = skip_feat.shape
            skip_feat = skip_feat.view(B*k, C_skip, H2, W2)
            x_up = torch.cat([x_up, skip_feat], dim=1)  # Concatenate skip features along channel dimension
            
        h = F.relu(self.conv1(x_up))
        out = F.relu(self.conv2(h))
        
        # Restore original batch dimensions [B*k, C_out, H*2, W*2] -> [B, k, C_out, H*2, W*2]
        out = out.view(B, k, out.size(1), out.size(2), out.size(3))
        
        return out