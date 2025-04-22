import torch
import torch.nn as nn
import torch.nn.functional as F
from Class_Func.Cross_Attention_Module import CrossAttention


# EncoderLayer module that applies CrossAttention and processes its output through convolutions
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, embed_dim=None, num_heads=2, kernel_size_q=20, stride_q=20, padding_q=10, 
                 kernel_size_kv=3, stride_kv=1, padding_kv=1,kv_length=50, 
                 kernel_size_t=4, stride_t=2, padding_t=1):
        """
        Parameters:
            in_channels: Number of input image channels
            embed_dim: Projection dimension for CrossAttention
            num_heads: Number of attention heads
            kernel_size_q: Kernel size for query projection
            stride_q: Stride for query projection
            padding_q: Padding for query projection
            kernel_size_kv: Kernel size for key-value convolution
            stride_kv: Stride for key-value convolution
            padding_kv: Padding for key-value convolution
            kv_length: Length of key-value sequence
            kernel_size_t: Kernel size for transpose convolution
            stride_t: Stride for transpose convolution
            padding_t: Padding for transpose convolution
        """
        
        super(EncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Downsample layer - reduces input image size by half
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Channel conversion layer - changes channels from C to 1
        self.channel_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Cross attention module
        self.cross_att = CrossAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                        kernel_size_q=kernel_size_q, stride_q=stride_q, padding_q=padding_q,
                                       kernel_size_kv=kernel_size_kv, stride_kv=stride_kv, padding_kv=padding_kv, kv_length=kv_length,
                                       in_channels=1)  # Input channels is 1 after channel_conv
        
        # Channel restoration and upsampling layer - restores channels from embed_dim to C
        # while upsampling to half the original size (H/2, W/2)
        # Uses transpose convolution for upsampling and channel conversion
        self.restore_conv = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=kernel_size_t, stride=stride_t, padding=padding_t)
        
        # Final convolution layer - maintains image size but doubles channel count
        self.final_conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1)
        
        # Optional: Add activation function
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x, r_image):
        """
        Parameters:
            x:       [B, k, C, H, W] - B is batch size, k is number of images, C is channel count, H and W are image dimensions
            r_image: [B, k, 1, H, W] - Reference image, single channel
            
        Processing flow:
            1. Downsample x to half its original size
            2. Convert channels from C to 1 via convolution
            3. Process features through CrossAttention to get attn_out
            4. Restore attn_out to original channel count C via transpose convolution, while upsampling to H/2 and W/2
            5. Add to downsampled features (skip connection)
            6. Apply final convolution to maintain image size but double the channel count
        """
        # Get input dimensions
        B, k, C, H, W = x.shape
        
        # Step 1: Downsample to half the original size
        # First reshape to [B*k, C, H, W] for downsampling
        x_reshaped = x.view(B*k, C, H, W)
        downsampled = self.downsample(x_reshaped)  # [B*k, C, H/2, W/2]
        
        # Get downsampled image dimensions
        _, _, H_d, W_d = downsampled.shape
        
        # Save downsampled result for later skip connection
        skip_connection = downsampled.clone()
        
        # Step 2: Convert channels from C to 1 via convolution
        x_single_channel = self.channel_conv(downsampled)  # [B*k, 1, H/2, W/2]
        
        # Reshape back to [B, k, 1, H/2, W/2] for CrossAttention
        x_prepared = x_single_channel.view(B, k, 1, H_d, W_d)
        
        # Step 3: Process through CrossAttention to get attn_out
        # r_image doesn't need downsampling, directly passed to CrossAttention
        
        # Call CrossAttention, output shape is [B, k, embed_dim, H_out, W_out]
        attn_out = self.cross_att(x_prepared, r_image)
        
        # Step 4: Restore attn_out to original channel count and upsample
        # Get attn_out output dimensions
        _, _, embed_out, H_out, W_out = attn_out.shape
        
        # Reshape to [B*k, embed_dim, H_out, W_out] for transpose convolution
        attn_out_reshaped = attn_out.view(B*k, embed_out, H_out, W_out)
        
        # Restore channel count and upsample via transpose convolution, output is [B*k, C, H/2, W/2]
        restored_features = self.restore_conv(attn_out_reshaped)
        
        # Ensure upsampled size matches skip_connection dimensions
        if restored_features.shape[1:] != skip_connection.shape[1:]:
            restored_features = F.interpolate(restored_features, size=(H_d, W_d), mode='bilinear', align_corners=False)
        
        # Step 5: Add restored features to downsampled features (skip connection)
        output = restored_features + skip_connection
        
        # Apply activation function
        output = self.act(output)
        
        # Step 6: Apply final convolution to maintain image size but double channel count
        output_reshaped = output.view(B*k, C, H_d, W_d)
        output_doubled = self.final_conv(output_reshaped)
        
        # Reshape output back to [B, k, C*2, H/2, W/2]
        output = output_doubled.view(B, k, C*2, H_d, W_d)
        
        return output
    