import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttCvT(nn.Module):
    def __init__(self, in_channels=1, embed_dim=8, num_heads=2, kernel_size_q=22, stride_q=20, padding_q=1, 
                 kernel_size_kv=22, stride_kv=20, padding_kv=1):
        """
        CvT-style Self-Attention Module: Performs self-attention fusion on k input image features.
        Args:
            in_channels: Number of input channels (1 for grayscale).
            embed_dim: Attention embedding dimension (e.g., 32).
            num_heads: Number of attention heads.
            kernel_size_q: Kernel size for query projection
            stride_q: Stride for query projection
            padding_q: Padding for query projection
            kernel_size_kv: Kernel size for key-value projection
            stride_kv: Stride for key-value projection
            padding_kv: Padding for key-value projection
        """
        super(SelfAttCvT, self).__init__()
        # Convolution projection that maps images to embed_dim features
        self.embed_dim = int(embed_dim)
        # Modify convolution layer parameters to downsample 800×800 images to 40×40
        self.conv_proj_q = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size_q, stride=stride_q, 
                                   padding=padding_q, bias=True)
        self.conv_proj_k = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size_kv, stride=stride_kv,
                                   padding=padding_kv, bias=True)
        self.conv_proj_v = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size_kv, stride=stride_kv,
                                   padding=padding_kv, bias=True)
        
        # Multi-head self-attention (output dimension = embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        x: Tensor [B, k, C_in, H, W], containing k images for each sample in the batch.
        Returns: Tensor [B, k, embed_dim, H_out, W_out], fused features with embed_dim channels for each image.
        """
        B, k_x, C, H, W = x.shape
        # Combine batch and image dimensions for convolution
        x_flat = x.view(B * k_x, C, H, W)           # [B*k, C_in, H, W]
        # Compute q, k, v using different convolution modules
        q = self.conv_proj_q(x_flat)            # [B*k, embed_dim, Hq, Wq]
        k = self.conv_proj_k(x_flat)            # [B*k, embed_dim, Hk, Wk]
        v = self.conv_proj_v(x_flat)            # [B*k, embed_dim, Hv, Wv]
        
        # Get feature map dimensions
        Hq, Wq = q.shape[2], q.shape[3]
        Hk, Wk = k.shape[2], k.shape[3]
        Hv, Wv = v.shape[2], v.shape[3]
        
        # Restore batch and image dimensions
        q = q.view(B, k_x, self.embed_dim, Hq, Wq)    # [B, k, embed_dim, Hq, Wq]
        k = k.view(B, k_x, self.embed_dim, Hk, Wk)    # [B, k, embed_dim, Hk, Wk]
        v = v.view(B, k_x, self.embed_dim, Hv, Wv)    # [B, k, embed_dim, Hv, Wv]
        
        # Flatten spatial dimensions, converting each image's patches to a sequence
        q = q.permute(0, 1, 3, 4, 2).contiguous().view(B, k_x * Hq * Wq, self.embed_dim)  # [B, Nq, embed_dim]
        k = k.permute(0, 1, 3, 4, 2).contiguous().view(B, k_x * Hk * Wk, self.embed_dim)  # [B, Nk, embed_dim]
        v = v.permute(0, 1, 3, 4, 2).contiguous().view(B, k_x * Hv * Wv, self.embed_dim)  # [B, Nv, embed_dim]
        
        # Transpose to format compatible with MultiheadAttention
        q = q.permute(1, 0, 2)  # [Nq, B, embed_dim]
        k = k.permute(1, 0, 2)  # [Nk, B, embed_dim]
        v = v.permute(1, 0, 2)  # [Nv, B, embed_dim]
        
        # Self-attention calculation using different q, k, v
        attn_out, _ = self.attention(q, k, v)
        
        # Convert back to [B, Nq, embed_dim]
        attn_out = attn_out.permute(1, 0, 2)  # [B, Nq, embed_dim]
        
        # LayerNorm normalization
        attn_out = self.norm(attn_out)
        
        # Reshape back to [B, k, embed_dim, Hq, Wq]
        attn_out = attn_out.view(B, k_x, Hq, Wq, attn_out.shape[-1]).permute(0, 1, 4, 2, 3).contiguous()
        return attn_out  # [B, k, embed_dim, Hq, Wq]
