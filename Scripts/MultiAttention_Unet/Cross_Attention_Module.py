import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self,  embed_dim=None, num_heads=2, kernel_size_q=22, stride_q=20, padding_q=1, 
                 kernel_size_kv=22, stride_kv=20, padding_kv=1, kv_length=40, in_channels=1):
        """
        Cross-Attention Module: Computes query from input x, key and value from reference image r_image,
        performs multi-head attention fusion, and uses 1x1 convolution to reduce dimensions for the fused features.
        
        Args:
            embed_dim: Projection dimension for input features
            num_heads: Number of attention heads
            kernel_size_q: Kernel size for query projection
            stride_q: Stride for query projection
            padding_q: Padding for query projection
            kernel_size_kv: Kernel size for key-value projection
            stride_kv: Stride for key-value projection
            padding_kv: Padding for key-value projection
            kv_length: Length parameter for key-value sequence
            in_channels: Number of input channels per image
        """
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_length = kv_length
        self.in_channels = in_channels

        # Modify q_proj to accept single-channel input and downsample 800x800 to 40x40 (reduce by 20x)
        self.q_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size_q, stride=stride_q, padding=padding_q)
        # Define convolution with specified kernel size, stride, and padding
        self.kv_proj = nn.Conv2d(1, embed_dim, kernel_size=kernel_size_kv, stride=stride_kv, padding=padding_kv)

        # Multi-head attention module, requires input shape [seq_len, batch, embed_dim]
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # LayerNorm for normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, r_image):
        """
        Args:
            x: [B, k, in_channels, H, W] - Input feature set, each sample has k images,
                                          each image has in_channels channels
            r_image: [B, k_r, 1, H_r, W_r] - Reference image features, each sample has k_r reference images,
                                            all single-channel.
        Returns:
            combined: [B, k, C*embed_dim, H_q, W_q] - Applies cross-attention query to each channel of each image in x
                                                     separately, reduces dimensions, and restructures to original shape
        """
        # Get shape information
        B, k_x, C, H_x, W_x = x.shape  # Get dimensions of x, each image in x has C channels, size H_x x W_x
        B_r, k_r, C_r, H_r, W_r = r_image.shape  # Get dimensions of r_image
        assert B == B_r, "Batch size of x and r_image must be the same"
        assert C == self.in_channels, "Number of channels in x must equal in_channels"
        assert C_r == 1, "Reference images must be single-channel"

        # ---------------------------
        # Process reference images r_image (source of key and value)
        r_image_flat = r_image.view(B * k_r, 1, H_r, W_r)
        r_feat = self.kv_proj(r_image_flat)  # r_feat has shape [B*k_r, embed_dim, kv_length, kv_length]
        r_feat = r_feat.view(B, k_r, self.embed_dim, self.kv_length, self.kv_length)
        
        # Convert r_feat to sequence format
        kv_tokens = r_feat.permute(0, 1, 3, 4, 2).contiguous().view(B, k_r * self.kv_length * self.kv_length, self.embed_dim)
        kv_seq = kv_tokens.permute(1, 0, 2)  # kv_seq has shape [k_r*kv_length*kv_length, B, embed_dim]
        # Note: kv_seq already meets the required dimensions [k*kv_length*kv_length, B, embed_dim]

        # ---------------------------
        # Prepare to process all images
        x_reshaped = x.view(B * k_x, C, H_x, W_x)
        
        # Use grouped convolution to process all channels simultaneously and get query features
        q_feat = self.q_proj(x_reshaped)  # [B*k_x, C*embed_dim, H_q, W_q]
        
        # Get feature map dimensions after convolution
        H_q, W_q = q_feat.size(2), q_feat.size(3)
        
        # Reshape to [B, k_x, C, embed_dim, H_q, W_q]
        q_feat = q_feat.view(B, k_x, C, self.embed_dim, H_q, W_q)
        
        # Modification: Reshape q_feat to [C*H_q*W_q*k_x, B, embed_dim]
        # First, swap dimension order to place k_x dimension at the innermost position
        q_feat = q_feat.permute(0, 2, 4, 5, 1, 3).contiguous()  # [B, C, H_q, W_q, k_x, embed_dim]
        
        # Then combine C, H_q, W_q, k_x into a single dimension
        q_seq = q_feat.view(B, C*H_q*W_q*k_x, self.embed_dim).permute(1, 0, 2)  # [C*H_q*W_q*k_x, B, embed_dim]
        
        # Perform multi-head attention calculation in parallel
        # Now q_seq has shape [C*H_q*W_q*k_x, B, embed_dim], kv_seq has shape [k_r*kv_length*kv_length, B, embed_dim]
        attn_out, _ = self.attention(q_seq, kv_seq, kv_seq)  
        # attn_out shape: [C*H_q*W_q*k_x, B, embed_dim]
        
        # LayerNorm normalization
        attn_out = self.norm(attn_out.permute(1, 0, 2))  # [B, C*H_q*W_q*k_x, embed_dim]
        
        # Reshape attn_out to [B, k_x, C*embed_dim, H_q, W_q]
        attn_out = attn_out.view(B, C, H_q, W_q, k_x, self.embed_dim)  # [B, C, H_q, W_q, k_x, embed_dim]
        attn_out = attn_out.permute(0, 4, 1, 5, 2, 3).contiguous()  # [B, k_x, C, embed_dim, H_q, W_q]
        attn_out = attn_out.view(B, k_x, C*self.embed_dim, H_q, W_q)  # [B, k_x, C*embed_dim, H_q, W_q]

        return attn_out