import torch
import torch.nn as nn 
import torch.nn.functional as F
from Scripts.MultiAttention_Unet.Encoder_layer import EncoderLayer
from Scripts.MultiAttention_Unet.Decoder_layer import DecoderLayer
from Scripts.MultiAttention_Unet.Self_Attention_Module import SelfAttCvT
from Scripts.MultiAttention_Unet.Cross_Attention_Module import CrossAttention

class MultiImageUNet(nn.Module):
    def __init__(self, in_channels=1):
        super(MultiImageUNet, self).__init__()
        
        self.in_channels = in_channels
        # Self-attention layer
        self.self_attention = SelfAttCvT()  # Using default parameters
        
        # Upsampling convolution layer after self-attention
        self.upconv = nn.ConvTranspose2d(8, 1, kernel_size=22, stride=20, padding=1)
        
        # First convolution layer after self-attention (maintains dimensions)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        
        # Encoder layers
        self.encoder1 = EncoderLayer(in_channels=32, num_heads=2, embed_dim=2, kernel_size_q=12, stride_q=10, padding_q=1, 
                                     kernel_size_kv=82, stride_kv=80, padding_kv=1, kv_length=10,
                                     kernel_size_t=12, stride_t=10, padding_t=0)
        self.encoder2 = EncoderLayer(in_channels=64, num_heads=2, embed_dim=4, kernel_size_q=7, stride_q=5, padding_q=1, 
                                     kernel_size_kv=42, stride_kv=40, padding_kv=1, kv_length=20,
                                     kernel_size_t=5, stride_t=5, padding_t=0)
        self.encoder3 = EncoderLayer(in_channels=128, num_heads=2, embed_dim=8, kernel_size_q=24, stride_q=2, padding_q=1, 
                                     kernel_size_kv=22, stride_kv=20, padding_kv=1, kv_length=40,
                                     kernel_size_t=22, stride_t=2, padding_t=0)
        self.encoder4 = EncoderLayer(in_channels=256, num_heads=2, embed_dim=8, kernel_size_q=13, stride_q=1, padding_q=1, 
                                     kernel_size_kv=22, stride_kv=20, padding_kv=1, kv_length=40,
                                     kernel_size_t=11, stride_t=1, padding_t=0)
        
        # Middle convolution layers
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # Decoder layers
        self.decoder1 = DecoderLayer(in_channels=512, out_channels=256, skip_channels=256)
        self.decoder2 = DecoderLayer(in_channels=256, out_channels=128, skip_channels=128)
        self.decoder3 = DecoderLayer(in_channels=128, out_channels=64, skip_channels=64)
        self.decoder4 = DecoderLayer(in_channels=64, out_channels=32, skip_channels=32)
        
        # Final convolution layer
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x_images, reference_images):
        """
        Args:
            x_images: [B, k, 1, H, W] - Batch of input images
            reference_images: [B, k, 1, H, W] - Used for reference and consistency.
        
        Returns:
            output: [B, k, H, W] - Batch of output images
        """
        B, k, C, H, W = x_images.shape
        
        # Step 1: Self-attention layer
        x_att = self.self_attention(x_images)  # [B, k, 8, H, W]
        B_att, k_att, C_att, H_att, W_att = x_att.shape
        # Apply upsampling convolution to input images
        x_up = x_att.view(B_att * k_att, C_att, H_att, W_att)
        x_up = self.upconv(x_up)  # [B*k, 1, H, W]
        x_up = x_up.view(B, k, C, H, W)
        x_up = x_up + x_images
        
        # Step 2: First convolution layer
        B_att, k_att, C_att, H_att, W_att = x_up.shape
        x_conv1 = x_up.view(B_att * k_att, C_att, H_att, W_att)
        x_conv1 = self.conv1(x_conv1)
        x_conv1 = x_conv1.view(B_att, k_att, 32, H_att, W_att)
        
        # Step 3: Encoder path
        x_enc1 = self.encoder1(x_conv1, reference_images)  # [B, k, 64, H/2, W/2]
        x_enc2 = self.encoder2(x_enc1, reference_images)   # [B, k, 128, H/4, W/4]
        x_enc3 = self.encoder3(x_enc2, reference_images)   # [B, k, 256, H/8, W/8]
        x_enc4 = self.encoder4(x_enc3, reference_images)   # [B, k, 512, H/16, W/16]
        
        # Step 4: Middle convolution layer
        B_enc, k_enc, C_enc, H_enc, W_enc = x_enc4.shape
        x_mid = x_enc4.view(B_enc * k_enc, C_enc, H_enc, W_enc)
        x_mid = self.conv2(x_mid)  # [B*k, 512, H/16, W/16]
        x_mid = x_mid.view(B_enc, k_enc, C_enc, H_enc, W_enc)
        
        # Step 5: Decoder path with skip connections
        x_dec1 = self.decoder1(x_mid, x_enc3)    # [B, k, 256, H/8, W/8]
        x_dec2 = self.decoder2(x_dec1, x_enc2)   # [B, k, 128, H/4, W/4]
        x_dec3 = self.decoder3(x_dec2, x_enc1)   # [B, k, 64, H/2, W/2]
        x_dec4 = self.decoder4(x_dec3, x_conv1)   # [B, k, 32, H, W]
        
        # Step 6: Final convolution layer
        B_dec, k_dec, C_dec, H_dec, W_dec = x_dec4.shape
        x_out = x_dec4.view(B_dec * k_dec, C_dec, H_dec, W_dec)
        x_out = self.conv3(x_out)  # [B*k, 1, H, W]
        # Reshape to [B, k, 1, H, W] format
        output = x_out.view(B_dec, k_dec, 1, H_dec, W_dec)
        
        return output