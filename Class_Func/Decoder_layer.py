import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        """
        解码器层：上采样 + 卷积
        Args:
            in_channels: 输入通道数（来自更深层）。
            out_channels: 输出通道数。
            skip_channels: 若使用skip连接，要拼接的skip特征通道数（默认为0表示无skip）。
        """
        super(DecoderLayer, self).__init__() 
        
        # 上采样（转置卷积，使尺寸翻倍）
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 拼接后输入通道 = out_channels + skip_channels（若有skip），经过两层卷积恢复out_channels通道
        in_conv1 = out_channels + skip_channels if skip_channels > 0 else out_channels
        self.conv1 = nn.Conv2d(in_conv1, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, skip_feat=None):
        """
        x: [B, k, C, H, W] 来自下一级的特征图，B为批次大小，k为图片数量
        skip_feat: [B, k, C_skip, H*2, W*2] 对应的编码器层skip特征
        返回: 上采样并卷积融合后的特征 [B, k, out_channels, H*2, W*2]
        """
        B, k, C, H, W = x.shape
        x = x.view(B*k, C, H, W)  # 重塑为[B*k, C, H, W]以适应2D卷积
        
        x_up = self.upconv(x)  # 上采样 (H,W -> 2H,2W)
        
        if skip_feat is not None:
            # 重塑skip_feat以适应拼接
            _, _, C_skip, H2, W2 = skip_feat.shape
            skip_feat = skip_feat.view(B*k, C_skip, H2, W2)
            x_up = torch.cat([x_up, skip_feat], dim=1)  # 通道维度拼接skip特征
            
        h = F.relu(self.conv1(x_up))
        out = F.relu(self.conv2(h))
        
        # 恢复原始批次维度 [B*k, C_out, H*2, W*2] -> [B, k, C_out, H*2, W*2]
        out = out.view(B, k, out.size(1), out.size(2), out.size(3))
        
        return out