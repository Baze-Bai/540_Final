import torch
import torch.nn as nn
import torch.nn.functional as F
from Class_Func.Cross_Attention_Module import CrossAttention


# 编写 EncoderLayer 模块，调用 CrossAttention 后对其输出进行卷积处理
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, embed_dim=None, num_heads=2, kernel_size_q=20, stride_q=20, padding_q=10, 
                 kernel_size_kv=3, stride_kv=1, padding_kv=1,kv_length=50, 
                 kernel_size_t=4, stride_t=2, padding_t=1):
        """
        参数:
            in_channels: 输入图片的通道数
            embed_dim: 用于 CrossAttention 投影的维度
            num_heads: 多头注意力头数
            kernel_size_kv: 卷积核大小
            stride_kv: 卷积步幅
            padding_kv: 卷积填充
        """
        
        super(EncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # 下采样层 - 将输入图像的尺寸减半
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 通道转换层 - 将通道数从C变为1
        self.channel_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # 跨注意力模块
        self.cross_att = CrossAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                        kernel_size_q=kernel_size_q, stride_q=stride_q, padding_q=padding_q,
                                       kernel_size_kv=kernel_size_kv, stride_kv=stride_kv, padding_kv=padding_kv, kv_length=kv_length,
                                       in_channels=1)  # 注意这里输入通道数为1
        
        # 通道恢复和上采样层 - 将通道数从embed_dim恢复到C，同时上采样到原来大小的一半(H/2, W/2)
        # 使用转置卷积实现上采样和通道转换
        self.restore_conv = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=kernel_size_t, stride=stride_t, padding=padding_t)
        
        # 最后一层卷积层 - 保持图片大小不变，将通道数翻倍
        self.final_conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1)
        
        # 可选：添加激活函数
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x, r_image):
        """
        参数:
            x:       [B, k, C, H, W] - B为批次大小，k为图片数量，C为通道数量，H和W为图片尺寸
            r_image: [B, k, 1, H, W] - 参考图像，单通道
        处理流程：
            1. 将x下采样到原来大小的一半；
            2. 通过卷积层将通道数C变为1；
            3. 将处理后的特征通过CrossAttention，得到attn_out；
            4. 将attn_out通过转置卷积恢复到原始通道数C，同时上采样到H/2和W/2；
            5. 与下采样后的特征相加；
            6. 通过最后一层卷积层，保持图片大小不变，将通道数翻倍。
        """
        # 获取输入维度
        B, k, C, H, W = x.shape
        
        # 步骤1：下采样到原来大小的一半
        # 首先将形状调整为[B*k, C, H, W]以便于应用下采样
        x_reshaped = x.view(B*k, C, H, W)
        downsampled = self.downsample(x_reshaped)  # [B*k, C, H/2, W/2]
        
        # 获取下采样后的图像尺寸
        _, _, H_d, W_d = downsampled.shape
        
        # 保存下采样结果，用于后续的跳跃连接
        skip_connection = downsampled.clone()
        
        # 步骤2：通过卷积层将通道数C变为1
        x_single_channel = self.channel_conv(downsampled)  # [B*k, 1, H/2, W/2]
        
        # 将形状调整回[B, k, 1, H/2, W/2]，以便传入CrossAttention
        x_prepared = x_single_channel.view(B, k, 1, H_d, W_d)
        
        # 步骤3：传入CrossAttention，得到attn_out
        # r_image不需要下采样，直接传入CrossAttention
        
        # 调用CrossAttention，输出形状为 [B, k, embed_dim, H_out, W_out]
        attn_out = self.cross_att(x_prepared, r_image)
        
        # 步骤4：将attn_out通过转置卷积恢复到原始通道数C并上采样
        # 获取attn_out的输出维度
        _, _, embed_out, H_out, W_out = attn_out.shape
        
        # 将形状调整为[B*k, embed_dim, H_out, W_out]以便于应用转置卷积
        attn_out_reshaped = attn_out.view(B*k, embed_out, H_out, W_out)
        
        # 通过转置卷积恢复通道数并上采样，输出维度为[B*k, C, H/2, W/2]
        restored_features = self.restore_conv(attn_out_reshaped)
        
        # 确保上采样后的尺寸与skip_connection一致
        if restored_features.shape[1:] != skip_connection.shape[1:]:
            restored_features = F.interpolate(restored_features, size=(H_d, W_d), mode='bilinear', align_corners=False)
        
        # 步骤5：将恢复后的特征与下采样特征相加（跳跃连接）
        output = restored_features + skip_connection
        
        # 应用激活函数
        output = self.act(output)
        
        # 步骤6：通过最后一层卷积层，保持图片大小不变，将通道数翻倍
        output_reshaped = output.view(B*k, C, H_d, W_d)
        output_doubled = self.final_conv(output_reshaped)
        
        # 将输出形状调整回[B, k, C*2, H/2, W/2]
        output = output_doubled.view(B, k, C*2, H_d, W_d)
        
        return output
    