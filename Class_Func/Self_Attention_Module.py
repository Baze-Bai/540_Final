import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttCvT(nn.Module):
    def __init__(self, in_channels=1, embed_dim=8, num_heads=2, kernel_size_q=22, stride_q=20, padding_q=1, 
                 kernel_size_kv=22, stride_kv=20, padding_kv=1):
        """
        CvT风格自注意模块：将输入的k张图像特征进行自注意力融合。
        Args:
            in_channels: 输入通道数（灰度图为1）。
            embed_dim: 注意力嵌入维度（例如32）。
            num_heads: 注意力头数。
            q,k,v的参数都固定
        """
        super(SelfAttCvT, self).__init__()
        # 卷积投影，将图像映射为embed_dim维度特征,输出维度为embed_dim
        self.embed_dim = int(embed_dim)
        # 修改卷积层参数以将800×800的图像降采样到40×40
        self.conv_proj_q = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size_q, stride=stride_q, 
                                   padding=padding_q, bias=True)
        self.conv_proj_k = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size_kv, stride=stride_kv,
                                   padding=padding_kv, bias=True)
        self.conv_proj_v = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size_kv, stride=stride_kv,
                                   padding=padding_kv, bias=True)
        
        # 多头自注意力（输出维度=embed_dim）
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        # 输出归一化
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        x: Tensor [B, k, C_in, H, W]，包含批次中每个样本的k张图像。
        返回: Tensor [B, k, embed_dim, H_out, W_out]，每张图像通道维度为embed_dim的融合特征。
        """
        B, k_x, C, H, W = x.shape
        # 将batch和图像维度合并，方便卷积
        x_flat = x.view(B * k_x, C, H, W)           # [B*k, C_in, H, W]
        # 分别使用不同的卷积模块计算q,k,v
        q = self.conv_proj_q(x_flat)            # [B*k, embed_dim, Hq, Wq]
        k = self.conv_proj_k(x_flat)            # [B*k, embed_dim, Hk, Wk]
        v = self.conv_proj_v(x_flat)            # [B*k, embed_dim, Hv, Wv]
        
        # 获取特征图尺寸
        Hq, Wq = q.shape[2], q.shape[3]
        Hk, Wk = k.shape[2], k.shape[3]
        Hv, Wv = v.shape[2], v.shape[3]
        
        # 恢复batch和图像维度
        q = q.view(B, k_x, self.embed_dim, Hq, Wq)    # [B, k, embed_dim, Hq, Wq]
        k = k.view(B, k_x, self.embed_dim, Hk, Wk)    # [B, k, embed_dim, Hk, Wk]
        v = v.view(B, k_x, self.embed_dim, Hv, Wv)    # [B, k, embed_dim, Hv, Wv]
        
        # 展开空间维度，将每张图像的patch flatten为序列
        q = q.permute(0, 1, 3, 4, 2).contiguous().view(B, k_x * Hq * Wq, self.embed_dim)  # [B, Nq, embed_dim]
        k = k.permute(0, 1, 3, 4, 2).contiguous().view(B, k_x * Hk * Wk, self.embed_dim)  # [B, Nk, embed_dim]
        v = v.permute(0, 1, 3, 4, 2).contiguous().view(B, k_x * Hv * Wv, self.embed_dim)  # [B, Nv, embed_dim]
        
        # 转置为适配MultiheadAttention的格式
        q = q.permute(1, 0, 2)  # [Nq, B, embed_dim]
        k = k.permute(1, 0, 2)  # [Nk, B, embed_dim]
        v = v.permute(1, 0, 2)  # [Nv, B, embed_dim]
        
        # 自注意力计算，使用不同的q,k,v
        attn_out, _ = self.attention(q, k, v)
        
        # 转回 [B, Nq, embed_dim]
        attn_out = attn_out.permute(1, 0, 2)  # [B, Nq, embed_dim]
        
        # LayerNorm归一化
        attn_out = self.norm(attn_out)
        
        # 重塑回 [B, k, embed_dim, Hq, Wq]
        attn_out = attn_out.view(B, k_x, Hq, Wq, attn_out.shape[-1]).permute(0, 1, 4, 2, 3).contiguous()
        return attn_out  # [B, k, embed_dim, Hq, Wq]
