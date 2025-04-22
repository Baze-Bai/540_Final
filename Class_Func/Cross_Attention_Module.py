import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self,  embed_dim=None, num_heads=2, kernel_size_q=22, stride_q=20, padding_q=1, 
                 kernel_size_kv=22, stride_kv=20, padding_kv=1, kv_length=40, in_channels=1):
        """
        跨注意模块：基于输入 x 计算 query，基于参考图 r_image 计算 key 与 value，
        进行多头注意力融合后，利用 1x1 卷积降维得到融合特征。
        参数：
            embed_dim: 输入特征的投影维度
            num_heads: 多头注意力的头数
            kernel_size_kv: KV投影的卷积核大小
            stride_kv: KV投影的卷积步长
            padding_kv: KV投影的卷积填充
            kv_length: KV序列的长度参数
            in_channels: 输入通道数，每张图片的通道数
        """
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_length = kv_length
        self.in_channels = in_channels

        # 修改q_proj接收单通道输入，并将800x800降采样到40x40（缩小20倍）
        self.q_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size_q, stride=stride_q, padding=padding_q)
        self.kv_proj = nn.Conv2d(1, embed_dim, kernel_size=kernel_size_kv, stride=stride_kv, padding=padding_kv) # 制定卷积核大小，步长，填充

        # 多头注意力模块，要求输入形状为 [seq_len, batch, embed_dim]
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # LayerNorm归一化操作
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, r_image):
        """
        参数:
            x:       [B, k, in_channels, H, W] - 输入特征集合，每个样本中有k张图片，
                                            每张图片有in_channels个通道
            r_image: [B, k_r, 1, H_r, W_r] - 参考图特征，每个样本中有k_r张参考图片，均为单通道。
        返回:
            combined:  [B, k, 4*in_channels, H_q, W_q] - 对x中每一张图片的每个通道单独执行跨注意查询
                                               并降维，最后将结果重组为原始形状
        """
        # 获取形状信息
        B, k_x, C, H_x, W_x = x.shape  # 获取x的维度, x中每张图片的通道数为C，尺寸为H_x x W_x
        B_r, k_r, C_r, H_r, W_r = r_image.shape # 获取r_image的维度
        assert B == B_r, "x和r_image的批量大小必须一致"
        assert C == self.in_channels, "输入x的通道数必须等于in_channels"
        assert C_r == 1, "参考图必须是单通道图像"

        # ---------------------------
        # 处理参考图 r_image（key 与 value 的来源）
        r_image_flat = r_image.view(B * k_r, 1, H_r, W_r)
        r_feat = self.kv_proj(r_image_flat)  # r_feat的维度为[B*k_r, embed_dim, kv_length, kv_length]
        r_feat = r_feat.view(B, k_r, self.embed_dim, self.kv_length, self.kv_length)
        
        # 将r_feat转换为序列格式
        kv_tokens = r_feat.permute(0, 1, 3, 4, 2).contiguous().view(B, k_r * self.kv_length * self.kv_length, self.embed_dim)
        kv_seq = kv_tokens.permute(1, 0, 2)  # kv_seq的维度为[k_r*kv_length*kv_length, B, embed_dim]
        # 注意：kv_seq已经满足要求的维度 [k*kv_length*kv_length, B, embed_dim]

        # ---------------------------
        # 准备处理所有图片
        x_reshaped = x.view(B * k_x, C, H_x, W_x)
        
        # 使用分组卷积同时处理所有通道，得到query特征
        q_feat = self.q_proj(x_reshaped)  # [B*k_x, C*embed_dim, H_q, W_q]
        
        # 获取卷积后的特征图尺寸
        H_q, W_q = q_feat.size(2), q_feat.size(3)
        
        # 重塑为[B, k_x, C, embed_dim, H_q, W_q]
        q_feat = q_feat.view(B, k_x, C, self.embed_dim, H_q, W_q)
        
        # 修改：将q_feat重组为[C*H_q*W_q*k_x, B, embed_dim]
        # 首先交换维度顺序，使k_x维度位于最内部
        q_feat = q_feat.permute(0, 2, 4, 5, 1, 3).contiguous()  # [B, C, H_q, W_q, k_x, embed_dim]
        
        # 然后将C, H_q, W_q, k_x合并为一个维度
        q_seq = q_feat.view(B, C*H_q*W_q*k_x, self.embed_dim).permute(1, 0, 2)  # [C*H_q*W_q*k_x, B, embed_dim]
        
        # 并行执行多头注意力计算
        # 现在q_seq维度为[C*H_q*W_q*k_x, B, embed_dim]，kv_seq维度为[k_r*kv_length*kv_length, B, embed_dim]
        attn_out, _ = self.attention(q_seq, kv_seq, kv_seq)  
        # attn_out维度: [C*H_q*W_q*k_x, B, embed_dim]
        
        # LayerNorm归一化
        attn_out = self.norm(attn_out.permute(1, 0, 2))  # [B, C*H_q*W_q*k_x, embed_dim]
        
        # 将attn_out重塑为[B, k_x, C*embed_dim, H_q, W_q]
        attn_out = attn_out.view(B, C, H_q, W_q, k_x, self.embed_dim)  # [B, C, H_q, W_q, k_x, embed_dim]
        attn_out = attn_out.permute(0, 4, 1, 5, 2, 3).contiguous()  # [B, k_x, C, embed_dim, H_q, W_q]
        attn_out = attn_out.view(B, k_x, C*self.embed_dim, H_q, W_q)  # [B, k_x, C*embed_dim, H_q, W_q]

        return attn_out