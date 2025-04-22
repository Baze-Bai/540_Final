import torch
import torch.nn as nn
import torch.nn.functional as F
from Class_Func.MultiImage_Unet import MultiImageUNet

# ============================================================
# 1. Diffusion 主类：仅像素 L1 双尺度 loss
# ============================================================
class MultiImageDiffusion(nn.Module):
    """
    hr_weight : 高分辨率像素 L1 的权重（相对 800‑级 MSE/L1）
    """
    def __init__(self,
                 unet: nn.Module,
                 timesteps      : int   = 1000,
                 beta_start     : float = 1e-4,
                 beta_end       : float = 0.02,
                 hr_weight      : float = 0.1,
                 device         : str   = 'cuda'):
        super().__init__()
        self.unet      = unet.to(device)
        self.device    = torch.device(device)
        self.timesteps = timesteps
        self.hr_weight = hr_weight

        # ---------------- ① 预计算 β、α、ᾱ(t) ----------------
        betas          = torch.linspace(beta_start, beta_end,
                                        timesteps, device=device)
        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_cumprod',
                             torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1 - alphas_cumprod))

        # ---------------- ② 帧次位置编码 MLP ----------------
        C = unet.in_channels
        self.pos_mlp = nn.Sequential(
            nn.Linear(1, C),
            nn.ReLU(),
            nn.Linear(C, C)
        )

        # ---------------- ③ 高分辨率上采样模块 ----------------
        self.hr_upsampler = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        # ### <<< DEL >>>  删除感知损失 backbone（VGG）相关代码
        # self.vgg_feat ...

    # --------------------------------------------------------
    # 1‑based 帧号 → [1,L,C,1,1]
    # --------------------------------------------------------
    def _pos_encode(self, idx_1based: torch.LongTensor) -> torch.Tensor:
        emb = self.pos_mlp(idx_1based.float().unsqueeze(-1))
        return emb.view(1, len(idx_1based), -1, 1, 1)

    # --------------------------------------------------------
    # q(x_t | x_0)
    # --------------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a   = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_1_a = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1_a * noise

    # --------------------------------------------------------
    # 最近邻插值+卷积上采样到高分辨率
    # --------------------------------------------------------
    def upsample_to_hr(self, x, H_hr, W_hr):
        # 将输入重塑为4D张量用于插值和卷积
        B, kx, C, H, W = x.shape
        x_flat = x.view(-1, C, H, W)
        
        # 使用最近邻插值
        x_nearest = F.interpolate(
            x_flat, size=(H_hr, W_hr), mode='nearest'
        )
        
        # 通过卷积网络优化细节
        x_hr = self.hr_upsampler(x_nearest)
        
        # 恢复原始维度
        return x_hr.view(B, kx, C, H_hr, W_hr)

    # ========================================================
    # 双尺度 loss（高分辨率仅像素 L1）
    # ========================================================
    def p_losses(self,
                 x0_seq     : torch.Tensor,   # [B,kx,1,800,800]
                 ref_seq    : torch.Tensor,   # [B,kr,1,800,800]
                 pos_x      : torch.LongTensor,
                 x0_seq_hr  : torch.Tensor,   # [B,kx,1,868,1551]
                 t: torch.LongTensor = None):
        B, kx, C, H, W = x0_seq.shape
        H_hr, W_hr     = x0_seq_hr.shape[-2:]

        # ---------- 1) 参考帧顺序号 ----------
        kr      = ref_seq.shape[1]
        pos_all = torch.arange(1, kx + kr + 1, device=self.device)
        mask    = torch.ones_like(pos_all, dtype=torch.bool)
        mask[pos_x.to(self.device) - 1] = False
        pos_r   = pos_all[mask][:kr]

        # ---------- 2) 位置编码 ----------
        pe_x = self._pos_encode(pos_x.to(self.device))
        pe_r = self._pos_encode(pos_r)

        # ---------- 3) 随机时间步 ----------
        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=self.device)

        noise   = torch.randn_like(x0_seq)
        x_noisy = self.q_sample(x0_seq, t, noise)
        x_in    = x_noisy + pe_x
        r_in    = ref_seq + pe_r

        # ---------- 4) UNet 预测 ----------
        pred_noise = self.unet(x_in, r_in)

        # ---------- 5) 800‑级损失 ----------
        loss_800 = F.mse_loss(pred_noise, noise)

        # ---------- 6) 高分辨率像素 L1 损失 ----------
        # a) 反推 x0_pred
        sqrt_a   = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_1_a = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        x0_pred  = (x_noisy - sqrt_1_a * pred_noise) / sqrt_a  # [B,kx,1,800,800]

        # b) 使用最近邻插值+卷积上采样到 868×1551
        x0_pred_hr = self.upsample_to_hr(x0_pred, H_hr, W_hr)

        # c) 逐像素 L1
        x_pred_flat = x0_pred_hr.view(-1, 1, H_hr, W_hr)
        x_tgt_flat  = x0_seq_hr.to(self.device).view(-1, 1, H_hr, W_hr)
        loss_hr     = F.l1_loss(x_pred_flat, x_tgt_flat)       # 逐像素L1损失

        # ---------- 7) 总损失 ----------
        return loss_800 + self.hr_weight * loss_hr

    # ========================================================
    # DDIM 采样：修改为返回高分辨率输出
    # ========================================================
    @torch.no_grad()
    def ddim_sample(self,
                    shape   : tuple,            # (B,kx,1,800,800)
                    ref_seq : torch.Tensor,     # [B,kr,1,800,800]
                    steps   : int = 50,
                    eta     : float = 0.0,
                    hr_output: bool = True,     # 是否输出高分辨率结果
                    hr_shape: tuple = (1551, 868)): # 高分辨率输出尺寸
        B, kx, C, H, W = shape
        device = self.device
        time_seq = torch.linspace(self.timesteps - 1, 0, steps,
                                  device=device).long()

        x = torch.randn(shape, device=device)
        abar  = self.sqrt_alphas_cumprod ** 2
        sqrt_abar     = self.sqrt_alphas_cumprod
        sqrt_one_abar = self.sqrt_one_minus_alphas_cumprod

        for i in time_seq:
            t = torch.full((B,), i, dtype=torch.long, device=device)
            e_t = self.unet(x, ref_seq)

            a_t      = abar[i]
            sqrt_a_t = sqrt_abar[i]
            sqrt1_a_t= sqrt_one_abar[i]
            x0_pred  = (x - sqrt1_a_t * e_t) / sqrt_a_t

            a_prev = abar[i-1] if i > 0 else torch.tensor(1.0, device=device)
            sigma  = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            z      = torch.randn_like(x) if i > 0 else 0.0
            x = torch.sqrt(a_prev) * x0_pred \
              + torch.sqrt(1 - a_prev - sigma**2) * e_t \
              + sigma * z
              
        # 返回最终生成的结果
        if hr_output:
            # 输出高分辨率结果 (868×1551)
            H_hr, W_hr = hr_shape
            x_hr = self.upsample_to_hr(x, H_hr, W_hr)
            # 返回:
            # x: 低分辨率结果，维度 [B,kx,1,800,800] - (批次,帧数,通道,高,宽)
            # x_hr: 高分辨率结果，维度 [B,kx,1,868,1551] - (批次,帧数,通道,高,宽)
            return x, x_hr
        else:
            # 返回: x: 低分辨率结果，维度 [B,kx,1,800,800] - (批次,帧数,通道,高,宽)
            return x
            
    # ========================================================
    # 推理函数：接收参考帧和输入帧位置，生成结果
    # ========================================================
    @torch.no_grad()
    def inference(self, 
                  x0_seq: torch.Tensor,      # [B,kx,1,800,800] 输入帧序列
                  ref_seq: torch.Tensor,      # [B,kr,1,800,800] 参考帧序列
                  pos_x: torch.LongTensor,    # 输入帧的位置索引 (1-based)
                  ddim_steps: int = 50,       # DDIM采样步数
                  hr_output: bool = True,     # 是否输出高分辨率结果
                  hr_shape: tuple = (1551, 868), # 高分辨率输出尺寸
                  eta: float = 0.0):          # DDIM噪声参数
        """
        推理函数：使用训练好的模型生成结果
        
        参数:
            x0_seq: 输入帧序列，形状 [B,kx,1,800,800]
            ref_seq: 参考帧序列，形状 [B,kr,1,800,800]
            pos_x: 输入帧的位置索引 (1-based)
            ddim_steps: DDIM采样步数
            hr_output: 是否输出高分辨率结果
            hr_shape: 高分辨率输出尺寸 (默认 1551×868)
            eta: DDIM采样噪声系数
            
        返回:
            如果hr_output=True: 返回(lr_result, hr_result)
                - lr_result: 低分辨率结果，[B,kx,1,800,800]
                - hr_result: 高分辨率结果，[B,kx,1,hr_shape[0],hr_shape[1]]
            如果hr_output=False: 仅返回lr_result
        """
        self.eval()  # 设置为评估模式
        B, kx, C, H, W = x0_seq.shape
        device = self.device
        
        # 1) 获取参考帧位置编码
        kr = ref_seq.shape[1]
        pos_all = torch.arange(1, kx + kr + 1, device=device)
        mask = torch.ones_like(pos_all, dtype=torch.bool)
        mask[pos_x.to(device) - 1] = False
        pos_r = pos_all[mask][:kr]
        
        # 2) 应用位置编码
        pe_r = self._pos_encode(pos_r)
        r_in = ref_seq.to(device) + pe_r
        
        # 3) 使用DDIM采样生成结果
        shape = (B, kx, C, H, W)
        
        return self.ddim_sample(
            shape=shape,
            ref_seq=r_in,
            steps=ddim_steps,
            eta=eta,
            hr_output=hr_output,
            hr_shape=hr_shape
        )