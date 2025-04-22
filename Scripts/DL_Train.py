import torch
import torch.nn as nn
import torch.nn.functional as F
from Class_Func.MultiImage_Unet import MultiImageUNet
from Class_Func.MultiImage_Diffusion import MultiImageDiffusion
from Scripts.Data_preprocess import process_images_into_sequences, plot_losses, create_dataloaders
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Class_Func.Unet_Model import UnetModel


# Process images into sequences
reference_images, sequence_images = process_images_into_sequences(image_dir='Dataset_images')

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    reference_images, 
    sequence_images,
    batch_size=2, 
    train_ratio=0.8, 
    val_ratio=0.1, 
    test_ratio=0.1, 
    shuffle=True, 
    num_workers=0
)

# 初始化UnetModel模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UnetModel(in_channels=1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练参数
num_epochs = 50
train_losses = []
val_losses = []

# 训练循环
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    
    for i, (ref_images, seq_images) in enumerate(train_loader):
        # 确保数据在正确的设备上
        ref_images = ref_images.to(device)  # [B, 6, 1, 1551, 868]
        seq_images = seq_images.to(device)  # [B, 5, 1, 1551, 868]
        
        # 保存原始高分辨率的seq_images用于计算损失
        original_seq_images = seq_images.clone()
        
        # 获取当前形状
        B = ref_images.size(0)  # 批量大小
        ref_k = ref_images.size(1)  # 参考图像的数量 k=6
        seq_k = seq_images.size(1)  # 序列图像的数量 k=5
        
        # 使用F.interpolate将图像从[B,k,1,1551,868]调整为[B,k,1,800,800]
        # 首先重塑张量以便按照每个图像进行处理
        ref_images_flat = ref_images.view(B * ref_k, 1, 1551, 868)
        
        # 调整大小
        ref_images_resized = F.interpolate(ref_images_flat, size=(800, 800), mode='bilinear', align_corners=False)
        
        # 重塑回原始批大小和序列长度
        ref_images = ref_images_resized.view(B, ref_k, 1, 800, 800)
        
        # 创建一个形状为[B, 5, 1, 800, 800]的零张量作为输入
        # 这里我们不使用真实的seq_images作为输入
        input_shape = (B, seq_k, 1, 800, 800)
        dummy_input = torch.zeros(input_shape, device=device)
        
        # 前向传播 - 只使用ref_images和dummy_input
        outputs = model(dummy_input, ref_images)  # 输出为 [B, 5, 1, 1551, 868]
        
        # 计算损失 (预测序列图像与真实序列图像之间的MSE)
        # 使用原始高分辨率的seq_images计算损失
        loss = criterion(outputs, original_seq_images)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        
        if (i+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # 计算平均训练损失
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for ref_images, seq_images in val_loader:
            ref_images = ref_images.to(device)  # [B, 6, 1, 1551, 868]
            seq_images = seq_images.to(device)  # [B, 5, 1, 1551, 868]
            
            # 保存原始高分辨率的seq_images用于计算损失
            original_seq_images = seq_images.clone()
            
            # 调整验证集数据大小
            B = ref_images.size(0)
            ref_k = ref_images.size(1)
            seq_k = seq_images.size(1)
            
            ref_images_flat = ref_images.view(B * ref_k, 1, 1551, 868)
            
            ref_images_resized = F.interpolate(ref_images_flat, size=(800, 800), mode='bilinear', align_corners=False)
            
            ref_images = ref_images_resized.view(B, ref_k, 1, 800, 800)
            
            # 创建一个形状为[B, 5, 1, 800, 800]的零张量作为输入
            input_shape = (B, seq_k, 1, 800, 800)
            dummy_input = torch.zeros(input_shape, device=device)
            
            # 前向传播 - 只使用ref_images和dummy_input
            outputs = model(dummy_input, ref_images)  # 输出为 [B, 5, 1, 1551, 868]
            
            # 使用原始高分辨率的seq_images计算损失
            val_loss = criterion(outputs, original_seq_images)
            epoch_val_loss += val_loss.item()
    
    # 计算平均验证损失
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # 打印结果
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # 绘制损失曲线
    plot_losses(train_losses, val_losses, epoch)
    
    # 每10个epoch保存模型
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'models/unet_model_epoch_{epoch+1}.pth')

print('训练完成!')

# 在测试集上评估模型
model.eval()
test_loss = 0.0
with torch.no_grad():
    for ref_images, seq_images in test_loader:
        ref_images = ref_images.to(device)  # [B, 6, 1, 1551, 868]
        seq_images = seq_images.to(device)  # [B, 5, 1, 1551, 868]
        
        # 保存原始高分辨率的seq_images用于计算损失
        original_seq_images = seq_images.clone()
        
        # 调整测试集数据大小
        B = ref_images.size(0)
        ref_k = ref_images.size(1)
        seq_k = seq_images.size(1)
        
        ref_images_flat = ref_images.view(B * ref_k, 1, 1551, 868)
        
        ref_images_resized = F.interpolate(ref_images_flat, size=(800, 800), mode='bilinear', align_corners=False)
        
        ref_images = ref_images_resized.view(B, ref_k, 1, 800, 800)
        
        # 创建一个形状为[B, 5, 1, 800, 800]的零张量作为输入
        input_shape = (B, seq_k, 1, 800, 800)
        dummy_input = torch.zeros(input_shape, device=device)
        
        outputs = model(dummy_input, ref_images)  # 输出为 [B, 5, 1, 1551, 868]
        
        # 使用原始高分辨率的seq_images计算损失
        loss = criterion(outputs, original_seq_images)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')

# 保存最终模型
torch.save(model.state_dict(), 'models/unet_model_final.pth')

