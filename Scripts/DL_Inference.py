import torch
import torch.nn.functional as F
from Class_Func.Unet_Model import UnetModel
from Scripts.Data_preprocess import process_images_into_sequences, create_dataloaders
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 载入模型
model = UnetModel(in_channels=1).to(device)
model.load_state_dict(torch.load('models/unet_model_final.pth', map_location=device))
model.eval()
print("Model loaded successfully!")

# 获取测试数据
reference_images, sequence_images = process_images_into_sequences(image_dir='Dataset_images')
_, _, test_loader = create_dataloaders(
    reference_images, 
    sequence_images,
    batch_size=1,  # 使用较小的batch size以便可视化
    train_ratio=0.8, 
    val_ratio=0.1, 
    test_ratio=0.1, 
    shuffle=False,  # 不需要打乱测试集
    num_workers=0
)

# 创建输出目录
output_dir = 'outputs/inference_results'
os.makedirs(output_dir, exist_ok=True)

# 在测试集上运行模型并保存结果
with torch.no_grad():
    for batch_idx, (ref_images, seq_images) in enumerate(test_loader):
        # 将数据移至设备
        ref_images = ref_images.to(device)  # [B, 6, 1, 1551, 868]
        seq_images = seq_images.to(device)  # [B, 5, 1, 1551, 868]
        
        # 保存原始大小的序列图像
        original_seq_images = seq_images.clone()
        
        # 调整大小以适应模型
        B = ref_images.size(0)
        ref_k = ref_images.size(1)
        seq_k = seq_images.size(1)
        
        ref_images_flat = ref_images.view(B * ref_k, 1, 1551, 868)
        seq_images_flat = seq_images.view(B * seq_k, 1, 1551, 868)
        
        ref_images_resized = F.interpolate(ref_images_flat, size=(800, 800), mode='bilinear', align_corners=False)
        seq_images_resized = F.interpolate(seq_images_flat, size=(800, 800), mode='bilinear', align_corners=False)
        
        ref_images = ref_images_resized.view(B, ref_k, 1, 800, 800)
        seq_images = seq_images_resized.view(B, seq_k, 1, 800, 800)
        
        # 获取模型输出
        outputs = model(seq_images, ref_images)  # 输出为 [B, 5, 1, 1551, 868]
        
        # 可视化并保存结果
        for i in range(seq_images.size(1)):
            # 创建一个包含3幅图的可视化：输入、真实输出和预测输出
            plt.figure(figsize=(15, 5))
            
            # 输入序列图像
            plt.subplot(1, 3, 1)
            input_img = seq_images[0, i, 0].cpu().numpy()
            plt.imshow(input_img, cmap='gray')
            plt.title(f'Input Image (Frame {i+1})')
            plt.axis('off')
            
            # 真实序列图像
            plt.subplot(1, 3, 2)
            target_img = original_seq_images[0, i, 0].cpu().numpy()
            plt.imshow(target_img, cmap='gray')
            plt.title(f'Ground Truth (Frame {i+1})')
            plt.axis('off')
            
            # 模型预测
            plt.subplot(1, 3, 3)
            output_img = outputs[0, i, 0].cpu().numpy()
            plt.imshow(output_img, cmap='gray')
            plt.title(f'Model Output (Frame {i+1})')
            plt.axis('off')
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(f'{output_dir}/batch_{batch_idx}_frame_{i+1}.png')
            plt.close()
        
        print(f"Processed batch {batch_idx+1}/{len(test_loader)}")

print("Inference completed. Results saved to", output_dir)

# 计算测试集上的平均损失
mse_loss = torch.nn.MSELoss()
total_loss = 0.0
with torch.no_grad():
    for ref_images, seq_images in test_loader:
        ref_images = ref_images.to(device)
        seq_images = seq_images.to(device)
        
        # 保存原始高分辨率的seq_images用于计算损失
        original_seq_images = seq_images.clone()
        
        # 调整测试集数据大小
        B = ref_images.size(0)
        ref_k = ref_images.size(1)
        seq_k = seq_images.size(1)
        
        ref_images_flat = ref_images.view(B * ref_k, 1, 1551, 868)
        seq_images_flat = seq_images.view(B * seq_k, 1, 1551, 868)
        
        ref_images_resized = F.interpolate(ref_images_flat, size=(800, 800), mode='bilinear', align_corners=False)
        seq_images_resized = F.interpolate(seq_images_flat, size=(800, 800), mode='bilinear', align_corners=False)
        
        ref_images = ref_images_resized.view(B, ref_k, 1, 800, 800)
        seq_images = seq_images_resized.view(B, seq_k, 1, 800, 800)
        
        outputs = model(seq_images, ref_images)
        
        loss = mse_loss(outputs, original_seq_images)
        total_loss += loss.item()

avg_test_loss = total_loss / len(test_loader)
print(f'Average Test Loss: {avg_test_loss:.4f}')

# 从test_loader中获取单批次数据示例
def get_single_batch_from_loader(dataloader):
    # 创建一个迭代器
    dataloader_iter = iter(dataloader)
    # 获取第一个批次
    ref_batch, seq_batch = next(dataloader_iter)
    return ref_batch, seq_batch

# 使用示例
single_ref_images, single_seq_images = get_single_batch_from_loader(test_loader)
print(f"获取的单个批次形状 - 参考图像: {single_ref_images.shape}, 序列图像: {single_seq_images.shape}")

# 如果需要特定索引的批次，可以使用以下方法：
def get_batch_by_index(dataloader, index):
    if index >= len(dataloader):
        raise IndexError(f"索引 {index} 超出了 dataloader 的长度 {len(dataloader)}")
    
    for i, (ref, seq) in enumerate(dataloader):
        if i == index:
            return ref, seq
    
    return None, None

# 获取第2个批次的示例（如果存在）
if len(test_loader) > 1:
    second_ref_images, second_seq_images = get_batch_by_index(test_loader, 1)
    print(f"获取的第2个批次形状 - 参考图像: {second_ref_images.shape}, 序列图像: {second_seq_images.shape}") 