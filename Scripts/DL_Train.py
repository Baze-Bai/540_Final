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
from Scripts.DL_Func import Train_DL

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiImageUNet(in_channels=1)
Train_DL(model, train_loader, val_loader, num_epochs=50, device=device)