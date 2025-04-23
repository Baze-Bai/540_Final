import os
import sys
# Add Scripts folder to sys.path
script_path = os.path.abspath(os.path.join(os.getcwd(), "..", "Scripts"))
if script_path not in sys.path:
    sys.path.append(script_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
from tqdm import tqdm
from Scripts.Data_preprocess import plot_losses
from Scripts.MultiAttention_Unet import MultiImage_Unet, Unet_Model
from Scripts.Data_preprocess import process_images_into_sequences, create_dataloaders

# Set device for model training and inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained model
def load_model(model_path='best_model.pth', device=device):
    """
    Load a trained MultiImageDiffusion model from a saved checkpoint.
    
    Args:
        model_path: Path to the saved model weights
        device: Device to load the model on (cuda or cpu)
        
    Returns:
        Loaded model in evaluation mode
    """
    # Initialize the model architecture
    unet = MultiImage_Unet(in_channels=1)
    model = Unet_Model()
    model.to(device)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded from {model_path}")
    return model

# DDIM sample generation with reference images
def generate_samples(model, test_loader, steps=50, eta=0.0):
    """
    Generate samples using the trained model on a test dataset with DDIM sampling.
    
    Args:
        model: Trained MultiImageDiffusion model
        test_loader: DataLoader containing test data
        steps: Number of denoising steps for DDIM sampling
        eta: Stochasticity parameter for DDIM sampling
        
    Returns:
        Tuple of (predictions, targets) containing generated and ground truth images
    """
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for ref_hr, x_seq_hr in tqdm(test_loader, desc="Generating predictions"):
            # Move data to device
            ref_hr = ref_hr.to(device)
            x_seq_hr = x_seq_hr.to(device)
            
            # Get data dimensions
            B, kr, C, H_hr, W_hr = ref_hr.shape
            _, kx, _, _, _ = x_seq_hr.shape
            
            # Resize reference images to 800x800 for processing
            ref800 = F.interpolate(
                ref_hr.view(-1, 1, H_hr, W_hr),
                size=(800, 800),
                mode='bilinear',
                align_corners=False
            ).view(B, kr, 1, 800, 800)
            
            # Generate predictions using DDIM sampling
            _, pred_hr = model.ddim_sample(
                shape=(B, kx, 1, 800, 800),
                ref_seq=ref800,
                steps=steps,
                eta=eta,
                hr_output=True,
                hr_shape=(H_hr, W_hr)
            )
            
            # Store predictions and targets
            all_predictions.append(pred_hr.cpu())
            all_targets.append(x_seq_hr.cpu())
    
    return all_predictions, all_targets

# Calculate SSIM and PSNR for all generated samples
def calculate_metrics_dl(predictions, targets):
    """
    Calculate SSIM and PSNR metrics between predicted and target images.
    
    Args:
        predictions: List of predicted image tensors
        targets: List of target image tensors
        
    Returns:
        Tuple of (ssim_scores, psnr_scores) containing metrics for each sample
    """
    ssim_scores = []
    psnr_scores = []
    
    # Iterate through all batches
    for pred_batch, target_batch in tqdm(zip(predictions, targets), desc="Calculating metrics"):
        # Get batch dimensions
        B, kx, C, H, W = pred_batch.shape
        
        # Process each sample in the batch
        for b in range(B):
            batch_ssim = []
            batch_psnr = []
            
            # Process each frame in the sequence
            for k in range(kx):
                # Get single images
                pred_img = pred_batch[b, k, 0].numpy()
                target_img = target_batch[b, k, 0].numpy()
                
                # Normalize images to [0, 1] range for metric calculation
                pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
                target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
                
                # Calculate metrics
                ssim_val = ssim(target_img, pred_img, data_range=1.0)
                psnr_val = psnr(target_img, pred_img, data_range=1.0)
                
                batch_ssim.append(ssim_val)
                batch_psnr.append(psnr_val)
            
            # Average metrics for this sample
            ssim_scores.append(np.mean(batch_ssim))
            psnr_scores.append(np.mean(batch_psnr))
    
    return ssim_scores, psnr_scores

# Plot and save metric results
def plot_metrics(ssim_scores, psnr_scores):
    """
    Plot and save visualization of SSIM and PSNR metrics.
    
    Args:
        ssim_scores: List of SSIM scores for each sample
        psnr_scores: List of PSNR scores for each sample
    """
    # Create output directory if it doesn't exist
    os.makedirs('metric_results', exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot SSIM scores
    ax1.hist(ssim_scores, bins=20, alpha=0.7, color='blue')
    ax1.axvline(np.mean(ssim_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(ssim_scores):.4f}')
    ax1.set_title('SSIM Distribution')
    ax1.set_xlabel('SSIM Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Plot PSNR scores
    ax2.hist(psnr_scores, bins=20, alpha=0.7, color='green')
    ax2.axvline(np.mean(psnr_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(psnr_scores):.4f}')
    ax2.set_title('PSNR Distribution')
    ax2.set_xlabel('PSNR (dB)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('metric_results/metrics_distribution.png')
    plt.close()
    
    # Save individual sample metrics
    plt.figure(figsize=(12, 6))
    x = range(len(ssim_scores))
    plt.scatter(x, ssim_scores, alpha=0.5, label='SSIM')
    plt.scatter(x, [p/60 for p in psnr_scores], alpha=0.5, label='PSNR/60')  # Scale PSNR for visualization
    plt.axhline(np.mean(ssim_scores), color='blue', linestyle='--', 
                label=f'Mean SSIM: {np.mean(ssim_scores):.4f}')
    plt.axhline(np.mean(psnr_scores)/60, color='green', linestyle='--', 
                label=f'Mean PSNR: {np.mean(psnr_scores):.4f} dB')
    plt.title('Metrics by Sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('metric_results/metrics_by_sample.png')
    plt.close()
    
    # Print summary statistics
    print(f"SSIM - Mean: {np.mean(ssim_scores):.4f}, Min: {np.min(ssim_scores):.4f}, Max: {np.max(ssim_scores):.4f}")
    print(f"PSNR - Mean: {np.mean(psnr_scores):.4f}, Min: {np.min(psnr_scores):.4f}, Max: {np.max(psnr_scores):.4f}")
    
    # Save detailed metrics to file
    with open('metric_results/metrics_summary.txt', 'w') as f:
        f.write(f"SSIM - Mean: {np.mean(ssim_scores):.4f}, Std: {np.std(ssim_scores):.4f}\n")
        f.write(f"SSIM - Min: {np.min(ssim_scores):.4f}, Max: {np.max(ssim_scores):.4f}\n\n")
        f.write(f"PSNR - Mean: {np.mean(psnr_scores):.4f}, Std: {np.std(psnr_scores):.4f}\n")
        f.write(f"PSNR - Min: {np.min(psnr_scores):.4f}, Max: {np.max(psnr_scores):.4f}\n\n")
        
        f.write("Individual Sample Metrics:\n")
        for i, (s, p) in enumerate(zip(ssim_scores, psnr_scores)):
            f.write(f"Sample {i+1}: SSIM = {s:.4f}, PSNR = {p:.4f} dB\n")

# Function to count parameters in PyTorch model
def count_parameters(model):
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing total, trainable, and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params
    }

# Independent inference function for generating images with a trained model
@torch.no_grad()
def generate_images(
    model: Unet_Model,   # Trained model
    ref_seq: torch.Tensor,        # [B,kr,1,800,800] Reference frame sequence
    pos_x: torch.LongTensor,      # Position indices of frames to generate (1-based)
    pos_r: torch.LongTensor = None, # Position indices of reference frames (1-based), auto-calculated if None
    ddim_steps: int = 50,         # Number of DDIM sampling steps
    hr_output: bool = True,       # Whether to output high-resolution results
    hr_shape: tuple = (1551, 868), # High-resolution output shape
    eta: float = 0.0,             # DDIM noise parameter
    device: str = None            # Inference device, uses model's default if None
):
    """
    Independent inference function: Generate images using a trained MultiImageDiffusion model.
    
    Args:
        model: Trained MultiImageDiffusion model
        ref_seq: Reference frame sequence with shape [B,kr,1,800,800]
        pos_x: Position indices of frames to generate (1-based)
        pos_r: Position indices of reference frames (1-based), auto-calculated based on pos_x if None
        ddim_steps: Number of DDIM sampling steps, higher steps give better quality but slower generation
        hr_output: Whether to output high-resolution results
        hr_shape: High-resolution output shape (default 1551×868)
        eta: DDIM sampling noise coefficient (0=deterministic sampling, >0 introduces randomness)
        device: Inference device, uses model's default if None
        
    Returns:
        If hr_output=True: Returns (lr_result, hr_result) tuple
            - lr_result: Low-resolution results with shape [B,kx,1,800,800]
            - hr_result: High-resolution results with shape [B,kx,1,hr_shape[0],hr_shape[1]]
        If hr_output=False: Returns only lr_result
    
    Usage example:
        model = ... # Load trained model
        ref_frames = ... # Reference frames with shape [1,2,1,800,800]
        pos_x = torch.tensor([3, 4]) # Generate frames 3 and 4
        pos_r = torch.tensor([1, 2]) # Reference frames are 1 and 2
        
        # Generate results
        lr_result, hr_result = generate_images(
            model=model,
            ref_seq=ref_frames,
            pos_x=pos_x,
            pos_r=pos_r
        )
    """
    # Set to evaluation mode
    model.eval()
    
    # Determine device
    if device is None:
        device = model.device
    else:
        device = torch.device(device)
    
    # Move data to correct device
    ref_seq = ref_seq.to(device)
    pos_x = pos_x.to(device)
    
    # Get shape information
    B, kr, C, H, W = ref_seq.shape
    kx = len(pos_x)
    
    # Auto-calculate reference frame positions if not provided
    # Assumes reference frames are positioned in order outside the frames to generate
    if pos_r is None:
        # Generate a sequence with all possible positions
        pos_all = torch.arange(1, kx + kr + 1, device=device)
        # Create mask to mark positions that are not input positions
        mask = torch.ones_like(pos_all, dtype=torch.bool)
        mask[pos_x - 1] = False
        # Select the first kr non-input positions as reference frame positions
        pos_r = pos_all[mask][:kr]
    else:
        pos_r = pos_r.to(device)
    
    # Calculate positional encoding for reference frames
    pe_r = model._pos_encode(pos_r)
    
    # Add positional encoding to reference frames
    r_in = ref_seq + pe_r
    
    # Set output shape
    shape = (B, kx, C, H, W)
    
    # Generate results using DDIM sampling algorithm
    return model.ddim_sample(
        shape=shape,
        ref_seq=r_in,
        steps=ddim_steps,
        eta=eta,
        hr_output=hr_output,
        hr_shape=hr_shape
    )

# Save example images from test set
def save_example_comparisons(predictions, targets, num_examples=5):
    """
    Save comparison visualizations between predicted and target images.
    
    Args:
        predictions: List of predicted image tensors
        targets: List of target image tensors
        num_examples: Number of example comparisons to save
    """
    os.makedirs('metric_results/examples', exist_ok=True)
    
    total_samples = len(predictions) * predictions[0].shape[0]
    sample_indices = np.random.choice(total_samples, size=min(num_examples, total_samples), replace=False)
    
    sample_counter = 0
    for batch_idx, (pred_batch, target_batch) in enumerate(zip(predictions, targets)):
        B, kx, C, H, W = pred_batch.shape
        
        for b in range(B):
            if sample_counter in sample_indices:
                # Create a figure for this example
                fig, axes = plt.subplots(2, kx, figsize=(kx*3, 6))
                
                # Process each frame in the sequence
                for k in range(kx):
                    # Get single images
                    pred_img = pred_batch[b, k, 0].numpy()
                    target_img = target_batch[b, k, 0].numpy()
                    
                    # Normalize for display
                    pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
                    target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
                    
                    # Calculate metrics for this specific image
                    img_ssim = ssim(target_img, pred_img, data_range=1.0)
                    img_psnr = psnr(target_img, pred_img, data_range=1.0)
                    
                    # Display images
                    axes[0, k].imshow(target_img, cmap='gray')
                    axes[0, k].set_title(f'Original Frame {k+1}')
                    axes[0, k].axis('off')
                    
                    axes[1, k].imshow(pred_img, cmap='gray')
                    axes[1, k].set_title(f'Predicted\nSSIM: {img_ssim:.3f}\nPSNR: {img_psnr:.1f}')
                    axes[1, k].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'metric_results/examples/comparison_sample_{sample_counter+1}.png')
                plt.close()
            
            sample_counter += 1


def save_generated_images(tensor, save_dir="./Images_output", filename_prefix="gen_img", normalize=True):
    """
    Convert generated tensor to images and save them to disk.
    
    Args:
        tensor: Image tensor with shape [B,kx,1,H,W] or [B,1,H,W]
        save_dir: Directory to save images
        filename_prefix: Prefix for filenames
        normalize: Whether to normalize values to [0,255] range
    
    Returns:
        True if successful
    """
    import os
    import numpy as np
    from PIL import Image
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure tensor is on CPU and convert to numpy
    tensor = tensor.detach().cpu()
    
    # Process different tensor dimensions
    if tensor.dim() == 5:  # [B,kx,1,H,W]
        B, kx, C, H, W = tensor.shape
        # Iterate through each batch and frame
        for b in range(B):
            for k in range(kx):
                img_tensor = tensor[b, k, 0]  # Extract single image [H,W]
                
                # Convert to uint8
                img_np = img_tensor.numpy().astype(np.uint8)
                
                # Create PIL image and save
                img = Image.fromarray(img_np)
                save_path = os.path.join(save_dir, f"{filename_prefix}_batch{b}_frame{k}.png")
                img.save(save_path)
                print(f"Image saved to: {save_path}")
    
    elif tensor.dim() == 4:  # [B,1,H,W]
        B, C, H, W = tensor.shape
        # Iterate through each batch
        for b in range(B):
            img_tensor = tensor[b, 0]  # Extract single image [H,W]
            
            # Normalize to [0,255] range
            if normalize:
                img_min = img_tensor.min()
                img_max = img_tensor.max()
                img_tensor = (img_tensor - img_min) / (img_max - img_min) * 255
            
            # Convert to uint8
            img_np = img_tensor.numpy().astype(np.uint8)
            
            # Create PIL image and save
            img = Image.fromarray(img_np)
            save_path = os.path.join(save_dir, f"{filename_prefix}_batch{b}.png")
            img.save(save_path)
            print(f"Image saved to: {save_path}")
    
    else:
        raise ValueError(f"Unsupported tensor dimensions: {tensor.shape}")
    
    return True

def Train_DL(model, train_loader, val_loader, num_epochs, device):
    """
    Train a diffusion-based model on image sequences.
    
    Args:
        model: The Unet_Model instance to train
        train_loader: DataLoader for training data containing tuples of (reference_images, target_sequence)
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        device: Device to train on (cuda or cpu)
        
    Returns:
        Trained model with best weights loaded
    """
    # For recording training and validation losses
    train_losses = []
    val_losses = []

    # For tracking the best model
    best_val_loss = float('inf')
    best_model_epoch = -1
    best_model_path = 'best_model.pth'

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set model to training mode
        epoch_loss = 0.0  # Record total training loss for each epoch
        
        for i, (ref_hr, x_seq_hr) in enumerate(train_loader):
            # Move data to device
            x_seq_hr = x_seq_hr.to(device)  # [B, kx, 1, H_hr, W_hr] - Input sequence
            ref_hr = ref_hr.to(device)      # [B, kr, 1, H_hr, W_hr] - Reference images
            
            # Get data dimensions
            B, kx_size, C, H_hr, W_hr = x_seq_hr.shape
            _, kr, _, _, _ = ref_hr.shape    # Get number of reference frames

            # Resize images to processing size 800x800
            x800 = F.interpolate(
                x_seq_hr.view(-1, 1, H_hr, W_hr),  # Reshape to [B*kx, 1, H_hr, W_hr] for interpolation
                size=(800, 800),                   # Target size
                mode='bilinear',                   # Bilinear interpolation
                align_corners=False                # Don't align corners
            ).view(B, kx_size, 1, 800, 800)        # Reshape back to [B, kx, 1, 800, 800]
            
            ref800 = F.interpolate(
                ref_hr.view(-1, 1, H_hr, W_hr),    # Reshape to [B*kr, 1, H_hr, W_hr] for interpolation
                size=(800, 800),                   # Target size
                mode='bilinear',                   # Bilinear interpolation
                align_corners=False                # Don't align corners
            ).view(B, kr, 1, 800, 800)             # Reshape back to [B, kr, 1, 800, 800]
            
            # Position indices for time steps in diffusion model
            pos_x = torch.tensor([2, 4, 6, 8, 10], dtype=torch.long).to(device)  # kx=5
            
            # Forward propagation to calculate loss
            loss = model.p_losses(x800, ref800, pos_x, x_seq_hr)  # Calculate dual-scale loss
            
            # Backpropagation and optimization
            optimizer = model.optimizer  # Get the optimizer from model
            optimizer.zero_grad()        # Clear gradients
            loss.backward()              # Backpropagation
            optimizer.step()             # Update parameters
            
            epoch_loss += loss.item()  # Accumulate batch loss
            
            # Print loss every 10 batches
            if i % 10 == 0:
                print(f"Training epoch {epoch:02d}, batch {i:04d} | Dual-scale loss = {loss.item():.5f}")
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        
        with torch.no_grad():  # Don't calculate gradients
            for i, (ref_hr, x_seq_hr) in enumerate(val_loader):
                # Move data to device
                x_seq_hr = x_seq_hr.to(device)
                ref_hr = ref_hr.to(device)
                
                # Get data dimensions
                B, kx_size, C, H_hr, W_hr = x_seq_hr.shape
                _, kr, _, _, _ = ref_hr.shape  # Get number of reference frames
                
                # Resize images to processing size 800x800
                x800 = F.interpolate(
                    x_seq_hr.view(-1, 1, H_hr, W_hr),
                    size=(800, 800),
                    mode='bilinear',
                    align_corners=False
                ).view(B, kx_size, 1, 800, 800)
                
                ref800 = F.interpolate(
                    ref_hr.view(-1, 1, H_hr, W_hr),
                    size=(800, 800),
                    mode='bilinear',
                    align_corners=False
                ).view(B, kr, 1, 800, 800)
                
                # Position indices
                pos_x = torch.tensor([2, 4, 6, 8, 10], dtype=torch.long).to(device)
                
                # Calculate validation loss
                loss = model.p_losses(x800, ref800, pos_x, x_seq_hr)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best-performing model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"Found better model! Epoch {epoch:02d} | Validation loss = {avg_val_loss:.5f}")
        
        # Print average loss for each epoch (training and validation)
        print(f"Training epoch {epoch:02d} | Training loss = {avg_train_loss:.5f} | Validation loss = {avg_val_loss:.5f}")
        
        # Plot and save loss curves after each epoch
        plot_losses(train_losses, val_losses, epoch)

    # Record best model information after training
    print(f"\nTraining completed! Best model at epoch {best_model_epoch:02d} with validation loss of {best_val_loss:.5f}")
    print(f"Best model saved to {best_model_path}")

    # Load the best model for subsequent use
    model.load_state_dict(torch.load(best_model_path))
    
    return model