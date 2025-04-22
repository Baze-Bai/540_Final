import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def load_grayscale_images(image_dir='Dataset_images'):
    """
    Load all PNG images from the specified directory as raw 16-bit arrays,
    then linearly map each image to 8-bit [0,255].

    Args:
        image_dir (str): Directory containing PNG images

    Returns:
        images (list of np.ndarray): each shape [H, W], dtype=uint8
        filenames (list of str)
    """
    # Sort filenames numerically to avoid 'slice-10.png' appearing before 'slice-2.png'
    files = glob.glob(os.path.join(image_dir, "*.png"))
    image_files = sorted(
        files,
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('-')[-1])
    )

    images = []
    filenames = []
    for img_file in image_files:
        # 1) Open original 16-bit image
        img16 = Image.open(img_file)
        arr16 = np.array(img16, dtype=np.uint16)

        # 2) Linear normalization to [0,255]
        minv, maxv = arr16.min(), arr16.max()
        if maxv > minv:
            arr_norm = (arr16.astype(np.float32) - minv) / (maxv - minv)
        else:
            arr_norm = np.zeros_like(arr16, dtype=np.float32)
        arr8 = (arr_norm * 255.0).round().astype(np.uint8)

        images.append(arr8)
        filenames.append(os.path.basename(img_file))
        print(f"Loaded {filenames[-1]} | raw 16-bit range [{minv},{maxv}] → mapped to uint8")

    print(f"Total images loaded: {len(images)}")
    return images, filenames


def process_images_into_sequences(image_dir='Dataset_images'):
    """
    Processes the normalized 8-bit images into reference/sequence tensors
    according to the specified pattern.

    Returns:
        reference_images (list of torch.FloatTensor): each shape [6,1,H,W]
        sequence_images  (list of torch.FloatTensor): each shape [5,1,H,W]
    """
    images, filenames = load_grayscale_images(image_dir)

    if not images:
        raise ValueError("No images were loaded")

    H, W = images[0].shape
    reference_images = []
    sequence_images  = []

    # According to original logic, slide every 3 images, taking 11 images as a group
    for start in range(0, len(images) - 10, 3):
        group = images[start:start + 11]
        print(f"Processing group idx {start}–{start+10}: files {filenames[start:start+11]}")

        # Take indices 0,2,4,6,8,10 (6 images) as reference
        ref_np = np.stack([group[i] for i in [0,2,4,6,8,10]], axis=0)
        # Take indices 1,3,5,7,9 (5 images) as sequence
        seq_np = np.stack([group[i] for i in [1,3,5,7,9]], axis=0)

        # Add channel dimension and convert to tensor
        ref_t = torch.from_numpy(ref_np).unsqueeze(1).float()  # [6,1,H,W]
        seq_t = torch.from_numpy(seq_np).unsqueeze(1).float()  # [5,1,H,W]

        reference_images.append(ref_t)
        sequence_images.append(seq_t)

    print(f"Processed {len(reference_images)} groups")
    print(f"  Reference tensor shape: {reference_images[0].shape}")
    print(f"  Sequence  tensor shape: {sequence_images[0].shape}")
    
    return reference_images, sequence_images


class ImageSequenceDataset(Dataset):
    """
    Dataset for reference and sequence images.
    """
    def __init__(self, reference_images, sequence_images):
        """
        Initialize the dataset with reference and sequence images.
        
        Args:
            reference_images (list): List of reference image tensors
            sequence_images (list): List of sequence image tensors
        """
        self.reference_images = reference_images
        self.sequence_images = sequence_images
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.reference_images)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (reference_images, sequence_images) tensors
        """
        return self.reference_images[idx].to('cuda'), self.sequence_images[idx].to('cuda')


def create_dataloaders(reference_images, sequence_images, batch_size=4, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True, num_workers=0):
    """
    Create train, validation and test dataloaders from reference and sequence images.
    
    Args:
        reference_images (list): List of reference image tensors
        sequence_images (list): List of sequence image tensors
        batch_size (int): Batch size for dataloaders
        train_ratio (float): Ratio of training data to total data
        val_ratio (float): Ratio of validation data to total data
        test_ratio (float): Ratio of test data to total data
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for dataloaders
    
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Create the dataset
    dataset = ImageSequenceDataset(reference_images, sequence_images)
    print("dataset created")
    
    # Split into train, validation and test sets sequentially (not randomly)
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Sequential dataset split
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Created train dataloader with {len(train_dataloader)} batches")
    print(f"Created validation dataloader with {len(val_dataloader)} batches")
    print(f"Created test dataloader with {len(test_dataloader)} batches")
    print(f"Train data: first {train_size} samples ({train_ratio:.0%})")
    print(f"Validation data: next {val_size} samples ({val_ratio:.0%})")
    print(f"Test data: last {test_size} samples ({test_ratio:.0%})")
    
    return train_dataloader, val_dataloader, test_dataloader


def plot_losses(train_losses, val_losses, current_epoch):
    """
    Plot training and validation loss curves and save them.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        current_epoch (int): Current training epoch
    """
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(train_losses) + 1))
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title(f'Training and Validation Loss Curves (Epoch {current_epoch+1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Create save directory if it doesn't exist
    if not os.path.exists('loss_plots'):
        os.makedirs('loss_plots')
    
    # Save current epoch's loss plot
    plt.savefig(f'loss_plots/loss_curve_epoch_{current_epoch+1}.png')
    
    # Save the latest loss curve (overwrite previous)
    plt.savefig('loss_plots/latest_loss_curve.png')
    
    plt.close()  # Close the figure to avoid memory leaks
    


