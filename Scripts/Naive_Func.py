from PIL import Image
import numpy as np
import os
import glob

def average_gray_images(img1, img2):
    """
    Takes two grayscale PIL.Image objects and returns their pixel-wise average as a grayscale image.
    Note: Assumes input images are already preprocessed grayscale images.

    Parameters:
        img1 (PIL.Image.Image): Grayscale image 1
        img2 (PIL.Image.Image): Grayscale image 2

    Returns:
        PIL.Image.Image: Averaged grayscale image, mode='L'
    """
    # Convert directly to numpy arrays (float32 to avoid overflow)
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # Pixel-wise average, and convert back to [0,255] uint8
    avg_arr = ((arr1 + arr2) / 2.0).round().clip(0, 255).astype(np.uint8)

    # Convert back to PIL.Image
    return Image.fromarray(avg_arr, mode='L')


def load_16bit_to_8bit(img_path):
    """
    Loads a 16-bit image and linearly maps it to 8-bit [0,255]

    Parameters:
        img_path (str): Path to image file

    Returns:
        np.ndarray: 8-bit grayscale image, dtype=uint8
    """
    # Open original 16-bit image
    img16 = Image.open(img_path)
    arr16 = np.array(img16, dtype=np.uint16)

    # Linear normalization to [0,255]
    minv, maxv = arr16.min(), arr16.max()
    if maxv > minv:
        arr_norm = (arr16.astype(np.float32) - minv) / (maxv - minv)
    else:
        arr_norm = np.zeros_like(arr16, dtype=np.float32)
    arr8 = (arr_norm * 255.0).round().astype(np.uint8)
    
    return arr8


# Modified function, using the same image loading method as Data_preprocess.py
def average_gray_paths(path1, path2, out_path):
    """
    Reads two 16-bit grayscale images from files, linearly maps them to 8-bit,
    calculates their average, and saves to out_path.
    
    Parameters:
        path1 (str): Path to first image
        path2 (str): Path to second image
        out_path (str): Output path for the averaged image
        
    Returns:
        PIL.Image.Image: Averaged grayscale image
    """
    # Use the same loading method as Data_preprocess.py
    arr1 = load_16bit_to_8bit(path1)
    arr2 = load_16bit_to_8bit(path2)
    
    # Convert to PIL.Image
    img1 = Image.fromarray(arr1, mode='L')
    img2 = Image.fromarray(arr2, mode='L')
    
    # Calculate average image
    avg = average_gray_images(img1, img2)
    avg.save(out_path)
    print(f'Average image saved to {out_path}')

    return avg

