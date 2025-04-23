# Image Interpolation Project

This project implements multiple methods for image interpolation:
- Machine Learning (Random Forest) based interpolation 
- Deep Learning (UNet) based interpolation
- Naive average-based interpolation

The application provides a Streamlit web interface for comparing these methods and evaluating them using SSIM and PSNR metrics.

## Installation

### Option 1: Using pip

```bash
# Install from source
pip install -e .
```

### Option 2: Manual installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Running the Streamlit Web Interface

```bash
streamlit run streamlit.py
```

### Basic Interpolation (2 images)
1. Upload two consecutive frames in the sidebar (PNG, JPG, TIFF formats supported)
2. Optional: Upload a real middle frame as reference to evaluate prediction quality
3. Click "Perform Basic Interpolation" button 
4. View the predictions from machine learning and naive average methods
5. If a reference frame was provided, view SSIM and PSNR evaluation metrics

### Direct DL Prediction
1. Click the "Run Direct DL Prediction" button
2. The system will load pre-saved images and run the deep learning model
3. View the automatically generated predictions

## Interpolation Methods

### 1. Naive Approach

The naive approach uses a simple pixel-by-pixel averaging of two consecutive frames to generate the middle frame. This method provides a baseline comparison for the more sophisticated approaches.

#### Implementation
The implementation is in `Scripts/Naive_Func.py`, which:
- Loads two consecutive image frames
- Converts images to 8-bit grayscale if needed
- Calculates the pixel-wise average between the two frames
- Returns and optionally saves the resulting image

#### Prediction
```python
from Scripts.Naive_Func import average_gray_paths
middle_frame = average_gray_paths(frame1_path, frame2_path, output_path)
```

### 2. Machine Learning Approach

The machine learning approach uses a Random Forest Regressor to predict the middle frame from two consecutive frames.

#### Training
The model training process:
1. Extracts features from frame pairs and their known middle frames
2. Trains a Random Forest model to predict middle frame pixel values
3. Saves the trained model for later use

To train the model:
```python
from Scripts.ML_Func import extract_features_targets, train_model

# Extract features and targets from image triplets
X_train, y_train = extract_features_targets(image_dir, target_size, samples_per_triple)

# Train Random Forest model
model = train_model(X_train, y_train, model_path)
```

#### Prediction
The prediction process:
1. Loads the trained model
2. Extracts features from two consecutive frames
3. Uses the model to predict the middle frame

```python
from Scripts.ML_Func import predict_middle
middle_frame = predict_middle(model, frame1_path, frame2_path, output_path, target_size, orig_size)
```

### 3. Deep Learning Approach

The deep learning approach uses custom UNet architectures for image interpolation, including multi-image UNet and diffusion models.

#### Model Architecture
The project implements several neural network architectures:
- `MultiImageUNet`: A UNet architecture adapted for multiple input images


#### Training
The deep learning training process:
1. Preprocesses image sequences into training data
2. Creates dataloaders for training, validation, and testing
3. Trains the model with appropriate loss functions and optimization algorithms
4. Tracks and plots training and validation losses

To train a deep learning model:
```python
from Scripts.Data_preprocess import process_images_into_sequences, create_dataloaders
from Scripts.DL_Func import train_unet_model

# Process images into sequences for training
reference_images, sequence_images = process_images_into_sequences(image_dir)

# Create train, validation, and test dataloaders
train_loader, val_loader, test_loader = create_dataloaders(reference_images, sequence_images)

# Train the model
train_unet_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
```

#### Prediction
For inference using a trained deep learning model:
```python
from Class_Func.MultiImage_Unet import MultiImageUNet
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiImageUNet(in_channels=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Make prediction
with torch.no_grad():
    inputs = torch.stack([frame1_tensor, frame2_tensor]).unsqueeze(0).to(device)
    predicted_frame = model(inputs).squeeze().cpu().numpy()
```

## Evaluation Metrics

This project uses two primary metrics to evaluate interpolation quality:

### 1. Structural Similarity Index (SSIM)
- Measures the perceived similarity between two images
- Range: [-1, 1], with 1 indicating perfect similarity
- Considers luminance, contrast, and structure
- Provides a better correlation with human perception than pixel-wise error metrics

### 2. Peak Signal-to-Noise Ratio (PSNR)
- Measures the ratio between maximum possible power of a signal and the power of corrupting noise
- Measured in decibels (dB)
- Higher values indicate better quality
- Commonly used for image compression quality evaluation

Implementation:
```python
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Calculate metrics between predicted and reference image
ssim_value = ssim(reference_image, predicted_image, data_range=255)
psnr_value = psnr(reference_image, predicted_image, data_range=255)
```

## Project Structure

- `streamlit.py` - Main web application
- `Scripts/` - Core functionality implementation
  - `ML_Func.py` - Machine Learning model functions
  - `DL_Func.py` - Deep Learning model functions
  - `Naive_Func.py` - Naive interpolation methods
  - `Data_preprocess.py` - Data preprocessing utilities
- `Class_Func/` - Model architecture implementations
  - `MultiImage_Unet.py` - Multi-image UNet architecture
  - `MultiImage_Diffusion.py` - Multi-image Diffusion model
  - `Unet_Model.py` - UNet model implementation
- `requirements.txt` - Project dependencies
- `setup.py` - Installation script

## Models

The project uses both ML and DL models:
- Random Forest regressor for Machine Learning approach
- UNet architectures for Deep Learning approach

### UNet Architecture

The project implements a custom UNet architecture designed specifically for image interpolation:

#### Basic UNet 
The standard UNet model consists of:
- An encoder path that downsamples the input image through multiple layers
- A decoder path that upsamples the features back to the original resolution
- Skip connections between corresponding encoder and decoder layers

#### MultiImageUNet
A specialized UNet architecture for processing multiple consecutive frames:

- **Input Processing**:
  - Takes multiple input frames and reference frames
  - Utilizes self-attention mechanisms to process temporal relationships

- **Encoder Path**:
  - Multiple encoder layers with cross-attention modules
  - Each encoder layer includes downsampling and channel expansion
  - Features are progressively reduced in spatial dimensions but increased in channel depth

- **Decoder Path**:
  - Decoder layers with skip connections from the encoder
  - Upsampling through transposed convolutions
  - Gradual restoration of spatial dimensions while reducing channel depth

- **Special Features**:
  - Cross-attention mechanisms for relating features across frames
  - Self-attention for capturing temporal information
  - Final convolution layers to generate the interpolated frame

#### UnetModel
The main UNet implementation that builds upon MultiImageUNet:
- Handles image resizing for consistent processing
- Includes a final upsampling layer to restore original resolution
- Applies additional convolutions for enhancing quality of the final output

Models are automatically downloaded from Hugging Face when needed. 