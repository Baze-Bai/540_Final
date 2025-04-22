import os
import re
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import glob

# Import the same image processing function from Naive_Func.py
def load_16bit_to_8bit(img_path):
    """
    Load 16-bit image and linearly map to 8-bit [0,255]

    Parameters:
        img_path (str): Path to the image file

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

# Extract index from filename that matches pattern slice-XXX.png
def extract_idx(fname):
    m = re.match(r'^slice-(\d+)\.png$', fname)
    return int(m.group(1)) if m else None

def collect_triples(Image_dir):
    # Collect all triples (t, t+2 → t+1) for frame interpolation
    files = [
        f for f in os.listdir(Image_dir)
        if extract_idx(f) is not None
    ]
    # Sort by numerical order
    files = sorted(files, key=lambda f: extract_idx(f))

    idx_map = { extract_idx(f): f for f in files }
    triples = []
    for t in sorted(idx_map):
        if (t+1) in idx_map and (t+2) in idx_map:
            triples.append((idx_map[t], idx_map[t+2], idx_map[t+1]))
    
    return triples

def build_dataset(Image_dir, Target_size, Samples_per_triple, Test_size):
    """
    Build training and testing datasets for the frame interpolation model
    
    Parameters:
        Image_dir: Directory containing image slices
        Target_size: Target image size (width, height) for processing
        Samples_per_triple: Number of pixels to sample from each triple
        Test_size: Proportion of data to use for testing
        
    Returns:
        X_train, X_test, y_train, y_test: Training and testing datasets
    """
    triples = collect_triples(Image_dir)
    X_parts, y_parts = [], []
    for f1, f2, fm in triples:
        # Read 16-bit images and convert to 8-bit
        i1 = load_16bit_to_8bit(os.path.join(Image_dir, f1))
        i2 = load_16bit_to_8bit(os.path.join(Image_dir, f2))
        im = load_16bit_to_8bit(os.path.join(Image_dir, fm))
        
        # Resize to target size
        p1 = Image.fromarray(i1, mode='L').resize(Target_size)
        p2 = Image.fromarray(i2, mode='L').resize(Target_size)
        pm = Image.fromarray(im, mode='L').resize(Target_size)
        
        # Flatten to 1D arrays
        i1_flat = np.array(p1).flatten()
        i2_flat = np.array(p2).flatten()
        im_flat = np.array(pm).flatten()
        
        # Random sampling of pixels
        idxs = np.random.choice(len(i1_flat), size=Samples_per_triple, replace=False)
        X_parts.append(np.stack([i1_flat[idxs], i2_flat[idxs]], axis=1))
        y_parts.append(im_flat[idxs])

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    print(f'Total samples: {X.shape[0]}, Feature dimensions: {X.shape[1]}')

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, Model_path):
    """
    Train a Random Forest regression model for frame interpolation
    
    Parameters:
        X_train: Training features
        y_train: Training targets
        Model_path: Path to save the trained model
        
    Returns:
        The trained model
    """
    # Initialize and train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    print('Starting model training...')
    model.fit(X_train, y_train)
    joblib.dump(model, Model_path)
    
    return model

def test_model(model, X_test, y_test):
    """
    Evaluate model performance on test data
    
    Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        y_pred: Model predictions
        mse: Mean squared error
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Test MSE: {mse:.4f}')
    return y_pred, mse

def predict_middle(model, frame_a_path, frame_b_path, out_path, Target_size, Orig_size, display=False):
    """
    Predict the middle frame between two frames
    
    Parameters:
        model: Trained model
        frame_a_path: Path to first frame
        frame_b_path: Path to second frame
        out_path: Path to save the generated middle frame
        Target_size: Size for model processing
        Orig_size: Original output size
        display: Whether to display the result
        
    Returns:
        PIL Image of the predicted middle frame
    """
    # Convert 16-bit to 8-bit
    a_arr = load_16bit_to_8bit(frame_a_path)
    b_arr = load_16bit_to_8bit(frame_b_path)
    
    # Resize to target size
    a = np.array(Image.fromarray(a_arr, mode='L').resize(Target_size)).flatten()
    b = np.array(Image.fromarray(b_arr, mode='L').resize(Target_size)).flatten()
    
    X_in = np.stack([a, b], axis=1)
    pred = model.predict(X_in)
    img = (pred.clip(0,255)
               .reshape(Target_size)
               .astype(np.uint8))
    pil = Image.fromarray(img, mode='L')
    # Resize back to original resolution
    pil = pil.resize(Orig_size, Image.BILINEAR)
    pil.save(out_path)
    
    if display:
        display_image(pil)
        
    return pil

def display_image(pil_img):
    """
    Display a PIL image object
    
    Parameters:
        pil_img: PIL Image object to display
    """
    try:
        # Try using matplotlib for display
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(pil_img, cmap='gray')
        plt.axis('off')
        plt.show()
    except ImportError:
        # If matplotlib is not available, use PIL's built-in display
        pil_img.show()


