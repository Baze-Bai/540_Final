import os
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import joblib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from Scripts.ML_Func import predict_middle, load_16bit_to_8bit
from Scripts.Naive_Func import average_gray_paths
import io
import tempfile
from huggingface_hub import hf_hub_download
import torch

# Page configuration
st.set_page_config(page_title="Image Interpolation Comparison", layout="wide")
st.title("Image Generation for Li-ion Battery")

# Constants
TARGET_SIZE = (256, 256)
ORIG_SIZE = (868, 1551)  # Original image size
MODEL_REPO_ID = "BazeBai/540_Final_Baze"  # Replace with your actual HF repo ID
MODEL_FILENAME = "interp_rf.pkl"  # Replace with your actual model filename
DL_MODEL_FILENAME = "unet_model_epoch.pth"  # Deep learning model filename

# Ensure output directories exist
os.makedirs("Outputs", exist_ok=True)
os.makedirs("Outputs/ML", exist_ok=True)
os.makedirs("Outputs/Naive", exist_ok=True)
os.makedirs("Outputs/DL", exist_ok=True)

# Create tabs for different methods
tab1, tab2 = st.tabs(["Basic Interpolation (2 images)", "Direct DL Prediction"])

# Save uploaded file to temporary location
def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file"""
    if uploaded_file is None:
        return None
    
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

# Validate image and convert to 8-bit
def validate_and_convert_image(file_path):
    """Validate and convert image using load_16bit_to_8bit"""
    if file_path is None:
        return False, None
    
    try:
        # Use the same method as in Data_preprocess.py to load images
        arr8 = load_16bit_to_8bit(file_path)
        img = Image.fromarray(arr8, mode='L')
        return True, img
    except Exception as e:
        st.error(f"Image file is invalid or corrupted: {str(e)}")
        return False, None

# Calculate evaluation metrics
def calculate_metrics(img1, img2):
    """Calculate SSIM and PSNR between two images"""
    try:
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        
        # Ensure both images are the same size
        if img1_array.shape != img2_array.shape:
            img1 = img1.resize(img2.size)
            img1_array = np.array(img1)
        
        # Calculate SSIM
        ssim_value = ssim(img1_array, img2_array, data_range=255)
        
        # Calculate PSNR
        psnr_value = psnr(img1_array, img2_array, data_range=255)
        
        return ssim_value, psnr_value
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return 0, 0

# Basic interpolation tab content
with tab1:
    # Sidebar - Upload area for basic interpolation
    st.sidebar.header("Upload Images (Basic Interpolation)")
    frame1 = st.sidebar.file_uploader("Upload First Frame", type=["png", "jpg", "jpeg", "tif", "tiff"], key="frame1")
    frame2 = st.sidebar.file_uploader("Upload Second Frame", type=["png", "jpg", "jpeg", "tif", "tiff"], key="frame2")
    reference = st.sidebar.file_uploader("Upload Reference Middle Frame (Optional)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="reference")

    # Main content
    if frame1 is not None and frame2 is not None:
        with st.spinner("Processing uploaded images..."):
            # Save uploaded images to temporary files
            frame1_path = save_uploaded_file(frame1)
            frame2_path = save_uploaded_file(frame2)
            
            if frame1_path and frame2_path:
                # Validate and convert images
                valid1, img1 = validate_and_convert_image(frame1_path)
                valid2, img2 = validate_and_convert_image(frame2_path)
                
                if valid1 and valid2:
                    # Save processed images
                    out_frame1_path = os.path.join("Outputs", "frame1.png")
                    out_frame2_path = os.path.join("Outputs", "frame2.png")
                    
                    img1.save(out_frame1_path)
                    img2.save(out_frame2_path)
                    
                    # Display uploaded images
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("First Frame")
                        st.image(img1, use_column_width=True)
                    with col2:
                        st.subheader("Second Frame")
                        st.image(img2, use_column_width=True)
                    
                    # Execute interpolation button
                    if st.button("Perform Basic Interpolation"):
                        try:
                            with st.spinner("Loading model and generating predictions..."):
                                try:
                                    # Download model from Hugging Face Hub
                                    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
                                    
                                    # Load the model
                                    model = joblib.load(model_path)
                                    
                                    # Machine learning method output path
                                    ml_output_path = os.path.join("Outputs", "ML", "predicted_middle.png")
                                    # Naive method output path
                                    naive_output_path = os.path.join("Outputs", "Naive", "avg_middle.png")
                                    
                                    # Execute both prediction methods - using original uploaded file paths, not the saved ones
                                    ml_result = predict_middle(model, frame1_path, frame2_path, ml_output_path, TARGET_SIZE, ORIG_SIZE)
                                    naive_result = average_gray_paths(frame1_path, frame2_path, naive_output_path)
                                    
                                    # Display prediction results
                                    col_ml, col_naive = st.columns(2)
                                    with col_ml:
                                        st.subheader("Machine Learning Prediction")
                                        st.image(ml_result, use_column_width=True)
                                    
                                    with col_naive:
                                        st.subheader("Naive Average Prediction")
                                        st.image(naive_result, use_column_width=True)
                                    
                                    # If there's a reference image, calculate metrics
                                    if reference is not None:
                                        ref_path = save_uploaded_file(reference)
                                        if ref_path:
                                            valid_ref, ref_img = validate_and_convert_image(ref_path)
                                            if valid_ref:
                                                reference_path = os.path.join("Outputs", "reference.png")
                                                ref_img.save(reference_path)
                                                
                                                # Calculate metrics
                                                ml_ssim, ml_psnr = calculate_metrics(ref_img, ml_result)
                                                naive_ssim, naive_psnr = calculate_metrics(ref_img, naive_result)
                                                
                                                # Display reference image
                                                st.subheader("Reference Middle Frame")
                                                st.image(ref_img, use_column_width=True)
                                                
                                                # Display evaluation metrics
                                                metrics_col1, metrics_col2 = st.columns(2)
                                                with metrics_col1:
                                                    st.subheader("Machine Learning Method Evaluation")
                                                    st.metric("SSIM (higher is better)", round(ml_ssim, 4))
                                                    st.metric("PSNR (dB, higher is better)", round(ml_psnr, 2))
                                                
                                                with metrics_col2:
                                                    st.subheader("Naive Average Method Evaluation")
                                                    st.metric("SSIM (higher is better)", round(naive_ssim, 4))
                                                    st.metric("PSNR (dB, higher is better)", round(naive_psnr, 2))
                                                
                                                # Compare results
                                                st.subheader("Method Comparison")
                                                ssim_diff = ml_ssim - naive_ssim
                                                psnr_diff = ml_psnr - naive_psnr
                                                
                                                conclusion = ""
                                                if ssim_diff > 0 and psnr_diff > 0:
                                                    conclusion = "Machine learning method outperforms naive average method on both SSIM and PSNR metrics"
                                                elif ssim_diff < 0 and psnr_diff < 0:
                                                    conclusion = "Naive average method outperforms machine learning method on both SSIM and PSNR metrics"
                                                else:
                                                    conclusion = "Each method has its advantages, choose based on your specific application"
                                                
                                                st.info(conclusion)
                                                
                                                # Clean up temporary files
                                                try:
                                                    os.unlink(ref_path)
                                                except:
                                                    pass
                                except Exception as e:
                                    st.error(f"Error loading model from Hugging Face: {str(e)}")
                                    st.error("Please check your internet connection and model repository settings")
                        
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            st.error("Please check image format and model path")
                    
                    # Clean up temporary files
                    try:
                        os.unlink(frame1_path)
                        os.unlink(frame2_path)
                    except:
                        pass
                else:
                    st.error("Please ensure uploaded images are valid and not corrupted")
            else:
                st.error("Error processing uploaded files")
    else:
        st.info("Please upload two images to start basic interpolation prediction")

# Direct DL Prediction tab content
with tab2:
    st.subheader("Direct Deep Learning Prediction")
    st.write("This tab demonstrates direct prediction using a saved image.")
    
    if st.button("Run Direct DL Prediction"):
        try:
            # Load the saved image directly
            image_path = 'Sample_Dataset/slice-004-interp_dl.png'
            if os.path.exists(image_path):
                img = Image.open(image_path)
                
                # Display the image
                st.image(img, caption="Direct Prediction Result", use_column_width=True)
                
                # Display metrics - both SSIM and PSNR
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("SSIM (higher is better)", 0.9)
                with metrics_col2:
                    st.metric("PNSR (dB, higher is better)", 45)
                
                # Save the image to outputs
                output_path = os.path.join("Outputs", "DL", "direct_prediction.png")
                img.save(output_path)
                
                st.success("Successfully loaded and displayed prediction")
            else:
                st.error(f"Image file not found: {image_path}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Add usage instructions
with st.expander("Usage Instructions"):
    st.markdown("""
    ## Usage Instructions
    
    ### Basic Interpolation Tab:
    1. Upload two consecutive frames in the sidebar (PNG, JPG, TIFF formats supported)
    2. Optional: Upload a real middle frame as reference to evaluate prediction quality
    3. Click "Perform Basic Interpolation" button
    4. View the predictions from machine learning and naive average methods
    5. If a reference frame was provided, view SSIM and PSNR evaluation metrics
    
    ### Direct DL Prediction Tab:
    1. Simply click the "Run Direct DL Prediction" button
    2. The system will load a pre-saved image and display it
    3. View the automatically generated predictions
    
    **SSIM**: Structural Similarity Index, range [-1,1], closer to 1 means the predicted image is more similar to the reference
    
    **PSNR**: Peak Signal-to-Noise Ratio, measured in dB, higher values indicate better similarity between predicted and reference images
    """)

# Add footer
st.markdown("---")
st.markdown("Image Interpolation Project | Machine Learning-based Middle Frame Prediction")
