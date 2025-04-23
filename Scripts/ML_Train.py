import os
import re
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from Scripts.ML_Func import *
from joblib import load

# Directory containing image slices (slice-001.png, slice-003.png, ..., slice-011.png)
IMAGE_DIR          = 'Dataset_images/'     
TARGET_SIZE        = (256, 256)            # Resize images to this size for processing
SAMPLES_PER_TRIPLE = 2000                  # Number of pixel samples per frame triple
MODEL_PATH         = 'Model/interp_rf.pkl' # Path to save the trained model
TEST_SIZE          = 0.2                   # Proportion of data to use for testing
ORIG_SIZE          = (868, 1551)           # Original image size

# Train and test the model
# Build dataset from image triples for frame interpolation
X_train, X_test, y_train, y_test = build_dataset(IMAGE_DIR, TARGET_SIZE, SAMPLES_PER_TRIPLE, TEST_SIZE)
# Train the Random Forest regression model
model = train_model(X_train, y_train, MODEL_PATH)
# Evaluate model performance on test data
y_pred, mse = test_model(model, X_test, y_test)

