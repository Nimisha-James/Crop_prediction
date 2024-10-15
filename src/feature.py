import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import pywt
import cv2
import os

# OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# GLCM Contrast Feature Extraction
def extract_glcm_features(image):
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the GLCM
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    # Calculate contrast and take the mean across all angles
    contrast = graycoprops(glcm, 'contrast').mean()
    return contrast

# Wavelet Features Extraction (Energy of Approximation Coefficients)
def extract_wavelet_features(image):
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform the discrete wavelet transform
    coeffs = pywt.dwt2(image, 'db1')
    cA, (cH, cV, cD) = coeffs
    # Calculate the energy of the approximation coefficients
    energy = np.sum(np.square(cA))
    return energy

# Entropy Feature Extraction
def extract_entropy(image):
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate Shannon entropy
    entropy_value = shannon_entropy(image)
    return entropy_value

# Color Feature Extraction
def extract_color_features(image):
    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
     # Calculate histograms for each color channel
    hist_r = cv2.calcHist([image_rgb[:, :, 0]], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb[:, :, 1]], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb[:, :, 2]], [0], None, [256], [0, 256])
    
    # Flatten histograms and calculate mean values
    hist_r_flat = hist_r.flatten()
    hist_g_flat = hist_g.flatten()
    hist_b_flat = hist_b.flatten()
    
    hist_r_mean = np.mean(hist_r_flat)
    hist_g_mean = np.mean(hist_g_flat)
    hist_b_mean = np.mean(hist_b_flat)
    
    # Mean of RGB histogram values
    hist_rgb_mean = np.mean([hist_r_mean, hist_g_mean, hist_b_mean])

    # Return color features as a dictionary
    color_features = {
    'Histogram RGB Mean': hist_rgb_mean
    }
    return color_features

# Function to extract all features with detailed names
def extract_all_features(image):
    # Extract texture features
    glcm_contrast = extract_glcm_features(image)
    wavelet_energy = extract_wavelet_features(image)
    entropy_value = extract_entropy(image)
    
    # Extract color features
    color_features = extract_color_features(image)
    
    # Combine all features into a single dictionary
    features = {
        'GLCM Contrast': glcm_contrast,
        'Wavelet Energy': wavelet_energy,
        'Entropy': entropy_value,
    }
    features.update(color_features)  # Add color features to the existing features
    
    return features
