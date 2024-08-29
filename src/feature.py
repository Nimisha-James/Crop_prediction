import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image
import os
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops

def process_soil_image(image_array, soil_type):
    # Wavelet transform of the image, and plot approximation and details
    titles = ['Approximation', 'Horizontal detail', 'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(image_array, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, (ax, a) in enumerate(zip(axes, [LL, LH, HL, HH])):
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.suptitle(f"{soil_type}", fontsize=12)
    plt.show(block=False)
    plt.pause(5)  # Display for 1 second
    plt.close()

    # Extract features based on soil type
    print(f"Soil Type: {soil_type}")
    if soil_type == "Alluvial Soil":
        grain_size_distribution = np.mean([region.equivalent_diameter for region in regionprops(label(closing(image_array > threshold_otsu(image_array), square(3))))])
        moisture_content = shannon_entropy(image_array)
        print(f"Grain Size Distribution: {grain_size_distribution:.2f} pixels")
        print(f"Moisture Content: {moisture_content:.2f}")
        
    elif soil_type == "Black Soil":
        contrast = graycoprops(graycomatrix(image_array, distances=[1], angles=[0], levels=256, symmetric=True, normed=True), 'contrast')[0, 0]
        cracking_patterns = np.mean([region.equivalent_diameter for region in regionprops(label(closing(image_array > threshold_otsu(image_array), square(3))))])
        print(f"Color Contrast: {contrast:.2f}")
        print(f"Cracking Patterns: {cracking_patterns:.2f} pixels")
        
    elif soil_type == "Clay Soil":
        texture = graycoprops(graycomatrix(image_array, distances=[1], angles=[0], levels=256, symmetric=True, normed=True), 'homogeneity')[0, 0]
        porosity = shannon_entropy(image_array)
        print(f"Texture: {texture:.2f}")
        print(f"Porosity: {porosity:.2f}")
        
    elif soil_type == "Red Soil":
        color_intensity = graycoprops(graycomatrix(image_array, distances=[1], angles=[0], levels=256, symmetric=True, normed=True), 'energy')[0, 0]
        grain_size = np.mean([region.equivalent_diameter for region in regionprops(label(closing(image_array > threshold_otsu(image_array), square(3))))])
        print(f"Color Intensity: {color_intensity:.2f}")
        print(f"Grain Size: {grain_size:.2f} pixels")
    
    print("-" * 40)
