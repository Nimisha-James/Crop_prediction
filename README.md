# Crop Prediction using Soil Imaging

This repository contains a project that uses a Swin Transformer model to classify soil images into four types: **Alluvial**, **Black**, **Red**, and **Clay**. After classifying the soil type, the model further extracts texture and other features to predict the most suitable crop for that soil type using numerical data.

## Project Overview

### 1. **Soil Image Classification**
   - The Swin Transformer model is trained to classify soil images into one of four types:
     - **Alluvial**
     - **Black**
     - **Red**
     - **Clay**
   - The model outputs the detected soil type based on the input image.

### 2. **Feature Extraction**
   - Once the soil type is identified, the model extracts specific texture and structural features from the soil image.
   - These features are crucial in understanding the soil's suitability for different crops.

### 3. **Crop Prediction**
   - The extracted features, along with additional numerical data (e.g., soil pH, moisture content, etc.), are used to predict the most suitable crop for the detected soil type.
   - This is done using a secondary model trained on agricultural data to ensure the most accurate crop recommendations.

## Installation

### Prerequisites
- Python 3.x
- PyTorch
- PyWavelets
- scikit-learn
- NumPy
- OpenCV

### Setting Up the Environment
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Nimisha-James/Crop_prediction
   cd Crop_prediction
   
