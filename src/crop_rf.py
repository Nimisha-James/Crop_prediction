import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained RandomForest model
rf = joblib.load(r'C:\Swin_Transformer\Crop_prediction\outputs\random_forest_model.pkl')

def preprocess_features(extracted_features):
    columns = ['GLCM Contrast', 'Wavelet Energy', 'Entropy', 'Histogram RGB Mean', 'Soil Type']
    feature_df = pd.DataFrame([extracted_features], columns=columns)

    # Convert categorical 'Soil Type' column to one-hot encoded columns
    feature_df = pd.get_dummies(feature_df, columns=['Soil Type'])

    # Expected columns in the final feature set
    expected_columns = [
        'GLCM Contrast', 'Wavelet Energy', 'Entropy', 'Histogram RGB Mean',
        'Soil Type_Alluvial Soil', 'Soil Type_Black Soil', 'Soil Type_Clay Soil', 'Soil Type_Red Soil'
    ]

    # Ensure all expected columns are present, add missing columns with zero values
    for col in expected_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0

    # Rearrange columns to match the model's input order
    feature_df = feature_df[expected_columns]

    # Scale the numeric columns
    scaler = StandardScaler()
    numeric_columns = ['GLCM Contrast', 'Wavelet Energy', 'Entropy', 'Histogram RGB Mean']
    feature_df[numeric_columns] = scaler.fit_transform(feature_df[numeric_columns])

    return feature_df


def predict_crop(extracted_features):
    # Preprocess the input features
    processed_features = preprocess_features(extracted_features)

    # Predict the crop using the loaded model
    predicted_crop = rf.predict(processed_features)
    
    return predicted_crop[0]
