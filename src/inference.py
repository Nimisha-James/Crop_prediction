import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import argparse
import pathlib
from model import build_model
from feature import extract_all_features  # Import the function from features.py

# OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Constants and other configurations
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224
CLASS_NAMES = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transform

# Annotate image with predicted soil type
def annotate_image(output_class, orig_image):
    class_name = CLASS_NAMES[int(output_class)]
    cv2.putText(
        orig_image, 
        f"{class_name}", 
        (5, 35), 
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5, 
        (0, 255, 255), 
        3, 
        lineType=cv2.LINE_AA
    )
    return orig_image

# Inference and Feature Extraction function
def inference_and_extract_features(model, testloader, device, orig_image):
    model.eval()
    with torch.no_grad():
        image = testloader.to(device)

        # Forward pass
        outputs = model(image)
        predictions = F.softmax(outputs, dim=1).cpu().numpy()
        output_class = np.argmax(predictions)

        # Convert the image to grayscale for feature extraction
        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

        # Extract features from grayscale image
        extracted_features = extract_all_features(gray_image)
        
        # Print the soil type and features
        print(f"Soil Type: {CLASS_NAMES[int(output_class)]}")
        print("Extracted Features:")
        for feature_name, value in extracted_features.items():
            print(f"  {feature_name}: {value}")

        # Annotate the original image with the soil type
        result = annotate_image(output_class, orig_image)

        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights', 
        default='../outputs/best_model.pth',
        help='path to the model weights',
    )
    args = vars(parser.parse_args())
    
    weights_path = pathlib.Path(args['weights'])
    infer_result_path = os.path.join('..', 'outputs', 'inference_results')
    os.makedirs(infer_result_path, exist_ok=True)

    checkpoint = torch.load(weights_path)
    model = build_model(
        fine_tune=False, 
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_image_paths = glob.glob(os.path.join('..', 'input', 'inference_data', '*'))

    transform = get_test_transform(IMAGE_RESIZE)

    for i, image_path in enumerate(all_image_paths):
        print(f"\nInference on image: {i+1}\n")
        image = cv2.imread(image_path)
        orig_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        result = inference_and_extract_features(
            model, 
            image,
            DEVICE,
            orig_image
        )
        
        # Save the image to disk
        image_name = image_path.split(os.path.sep)[-1]
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_name += '.jpg'  # Add a default extension
        cv2.imwrite(
            os.path.join(infer_result_path, image_name), result
        )
    print("\n")
