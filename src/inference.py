import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import argparse
import pathlib
import warnings
from model import build_model
from feature import extract_all_features
from crop_rf import predict_crop

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224
CLASS_NAMES = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

warnings.filterwarnings("ignore", category=FutureWarning)

def get_test_transform(image_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

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

def inference_and_extract_features(model, testloader, device, orig_image,x):
    model.eval()
    with torch.no_grad():
        image = testloader.to(device)

        outputs = model(image)
        predictions = F.softmax(outputs, dim=1).cpu().numpy()
        output_class = np.argmax(predictions)

        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

        extracted_features = extract_all_features(gray_image)
        soil_type = {'Soil Type': CLASS_NAMES[int(output_class)]}
        extracted_features.update(soil_type)
        
        # Print the soil type and features
        soil_type_str = f"Soil Type: {CLASS_NAMES[int(output_class)]}"
        print(soil_type_str)
        print("Extracted Features:")
        extracted_features_str = ""
        for feature_name, value in extracted_features.items():
            feature_line = f"  {feature_name}: {value}"
            print(feature_line)
            extracted_features_str += feature_line + "\n"

        # Pass the extracted features to the Random Forest for crop prediction
        predicted_crop = predict_crop(extracted_features)

        predicted_crop_str = f"\nPredicted Crop: {predicted_crop}"
        print(predicted_crop_str)

        # Save extracted features and prediction to output.txt
        with open(output_file_path, 'a') as f:
            f.write(f"\n\nInference on image: {x}\n\n")
            f.write(f"{soil_type_str}\n")
            f.write("Extracted Features:\n")
            extracted_features_str = ""
            for feature_name, value in extracted_features.items():
                feature_line = f"  {feature_name}: {value}"
                f.write(f"{feature_line}\n")
                extracted_features_str += feature_line + "\n"
            f.write(predicted_crop_str + "\n")
            f.write("____________________________________________________________________________________________________________________________")

        result = annotate_image(output_class, orig_image)

        return result

if __name__ == '__main__':
    
    output_file_path = os.path.join('..', 'outputs', 'crop_output.txt')
    with open(output_file_path, 'w') as f:
        f.write("")
        
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
    model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE)
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
        
        result = inference_and_extract_features(model, image, DEVICE, orig_image,i+1)
        print("______________________________________________________________________________________________________________")

        image_name = image_path.split(os.path.sep)[-1]
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_name += '.jpg'
        cv2.imwrite(os.path.join(infer_result_path, image_name), result)

