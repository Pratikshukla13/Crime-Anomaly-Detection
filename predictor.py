# anomaly_detection/notebooks/predictor.py

import torch
from network.c3d import C3DPretrained
from utils.video_utils import extract_video_features

def predict_crime(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = C3DPretrained()
    model.load_state_dict(torch.load("final_crime_classifier.pth", map_location=device))
    model.eval().to(device)

    features = extract_video_features(video_path).to(device)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

    class_names = ['abuse', 'arson', 'assault', 'arrest']  # adjust to match your labels
    return class_names[predicted.item()]
