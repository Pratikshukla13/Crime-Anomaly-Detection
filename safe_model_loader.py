import torch
from network.anomaly_detector_model import AnomalyDetector

def register_anomaly_detector():
    torch.serialization.add_safe_globals({'AnomalyDetector': AnomalyDetector})
    return AnomalyDetector
