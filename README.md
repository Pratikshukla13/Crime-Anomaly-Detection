# Crime Anomaly Detection in Surveillance Video

A comprehensive, end-to-end deep learning system designed to identify criminal activities (theft, violence, etc.) in real-time surveillance feeds. This project features a modular architecture including custom data loaders, model definitions, and a prediction server.

## 🚀 Key Features
- **Temporal Analysis:** Specialized `CrimeVideoDataset.py` for handling sequential video frames.
- **Modular Design:** Separate components for model architecture (`model.py`), training (`train.ipynb`), and inference (`predictor.py`).
- **Deployment Ready:** Includes `server.py` to handle remote requests for anomaly scoring.
- **Safety First:** Implements `safe_model_loader.py` to ensure weight integrity during deployment.

## 📁 Repository Structure
- `anomaly_detection.ipynb`: Main research and evaluation notebook.
- `classifier.py`: Implementation of the anomaly classification head.
- `crime_descriptions.py`: Mapping of anomaly scores to human-readable crime categories.
- `train_classifier.py`: Scripted training pipeline for production environments.

## 🛠️ Tech Stack
- **Framework:** PyTorch (Deep Learning)
- **Deployment:** Flask/FastAPI (Server-side)
- **Data:** OpenCV (Video processing)

## ⚙️ Setup
1. **Clone & Install:**
   ```bash
   git clone [https://github.com/Pratikshukla13/Crime-Anomaly-Detection.git](https://github.com/Pratikshukla13/Crime-Anomaly-Detection.git)
   pip install -r requirements.txt
