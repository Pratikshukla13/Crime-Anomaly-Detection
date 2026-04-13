from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import tempfile

app = FastAPI()

model = torch.jit.load("crime_classifier_scripted.pt")
model.eval()

class_names = ['Abuse', 'Arrest', 'Arson', 'Assault']

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // max_frames, 1)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = transform(image)
        frames.append(image)
        if len(frames) == max_frames:
            break

    cap.release()
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))

    return torch.stack(frames)

@app.post("/predict")
async def predict_crime(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    input_tensor = extract_frames(temp_video_path).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]

    os.remove(temp_video_path)

    report = {
        "crime_type": predicted_label,
        "summary": f"Suspected case of {predicted_label} detected in the submitted video footage.",
        "recommendation": "Further investigation is recommended by the concerned law enforcement authority."
    }

    return JSONResponse(content=report)

# To run: uvicorn server:app --reload