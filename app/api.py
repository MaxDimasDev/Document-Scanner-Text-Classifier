from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import io
import os

app = FastAPI(title="Document Classification API")

# Clases same as in train_model.py
CLASSES = ['contract', 'invoice', 'receipt']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to load model
def load_model(path="app/models/trained_model.pth", num_classes=len(CLASSES)):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Test endpoint
@app.get("/")
def read_root():
    return {"message": "API de Clasificación de Documentos"}

# Predict/Main endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image)
        input_tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_idx = int(predicted.item())

            label = CLASSES[predicted_idx]

            probabilities = torch.softmax(outputs, dim=1)
            confidence = float(probabilities[0][predicted_idx].item())

        return JSONResponse({
            "predicted_class": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")
        