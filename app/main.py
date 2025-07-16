import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# Load model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSES = ['contract', 'invoice', 'receipt']
    
    def load_model(path="app/models/trained_model.pth", num_classes=len(CLASSES)):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model

    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Main interface
st.title('Document Classifier')

# Upload image
uploaded_file = st.file_uploader("Upload your document", type=['png','jpg','jpeg',])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Document uploaded', use_container_width=True)
        
        
 
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
 
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
 
        prediction = [CLASSES[predicted_idx]]
        with col2:
            st.success(f"Document type: {prediction[0]}")
            st.metric("Accuracy", f"{confidence:.2%}")

    except Exception as e:
        st.error(f"Error processing the document: {e}")

# Model info
st.sidebar.header("About the model")
st.sidebar.write("Suported Categories:")
for category in ['Contract', 'Invoice', 'Receipt']:
    st.sidebar.write(f"- {category}")
