# Document-Scanner-Text-Classifier

AI-powered document automation system that classifies scanned documents and PDFs into categories such as invoices, receipts, and contracts.

## Features
- Classifies documents into predefined categories
- Utilizes a deep learning model for classification
- FastAPI backend serving a PyTorch model
- Streamlit frontend for easy document upload and classification
- Achieves over 90% accuracy on validation data

## Setup

1. Create and activate a Python virtual environment
   ```
   python3 -m venv venv
   # For Mac source venv/bin/activate  
   # For Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Backend API

Start the FastAPI server:
```bash
python -m uvicorn app.api:app --reload
```

API docs available at http://localhost:8000/docs

## Running the Frontend

Start the Streamlit app:
```bash
streamlit run app/main.py
```

Upload documents (PNG, JPG, JPEG) to classify them.

## Project Structure

- `app/` - Backend and frontend code
- `app/models/` - Trained PyTorch model
- `data/` - Dataset folders for training, validation, and testing
- `train_model.py` - Training script

## Notes

- If model trained using "train_model.py" ensure the model is trained fully through all epochs for best accuracy.

Feel free to contribute or raise issues.
