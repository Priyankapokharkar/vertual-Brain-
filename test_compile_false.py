import sys
print("Importing keras...")
from tensorflow import keras
from pathlib import Path

model_path = Path("D:/Projects 2025/Final Year projects/Vertual brain/models/pretrained/multi_disease_detector.h5")
print(f"Loading model from {model_path} with compile=False...")
model = keras.models.load_model(str(model_path), compile=False)
print("Model loaded successfully!")
