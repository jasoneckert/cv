from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json

app = FastAPI()

# Load latest model
MODEL_DIR = "model_registry/v_20260215_185521"  # Replace with your saved model folder
model = tf.keras.models.load_model(MODEL_DIR)

# Load metadata
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)
IMG_SIZE = metadata["img_size"]
CLASS_NAMES = metadata["class_names"]

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)/255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    bytes_data = await file.read()
    img_array = preprocess_image(bytes_data)
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    return {"prediction": pred_class}