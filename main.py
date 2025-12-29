from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Input schema
class DogData(BaseModel):
    walk_minutes: float
    play_minutes: float
    meal_quality: float

@app.get("/")
def root():
    return {"message": "Dog Happiness API is running 🐶"}

@app.post("/predict")
def predict(data: DogData):
    X = np.array([[data.walk_minutes, data.play_minutes, data.meal_quality]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return {"happy": int(prediction[0])}
