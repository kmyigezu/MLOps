from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import os
import numpy as np
from typing import List

app = FastAPI(title="Decision Tree Classifier API")

model = None

def load_model():
    global model
    if model is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(tracking_uri)
        
        model_name = "best"
        model_version = "1"
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"Loading model from: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        
    return model

class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float
    feature11: float
    feature12: float
    feature13: float

class BatchModelInput(BaseModel):
    inputs: List[ModelInput]

@app.on_event("startup")
async def startup_event():
    load_model()
    print("Model has been loaded")

@app.get("/")
def read_root():
    return {"message": "Decision Tree Classifier API"}

@app.post("/predict/")
def predict_single(data: ModelInput):
    model = load_model()
    features = np.array([[
        data.feature1, data.feature2, data.feature3, data.feature4, 
        data.feature5, data.feature6, data.feature7, data.feature8, 
        data.feature9, data.feature10, data.feature11, data.feature12, 
        data.feature13
    ]])
    
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction), "class": int(prediction)}