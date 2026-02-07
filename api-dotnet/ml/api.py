from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# โหลดโมเดล
model = joblib.load("model.joblib")

class PredictRequest(BaseModel):
    value: float

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array([[req.value]])
    result = model.predict(X)
    return {
        "input": req.value,
        "prediction": result[0]
    }
