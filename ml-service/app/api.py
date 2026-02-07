from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="ML Quality Prediction API")

# โหลดโมเดล (ตอนนี้สมมติว่ามีไฟล์ model.joblib)
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("model.joblib")
        print("Model loaded")
    except Exception as e:
        print("Model not loaded:", e)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.get("/predict")
def predict(x: float):
    if model is None:
        return {"error": "Model not loaded"}

    prediction = model.predict(np.array([[x]]))
    return {
        "input": x,
        "prediction": float(prediction[0])
    }
