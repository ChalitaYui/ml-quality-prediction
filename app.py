from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.joblib")

@app.get("/predict")
def predict(x: float):
    prediction = model.predict(np.array([[x]]))
    return {
        "input": x,
        "prediction": prediction[0]
    }
