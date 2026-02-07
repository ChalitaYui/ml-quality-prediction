import joblib
import numpy as np

model = joblib.load("model.joblib")
X_test = np.array([[6]])
pred = model.predict(X_test)

print("Prediction:", pred)
