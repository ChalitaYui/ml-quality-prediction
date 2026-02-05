import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = {
    "temperature": [30, 32, 31, 35, 36],
    "pressure": [100, 102, 101, 105, 107],
    "quality": [80, 82, 81, 85, 88]
}

df = pd.DataFrame(data)

# Features and target
X = df[["temperature", "pressure"]]
y = df["quality"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
prediction = model.predict([[33, 103]])
print("Predicted quality:", prediction[0])
