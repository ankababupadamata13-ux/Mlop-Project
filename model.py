import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dummy dataset (for testing)
data = {
    'MinTemp': [10, 20, 15, 25],
    'MaxTemp': [20, 30, 25, 35],
    'Rainfall': [0, 5, 0, 10],
    'Humidity9am': [60, 80, 65, 90],
    'Humidity3pm': [55, 85, 60, 88],
    'RainTomorrow': [0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm']]
y = df['RainTomorrow']

model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and model.pkl created successfully ✅")
