import os
import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Get current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)

#  Only 5 features (same as training model)
FEATURES = [
    'MinTemp',
    'MaxTemp',
    'Rainfall',
    'Humidity9am',
    'Humidity3pm'
]

LABELS = {0: "NO ", 1: "YES "}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get input values from form
            input_data = [float(request.form[feature]) for feature in FEATURES]
            
            # Convert to numpy array
            input_array = np.array(input_data).reshape(1, -1)

            # Make prediction
            pred = model.predict(input_array)[0]
            prediction = LABELS.get(pred, "Unknown")

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, features=FEATURES)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
