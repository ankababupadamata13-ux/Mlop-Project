import os
import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

FEATURES = [
    'MinTemp',
    'MaxTemp',
    'Rainfall',
    'Humidity9am',
    'Humidity3pm'
]

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    advice = None

    if request.method == "POST":
        try:

            input_data = [float(request.form[feature]) for feature in FEATURES]
            input_array = np.array(input_data).reshape(1, -1)

            pred = model.predict(input_array)[0]

            # Farmer friendly result
            if pred == 1:
                prediction = " Rainfall Expected Tomorrow"
                advice = "Good time for sowing crops. Reduce irrigation and prepare drainage in fields."
            else:
                prediction = " No Rain Expected Tomorrow"
                advice = "Irrigation is required for crops. Plan watering schedule."

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        advice=advice,
        features=FEATURES
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

