import os
import joblib
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm']

# 1️⃣ Web route (browser form)
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    advice = None
    if request.method == "POST":
        try:
            input_data = [float(request.form[feature]) for feature in FEATURES]
            input_array = np.array(input_data).reshape(1, -1)
            pred = model.predict(input_array)[0]

            if pred == 1:
                prediction = "Rainfall Expected Tomorrow"
                advice = "Good time for sowing crops. Reduce irrigation and prepare drainage in fields."
            else:
                prediction = "No Rain Expected Tomorrow"
                advice = "Irrigation is required for crops. Plan watering schedule."

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, advice=advice, features=FEATURES)

# 2️⃣ Mobile API route (JSON)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        input_data = [float(data[feature]) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)
        pred = model.predict(input_array)[0]

        if pred == 1:
            prediction = "Rainfall Expected Tomorrow"
            advice = "Good time for sowing crops. Reduce irrigation and prepare drainage in fields."
        else:
            prediction = "No Rain Expected Tomorrow"
            advice = "Irrigation is required for crops. Plan watering schedule."

        return jsonify({"prediction": prediction, "advice": advice})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # 0.0.0.0 allows mobile devices in same Wi-Fi to access
    app.run(host="0.0.0.0", port=5000, debug=True)
