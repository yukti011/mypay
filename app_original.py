from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

mypay = Flask(__name__)
CORS(mypay) 

# Load the saved model
model_path = "best_rf_model (1).pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

@mypay.route('/')
def home():
    return "Welcome to the Random Forest Prediction API!"

@mypay.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Extract features from the request
        features = np.array(data['features']).reshape(1, -1)
        # Make a prediction
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    mypay.run(debug=True)
