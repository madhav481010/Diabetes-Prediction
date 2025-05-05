from flask import Flask, request, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = [float(x) for x in request.form.values()]
    sample = np.array([input_data])
    sample_scaled = scaler.transform(sample)

    # Make prediction
    prediction = model.predict(sample_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return send_from_directory(os.getcwd(), 'index.html') + f"<h2>Prediction: {result}</h2>"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
