from flask import Flask, request, render_template, send_from_directory
import pickle
import numpy as np
import json
from pathlib import Path

app = Flask(__name__)

# Load model and metrics
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_metrics.json', 'rb') as f:
    metrics_data = json.load(f)

# Prepare model cards
model_cards = []
for name, data in metrics_data.items():
    model_cards.append({
        'name': name,
        'metrics': data,
        'roc_image': f"roc_{name.lower().replace(' ', '_')}.png"
    })

# Validation ranges
valid_ranges = {
    "Pregnancies": (0, 20),
    "Glucose": (50, 200),
    "BloodPressure": (40, 140),
    "SkinThickness": (10, 100),
    "Insulin": (15, 846),
    "BMI": (15, 50),
    "DiabetesPedigreeFunction": (0.1, 2.5),
    "Age": (15, 100),
}

@app.route('/')
def home():
    return render_template('combined.html', prediction=None, error=[], models=model_cards)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    error_messages = []
    form_data = {}

    for key, (min_val, max_val) in valid_ranges.items():
        try:
            value = float(request.form[key])
            form_data[key] = value
            if not (min_val <= value <= max_val):
                error_messages.append(f"{key} must be between {min_val} and {max_val}.")
            input_data.append(value)
        except ValueError:
            error_messages.append(f"{key} must be a valid number.")
            form_data[key] = ''

    if error_messages:
        return render_template('combined.html', 
                            prediction=None, 
                            error=error_messages, 
                            models=model_cards,
                            input_data=form_data)

    # Predict
    sample = np.array([input_data])
    prediction = model.predict(sample)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    return render_template('combined.html', 
                        prediction=result, 
                        error=[],
                        models=model_cards,
                        input_data=form_data)

@app.route('/roc_images/<filename>')
def roc_images(filename):
    return send_from_directory('artifacts', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
