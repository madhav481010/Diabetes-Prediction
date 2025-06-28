from flask import Flask, request, render_template
import pickle
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)

# Define the expected raw features and their valid ranges
VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (50, 200),
    "BloodPressure": (40, 140),
    "Insulin": (15, 846),
    "BMI": (15, 50),
    "Age": (15, 100)
}

def load_model_with_fallback():
    """Try multiple possible paths for model"""
    model_paths = [
        'models/best_model.pkl',
        'best_model.pkl',
        os.path.join('app', 'models', 'best_model.pkl')
    ]
    
    # Try loading model
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            continue
    
    raise FileNotFoundError("Could not find model file in any standard location")

# Load model with fallback paths
try:
    model = load_model_with_fallback()
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy model if running in development
    if os.getenv('FLASK_ENV') == 'development':
        print("Creating dummy model for development")
        model = lambda x: np.array([0])  # Dummy model
    else:
        exit(1)

@app.route('/')
def home():
    return render_template('form.html', prediction=None, probability=None, error_messages=[])

@app.route('/predict', methods=['POST'])
def predict():
    # Collect and validate raw input
    input_data = {}
    error_messages = []

    for feature, (min_val, max_val) in VALID_RANGES.items():
        try:
            value = float(request.form.get(feature, 0))
            if not (min_val <= value <= max_val):
                error_messages.append(f"{feature} must be between {min_val} and {max_val}")
            input_data[feature] = value
        except ValueError:
            error_messages.append(f"{feature} must be a valid number")

    if error_messages:
        return render_template('form.html', 
                            prediction=None, 
                            probability=None,
                            error_messages=error_messages)

    # Feature engineering
    input_data['Glucose_BMI_Interaction'] = input_data['Glucose'] * input_data['BMI']
    input_data['Metabolic_Score'] = (input_data['Glucose'] * input_data['Insulin']) / (input_data['BMI'] + 1e-6)

    # Create array in the order the model expects
    sample_values = np.array([[
        input_data['Glucose'],
        input_data['BMI'],
        input_data['Age'],
        input_data['Glucose_BMI_Interaction'],
        input_data['Pregnancies'],
        input_data['BloodPressure'],
        input_data['Metabolic_Score']
    ]])

    # Predict
    try:
        prediction = model.predict(sample_values)
        
        # Handle probability if available
        try:
            probability = model.predict_proba(sample_values)[0][1]
        except AttributeError:
            probability = 0.5 if prediction[0] == 1 else 0.5
        
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return render_template('form.html', 
                            prediction=result,
                            probability=f"{probability:.1%}",
                            error_messages=[])
    
    except Exception as e:
        error_messages.append(f"Prediction error: {str(e)}")
        return render_template('form.html', 
                            prediction=None,
                            probability=None,
                            error_messages=error_messages)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
