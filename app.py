from flask import Flask, request, render_template
import pickle
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)

# Hardcode the expected feature order since feature_names.pkl is missing
FEATURE_ORDER = [
    'Glucose',
    'BMI',
    'Age',
    'Glucose_BMI_Interaction',
    'Pregnancies',
    'BloodPressure',
    'Metabolic_Score'
]

# Define valid input ranges for the raw features we'll collect
VALID_RANGES = {
    "Glucose": (50, 200),          # mg/dL
    "BMI": (15, 50),               # kg/m²
    "Age": (15, 100),              # years
    "Pregnancies": (0, 20),        # number
    "BloodPressure": (40, 140),    # mmHg
    "Insulin": (15, 846)           # pmol/L (needed for Metabolic_Score)
}

def load_model_with_fallback():
    """Try multiple possible paths for model and scaler"""
    model_paths = [
        'models/best_model.pkl',
        'best_model.pkl',
        os.path.join('app', 'models', 'best_model.pkl')
    ]
    
    scaler_paths = [
        'data/scaler.pkl',
        'scaler.pkl',
        os.path.join('app', 'data', 'scaler.pkl')
    ]
    
    # Try loading model
    model = None
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
                break
        except:
            continue
    
    # Try loading scaler
    scaler = None
    for path in scaler_paths:
        try:
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
                break
        except:
            continue
    
    if model is None:
        raise FileNotFoundError("Could not find model file in any standard location")
    
    return model, scaler

# Load model with fallback paths
try:
    model, scaler = load_model_with_fallback()
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    # Create a dummy model if running in development
    if os.getenv('FLASK_ENV') == 'development':
        print("Creating dummy model for development")
        model = lambda x: np.array([0])  # Dummy model that always returns "Not Diabetic"
        scaler = lambda x: x  # Identity scaler
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

    # Feature engineering (same as during training)
    input_data['Glucose_BMI_Interaction'] = input_data['Glucose'] * input_data['BMI']
    input_data['Metabolic_Score'] = (input_data['Glucose'] * input_data['Insulin']) / (input_data['BMI'] + 1e-6)

    # Create array in correct feature order
    sample_values = np.array([[
        input_data['Glucose'],
        input_data['BMI'],
        input_data['Age'],
        input_data['Glucose_BMI_Interaction'],
        input_data['Pregnancies'],
        input_data['BloodPressure'],
        input_data['Metabolic_Score']
    ]])

    # Preprocess and predict
    try:
        if scaler is not None:
            sample_scaled = scaler.transform(sample_values)
        else:
            sample_scaled = sample_values  # Fallback if scaler not found
            
        prediction = model.predict(sample_scaled)
        
        # Handle different model types (some may not have predict_proba)
        try:
            probability = model.predict_proba(sample_scaled)[0][1]
        except AttributeError:
            probability = 0.5 if prediction[0] == 1 else 0.5  # Default if no probabilities
        
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
