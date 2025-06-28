from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and artifacts
try:
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    exit(1)

# Define valid input ranges for the raw features we'll collect
valid_ranges = {
    "Glucose": (50, 200),          # mg/dL
    "BMI": (15, 50),               # kg/m²
    "Age": (15, 100),              # years
    "Pregnancies": (0, 20),        # number
    "BloodPressure": (40, 140),    # mmHg
    "Insulin": (15, 846)           # pmol/L (needed for Metabolic_Score)
}

@app.route('/')
def home():
    return render_template('form.html', prediction=None, probability=None, error_messages=[])

@app.route('/predict', methods=['POST'])
def predict():
    # Collect and validate raw input
    input_data = {}
    error_messages = []

    for feature, (min_val, max_val) in valid_ranges.items():
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
        sample_scaled = scaler.transform(sample_values)
        prediction = model.predict(sample_scaled)
        probability = model.predict_proba(sample_scaled)[0][1]  # Probability of being diabetic
        
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
