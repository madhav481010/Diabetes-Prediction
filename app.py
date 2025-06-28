from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Configure paths and load model with error handling
try:
    model_path = os.path.join('models', 'best_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except:
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Clinical ranges for non-diabetic individuals
non_diabetic_ranges = {
    "Pregnancies": (0, 5),
    "Glucose": (70, 99),        # Normal fasting glucose
    "BloodPressure": (60, 80),  # Optimal BP range
    "SkinThickness": (10, 30),  # Lower end of normal
    "Insulin": (3, 25),         # Normal fasting insulin (μU/mL)
    "BMI": (18.5, 24.9),        # Normal weight range
    "DiabetesPedigreeFunction": (0.1, 0.5),  # Lower genetic risk
    "Age": (15, 100),
}

@app.route('/')
def home():
    return render_template('form.html', prediction=None, probability=None, error_messages=[])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('form.html', 
                            prediction="Error",
                            probability="Model not loaded",
                            error_messages=["System error: Prediction model unavailable"])

    input_data = []
    error_messages = []
    non_diabetic_warnings = []

    # Validate inputs and check non-diabetic ranges
    for key, (min_val, max_val) in non_diabetic_ranges.items():
        try:
            value = float(request.form[key])
            
            # Basic validation
            if value < valid_ranges[key][0] or value > valid_ranges[key][1]:
                error_messages.append(f"{key} must be between {valid_ranges[key][0]} and {valid_ranges[key][1]}")
            
            # Non-diabetic range check
            elif value < min_val or value > max_val:
                non_diabetic_warnings.append(f"{key} is outside typical non-diabetic range ({min_val}-{max_val})")
            
            input_data.append(value)
            
        except ValueError:
            error_messages.append(f"{key} must be a valid number.")

    if error_messages:
        return render_template('form.html', 
                             prediction=None,
                             probability=None,
                             error_messages=error_messages)

    # Prepare input and predict
    sample = np.array([input_data])
    
    try:
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        
        # If model says diabetic but values suggest otherwise
        if prediction == 1 and len(non_diabetic_warnings) == 0:
            non_diabetic_warnings.append("Model predicted diabetic despite normal values - please verify with a doctor")
        
        return render_template('form.html',
                            prediction=result,
                            probability=f"{probability:.1%}",
                            non_diabetic_warnings=non_diabetic_warnings,
                            error_messages=[])
    
    except Exception as e:
        error_messages.append(f"Prediction error: {str(e)}")
        return render_template('form.html',
                            prediction=None,
                            probability=None,
                            error_messages=error_messages)

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    app.run(host='0.0.0.0', port=5000)
