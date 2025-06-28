from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model with fallback
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Define input ranges and feature engineering
FEATURES = {
    # Raw features to collect from user
    'raw': {
        'Pregnancies': (0, 20),
        'Glucose': (50, 200),
        'BloodPressure': (40, 140),
        'Insulin': (15, 846),
        'BMI': (15, 50),
        'Age': (15, 100)
    },
    # Features to engineer
    'engineered': {
        'Glucose_BMI_Interaction': lambda x: x['Glucose'] * x['BMI'],
        'Metabolic_Score': lambda x: (x['Glucose'] * x['Insulin']) / (x['BMI'] + 1e-6)
    },
    # Final feature order expected by model
    'order': [
        'Glucose', 'BMI', 'Age', 
        'Glucose_BMI_Interaction',
        'Pregnancies', 'BloodPressure',
        'Metabolic_Score'
    ]
}

@app.route('/')
def home():
    return render_template('diabetes_form.html', 
                         prediction=None, 
                         probability=None,
                         error_messages=[])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('diabetes_form.html',
                            prediction="Error",
                            probability="Model not loaded",
                            error_messages=["System error: Model unavailable"])

    # Collect and validate inputs
    input_data = {}
    error_messages = []

    for feature, (min_val, max_val) in FEATURES['raw'].items():
        try:
            value = float(request.form.get(feature, 0))
            if not (min_val <= value <= max_val):
                error_messages.append(f"{feature} must be between {min_val} and {max_val}")
            input_data[feature] = value
        except ValueError:
            error_messages.append(f"{feature} must be a valid number")

    if error_messages:
        return render_template('diabetes_form.html',
                            prediction=None,
                            probability=None,
                            error_messages=error_messages)

    # Feature engineering
    for feature, func in FEATURES['engineered'].items():
        input_data[feature] = func(input_data)

    # Prepare final input array
    sample = np.array([[input_data[feature] for feature in FEATURES['order']]])

    # Make prediction
    try:
        prediction = model.predict(sample)[0]
        try:
            probability = model.predict_proba(sample)[0][1]
        except AttributeError:
            probability = 0.75 if prediction == 1 else 0.25

        result = {
            'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'probability': f"{probability:.1%}",
            'confidence': 'High' if probability > 0.7 or probability < 0.3 else 'Medium',
            'features': input_data
        }

        return render_template('diabetes_form.html',
                            prediction=result['prediction'],
                            probability=result['probability'],
                            confidence=result['confidence'],
                            features=result['features'],
                            error_messages=[])

    except Exception as e:
        error_messages.append(f"Prediction error: {str(e)}")
        return render_template('diabetes_form.html',
                            prediction=None,
                            probability=None,
                            error_messages=error_messages)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
