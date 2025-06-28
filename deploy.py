import pickle
import numpy as np
import os
from pathlib import Path

# Define the expected feature order based on our feature engineering
FEATURE_ORDER = [
    'Glucose',
    'BMI',
    'Age',
    'Glucose_BMI_Interaction',
    'Pregnancies',
    'BloodPressure',
    'Metabolic_Score'
]

def load_model(model_path):
    """Safely load the trained model"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def calculate_features(raw_input):
    """Perform feature engineering on raw input data"""
    features = {
        'Glucose': raw_input[1],
        'BMI': raw_input[5],
        'Age': raw_input[7],
        'Pregnancies': raw_input[0],
        'BloodPressure': raw_input[2],
        'Insulin': raw_input[4]  # Needed for Metabolic_Score calculation
    }
    
    # Calculate derived features
    features['Glucose_BMI_Interaction'] = features['Glucose'] * features['BMI']
    features['Metabolic_Score'] = (features['Glucose'] * features['Insulin']) / (features['BMI'] + 1e-6)
    
    # Return features in correct order
    return [features[feature] for feature in FEATURE_ORDER]

def main():
    # Path to the trained model - using relative path with fallback
    model_path = os.path.join('models', 'best_model.pkl')
    if not os.path.exists(model_path):
        model_path = 'best_model.pkl'  # Fallback to local directory
    
    # Load model
    model = load_model(model_path)
    
    # Example input in original format: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
    # Note: SkinThickness and DPF won't be used in our feature-engineered model
    sample = np.array([[1, 90, 62, 12, 66, 25.1, 0.167, 21]])
    
    # Perform feature engineering
    engineered_features = calculate_features(sample[0])
    sample_engineered = np.array([engineered_features])
    
    # Make prediction
    try:
        prediction = model.predict(sample_engineered)
        prediction_proba = model.predict_proba(sample_engineered)[0][1]
        
        # Output result
        print("\nPrediction Result:")
        print(f"Outcome: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
        print(f"Probability: {prediction_proba:.2%}")
        print(f"Confidence: {'High' if prediction_proba > 0.7 or prediction_proba < 0.3 else 'Medium'}")
        
    except Exception as e:
        print(f"Prediction error: {e}")
    
    print("\nWorking Directory:", Path(os.getcwd()))

if __name__ == '__main__':
    main()
