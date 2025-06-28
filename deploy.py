import pickle
import numpy as np
import os
import json
from pathlib import Path

# Load necessary artifacts
try:
    # Load model
    model_path = 'models/best_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names and preprocessing objects
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('data/imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    
    with open('data/feature_selector.pkl', 'rb') as f:
        feature_selector = pickle.load(f)

except Exception as e:
    print(f"Error loading artifacts: {e}")
    exit(1)

# Display the expected feature order
print("\nExpected features in order:")
for i, feature in enumerate(feature_names):
    print(f"{i+1}. {feature}")

# Example input for the selected features
# Based on: Glucose, BMI, Age, Glucose_BMI_Interaction, Pregnancies, BloodPressure, Metabolic_Score
sample_data = {
    'Glucose': 90,
    'BMI': 25.1,
    'Age': 21,
    'Pregnancies': 1,
    'BloodPressure': 62,
    # These will be calculated automatically:
    'Glucose_BMI_Interaction': None,  # Will be calculated as Glucose * BMI
    'Metabolic_Score': None  # Will be calculated as (Glucose * Insulin) / (BMI + 1e-6)
}

# Since we need Insulin to calculate Metabolic_Score, we'll add it temporarily
sample_data['Insulin'] = 66  # This won't be in the final features

# Calculate derived features
sample_data['Glucose_BMI_Interaction'] = sample_data['Glucose'] * sample_data['BMI']
sample_data['Metabolic_Score'] = (sample_data['Glucose'] * sample_data['Insulin']) / (sample_data['BMI'] + 1e-6)

# Create array in correct feature order
sample_values = np.array([[
    sample_data['Glucose'],
    sample_data['BMI'],
    sample_data['Age'],
    sample_data['Glucose_BMI_Interaction'],
    sample_data['Pregnancies'],
    sample_data['BloodPressure'],
    sample_data['Metabolic_Score']
]])

print("\nInput features with values:")
for name, value in zip(feature_names, sample_values[0]):
    print(f"{name}: {value:.2f}")

# Preprocess the input (same pipeline as training)
try:
    # Scale the input
    sample_scaled = scaler.transform(sample_values)
    
    # Make prediction
    prediction = model.predict(sample_scaled)
    prediction_proba = model.predict_proba(sample_scaled)[0][1]  # Probability of being diabetic
    
    # Output result
    print("\nPrediction Result:")
    print(f"Outcome: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
    print(f"Probability: {prediction_proba:.2%}")
    print(f"Confidence: {'High' if prediction_proba > 0.7 or prediction_proba < 0.3 else 'Medium'}")
    
except Exception as e:
    print(f"Error during prediction: {e}")

# Verify working directory
print("\nWorking Directory:", Path(os.getcwd()))
