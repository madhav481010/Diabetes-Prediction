import pickle
import numpy as np
import os

# Path to the trained model
model_path = 'best_model.pkl'

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Example input: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
sample = np.array([[1, 90, 62, 12, 66, 25.1, 0.167, 21]])

# Make prediction
prediction = model.predict(sample)

# Output result
print("Predicted Outcome:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
print("Working Directory:", os.getcwd())
