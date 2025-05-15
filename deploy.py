import pickle
import numpy as np
import os

# Absolute paths (current dir)
model_path = 'best_model.pkl'  # Adjusted from models/best_model.pkl
scaler_path = 'scaler.pkl'     # Adjusted from data/scaler.pkl

# Load model and scaler
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Example input
sample = np.array([[1, 90, 62, 12, 66, 25.1, 0.167, 21]])
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print("Predicted Outcome:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
print("Working Directory:", os.getcwd())
