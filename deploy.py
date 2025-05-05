# deploy.py
import pickle
import numpy as np

with open('models/best_model.pkl', 'rb') as f: model = pickle.load(f)
with open('data/scaler.pkl', 'rb') as f: scaler = pickle.load(f)

# Example input: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
sample = np.array([[1, 90, 62, 12, 66, 25.1, 0.167, 21]])
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print(" Predicted Outcome:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
