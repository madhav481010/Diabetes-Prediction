from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define valid input ranges for server-side validation
valid_ranges = {
    "Pregnancies": (0, 20),
    "Glucose": (50, 200),
    "BloodPressure": (40, 140),
    "SkinThickness": (10, 100),
    "Insulin": (15, 846),
    "BMI": (15, 50),
    "DiabetesPedigreeFunction": (0.1, 2.5),
    "Age": (15, 100),
}

@app.route('/')
def home():
    return render_template('form.html', prediction=None, error=[])

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    error_messages = []

    for key, (min_val, max_val) in valid_ranges.items():
        try:
            value = float(request.form[key])
            if not (min_val <= value <= max_val):
                error_messages.append(f"{key} must be between {min_val} and {max_val}.")
            input_data.append(value)
        except ValueError:
            error_messages.append(f"{key} must be a valid number.")

    if error_messages:
        return render_template('form.html', prediction=None, error=error_messages)

    # Predict
    sample = np.array([input_data])
    prediction = model.predict(sample)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return render_template('form.html', prediction=result, error=[])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
