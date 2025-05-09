from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    # Return form with no prediction when loading the homepage
    return render_template('form.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = [float(x) for x in request.form.values()]
    sample = np.array([input_data])
    sample_scaled = scaler.transform(sample)

    # Make prediction
    prediction = model.predict(sample_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    # Return the form again with the prediction result
    return render_template('form.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
