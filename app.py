from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the root directory (not models/)
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # Show form with no prediction initially
    return render_template('form.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form and convert to float
    input_data = [float(x) for x in request.form.values()]
    sample = np.array([input_data])

    # No scaler used — directly predict
    prediction = model.predict(sample)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    # Show result on the form
    return render_template('form.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
