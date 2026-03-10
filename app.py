from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    time = float(request.form['time'])
    amount = float(request.form['amount'])
    
    # Create a feature array, including default values for the hidden fields
    features = [time, amount]
    for i in range(1, 29):
        features.append(float(request.form.get(f'V{i}', 0)))

    # Convert to numpy array and reshape for prediction
    features_array = np.array([features])

    # Make prediction
    prediction = model.predict(features_array)
    prediction_text = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'

    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
