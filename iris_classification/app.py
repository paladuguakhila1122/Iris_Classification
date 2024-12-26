from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("iris_classifier_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
label_encoder = joblib.load("iris_label_encoder.pkl")  # Save this during preprocessing

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]
        # Preprocess the input
        sample = np.array(data).reshape(1, -1)
        sample = scaler.transform(sample)
        
        # Make prediction
        prediction = model.predict(sample)
        species = label_encoder.inverse_transform(prediction)
        
        return jsonify({'species': species[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
