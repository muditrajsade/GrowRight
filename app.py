
import numpy as np
import os
from flask import Flask, request, render_template
import pickle
import os
from tensorflow.keras.models import load_model
app = Flask(__name__)


# Paths
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'crop_model.h5')
scaler_path = os.path.join(base_path, 'scaler.pkl')
encoder_path = os.path.join(base_path, 'label_encoder.pkl')

# Load model and preprocessing tools
try:
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
except Exception as e:
    raise Exception(f"Error loading model or tools: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/FindYourCrop', methods=['GET', 'POST'])
def FindYourCrop():
    if request.method == 'POST':
        try:
            # Get form input
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Log transform 'K'
            K_log = np.log1p(K)

            # Feature array
            features = np.array([[N, P, K_log, temperature, humidity, ph, rainfall]])

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            prediction_prob = model.predict(features_scaled)
            predicted_class = np.argmax(prediction_prob)
            predicted_crop = encoder.inverse_transform([predicted_class])[0]

            return render_template('FindYourCrop.html', prediction=predicted_crop)

        except Exception as e:
            return render_template('FindYourCrop.html', prediction=f"⚠️ Error: {e}")
    
    return render_template('FindYourCrop.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # get port from env or default 5000
    app.run(host="0.0.0.0", port=port, debug=True)