import os
import keras
from keras.preprocessing import image # type: ignore
import numpy as np
from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
from io import BytesIO
import base64

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
CORS(app)  # This will enable CORS for all routes

# Load the saved model
model = keras.models.load_model('final_model.keras')

# Function to preprocess and predict on a single image
def preprocess_image(img_data):
    img = image.load_img(BytesIO(img_data), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, img_data):
    img_array = preprocess_image(img_data)
    prediction = model.predict(img_array)
    return prediction


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    img_file = request.files['image']
    img_data = img_file.read()

    prediction = predict_image(model, img_data)
    classes = ['Chickenpox', 'Healthy', 'Measles', 'Monkeypox']
    predicted_class = classes[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
   app.run()
