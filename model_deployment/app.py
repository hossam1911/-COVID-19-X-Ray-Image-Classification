from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import logging
print(tf.__version__)
app = Flask(__name__)

from werkzeug.middleware.shared_data import SharedDataMiddleware
app = Flask(__name__, static_url_path='/static')
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/favicon.ico':  app.config.get('FAVICON', '')
})
app.logger.setLevel(logging.ERROR)
model = tf.keras.models.load_model(r'E:\Covid-19\model_deployment\my_model.h5')

class_names = {
    0: "COVID-19",
    1: "Lung Opacity",
    2: "Normal",
    3: "Viral Pneumonia"
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        # Read the image using PIL
        image = Image.open(file)
        # Preprocess the image (resize, normalize, etc.)
        processed_image = preprocess_image(image)
        # Perform inference using the loaded model
        predicted_class = predict_class(processed_image)
        # Render the result page with the predicted class
        return render_template('result.html', predicted_class=predicted_class)
    return render_template('index.html')

def preprocess_image(image):
    # Resize the image to the expected input size of your model
    image = image.resize((299, 299))
    # Normalize pixel values (if needed)
    image = np.array(image) / 255.0
    # Convert to the expected input format for your model (e.g., add an extra dimension)
    image = np.expand_dims(image, axis=0)
    return image

def predict_class(image):
    # Perform inference using the loaded model
    predictions = model.predict(image)
    # Get the predicted class label (class number)
    predicted_class_number = np.argmax(predictions[0])
    # Get the corresponding class name
    predicted_class_name = class_names[predicted_class_number]
    return predicted_class_name