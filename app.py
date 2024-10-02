from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5', compile=False)

# Define a function to preprocess the uploaded image
def process_image(image_path):
      image = load_img(image_path, target_size=(256, 256)) 
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Normalize pixel values to the range [0, 1]
    image = image.astype('float32') / 255.0
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Save the uploaded image to a local directory
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    try:
        processed_image = process_image(image_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_labels = {0: 'cat', 1: 'dog'}  # Ensure this matches your training labels
        classification = class_labels.get(predicted_class, "Unknown")
        os.remove(image_path)

        return jsonify({'prediction': classification})  
    except Exception as e:
        return jsonify({'error': str(e)}) 

if __name__ == '__main__':
    app.run(port=3000, debug=True)
