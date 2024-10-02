from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your custom-trained model (ensure the model file is in the correct location)
model = load_model('model.h5', compile=False)

# Define a function to preprocess the uploaded image
def process_image(image_path):
    """
    Preprocess the image to match the input format expected by the model.
    """
    # Load the image file and resize it to match the model's expected input size
    image = load_img(image_path, target_size=(256, 256))  # Ensure target_size matches model's input layer
    # Convert the image to a numpy array
    image = img_to_array(image)
    # Reshape the image to add the batch dimension
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
        # Preprocess the uploaded image
        processed_image = process_image(image_path)

        # Perform prediction using the pre-trained model
        prediction = model.predict(processed_image)

        # Get the predicted class index (highest probability)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map predicted index to corresponding label (class)
        class_labels = {0: 'cat', 1: 'dog'}  # Ensure this matches your training labels
        classification = class_labels.get(predicted_class, "Unknown")

        # Clean up the saved image file (optional)
        os.remove(image_path)

        return jsonify({'prediction': classification})  # Return the classification as JSON
    except Exception as e:
        return jsonify({'error': str(e)})  # Return error message in case of failure

if __name__ == '__main__':
    app.run(port=3000, debug=True)
