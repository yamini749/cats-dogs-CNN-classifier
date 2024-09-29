from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your own custom model
model = load_model('model.h5', compile=False)  # Load your trained model
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    try:
        # Load and preprocess the image
        image = load_img(image_path, target_size=(256, 256, 3))  # Change target_size if your model expects a different size
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Reshape to match model's input shape
        image = image.astype('float32') / 255  # Normalize pixel values

        # Predict using your loaded model
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)  # Get the index of the highest probability

        # Assuming you have a mapping of class indices to human-readable labels
        class_labels = {0: 'cat', 1: 'dog'}  # Define your class labels here
        classification = class_labels.get(predicted_class[0], "Unknown")

        return jsonify({'prediction': classification})  # Return a JSON response
    except Exception as e:
        return jsonify({'error': str(e)})  # Return any errors as a JSON response

if __name__ == '__main__':
    app.run(port=3000, debug=True)
