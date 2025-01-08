import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static/models"  # Update this to the correct folder containing your model

# Load model
model_path = os.path.join(STATIC_FOLDER, "final_resnet_model.h5")
resnet_model = tf.keras.models.load_model(model_path)

# Define image size for preprocessing
IMAGE_SIZE = (224, 224)

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Read and preprocess image from path
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Predict & classify image
def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.reshape(preprocessed_image, (1, *IMAGE_SIZE, 3))

    # Make prediction
    probabilities = model.predict(preprocessed_image)

    # Get the highest probability and the predicted class
    confidence = np.max(probabilities) * 100
    predicted_class = np.argmax(probabilities, axis=1)[0]
    class_labels = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis Lesion', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Squamous Cell Carcinoma', 'Vascular Lesion']

    # Mapping for categories
    benign_labels = ['Benign Keratosis Lesion', 'Dermatofibroma', 'Melanocytic Nevus']
    malignant_labels = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Melanoma', 'Squamous Cell Carcinoma', 'Vascular Lesion']

    # Determine category
    label = class_labels[predicted_class]
    if label in benign_labels:
        category = "jinak (benign)"
    elif label in malignant_labels:
        category = "ganas (malignant)"
    else:
        category = "tidak diketahui"

    return label, confidence, category

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Handle image upload and classification
@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")

    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)

    label, confidence, category = classify(resnet_model, upload_image_path)
    confidence = round(confidence, 2)

    return render_template(
        "classify.html",
        image_file_name=file.filename,
        label=label,
        prob=confidence,
        category=category,
    )

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)