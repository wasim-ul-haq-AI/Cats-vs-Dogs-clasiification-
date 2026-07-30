import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "cats_dogs_mobilenetv2_final.keras")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["cat", "dog"]

model = keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_uploaded_image(filepath):
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None, image_path=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded", image_path=None)

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction="No file selected", image_path=None)

    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img_array = preprocess_uploaded_image(filepath)
    prob = float(model.predict(img_array, verbose=0)[0][0])
    label = CLASS_NAMES[int(prob > 0.5)]
    confidence = prob if prob > 0.5 else 1 - prob

    result_text = f"{label.upper()} ({confidence:.2%} confidence)"
    image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template("index.html", prediction=result_text, image_path=image_url)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)