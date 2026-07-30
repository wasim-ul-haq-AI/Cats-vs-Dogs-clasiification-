# 🐱 vs 🐶 Cats vs Dogs Classifier — MobileNetV2 Transfer Learning

An end-to-end image classification project that uses **transfer learning with MobileNetV2** to classify images as cats or dogs, complete with data augmentation, fine-tuning, Grad-CAM explainability, and a deployable **Flask web app**.

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | **98.37%** |
| Test AUC | **0.9988** |
| Test Loss | 0.0414 |

**Classification Report**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Cat | 0.99 | 0.98 | 0.98 |
| Dog | 0.98 | 0.99 | 0.98 |

Evaluated on a held-out, stratified test set of 3,750 images (1,875 per class).

---

## 🧠 Approach

The pipeline follows a standard transfer-learning workflow:

1. **EDA** — class balance check, sample image inspection, image size distribution, corrupt-file detection
2. **Preprocessing** — label encoding, resize to 224×224, MobileNetV2-specific `[-1, 1]` pixel scaling
3. **Data Augmentation** — random flip, rotation, zoom, contrast, and translation applied only to training data
4. **Dataset Split** — stratified 70/15/15 train/validation/test split
5. **Transfer Learning** — MobileNetV2 (ImageNet weights) as a frozen feature extractor
6. **Custom Head** — Global Average Pooling → Dropout → Dense(128, ReLU) → Dropout → Dense(1, sigmoid)
7. **Training** — Adam optimizer, binary cross-entropy loss, EarlyStopping + ReduceLROnPlateau + ModelCheckpoint callbacks
8. **Fine-Tuning** — unfroze the last 30 layers of MobileNetV2 and retrained at a 100x lower learning rate (1e-5) to squeeze out additional accuracy without catastrophic forgetting
9. **Explainability** — Grad-CAM heatmaps to visualize which regions of an image drove each prediction
10. **Deployment** — a lightweight Flask app for uploading an image and getting a live prediction

---

## 🗂️ Project Structure

```
├── cats_dogs_mobilenet_pipeline.py   # Full training pipeline (EDA → training → Grad-CAM)
├── flask_app/
│   ├── app.py                        # Flask server
│   ├── templates/
│   │   └── index.html                # Upload UI
│   ├── static/
│   │   └── uploads/                  # Uploaded images (runtime, gitignored)
│   ├── requirements.txt
│   └── cats_dogs_mobilenetv2_final.keras   # Trained model (not committed — see below)
└── README.md
```

> **Note:** The trained `.keras` model file is not included in this repo due to size. See [Training](#-training) below to regenerate it, or download it from the Releases section if provided.

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/<your-username>/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier/flask_app

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Place your trained model file (`cats_dogs_mobilenetv2_final.keras`) in the `flask_app/` folder, then run:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser, upload a cat or dog image, and get an instant prediction with confidence score.

---

## 🏋️ Training

The full training pipeline is in `cats_dogs_mobilenet_pipeline.py` (best run as notebook cells in Kaggle/Jupyter). It expects a dataset directory of cat/dog images (either as class subfolders or filenames containing "cat"/"dog") and produces the saved `.keras` model used by the Flask app.

Key training choices:
- **Base model frozen first**, head trained alone — prevents destructive gradients from a randomly-initialized head corrupting pretrained ImageNet weights
- **Fine-tuning at 1e-5 LR** on only the last 30 layers — adapts high-level features to this specific task while preserving general visual knowledge learned from ImageNet
- **Stratified splitting** — keeps class balance consistent across train/val/test

---

## 🔍 Explainability (Grad-CAM)

Rather than treating the model as a black box, Grad-CAM is used to generate heatmaps showing which parts of an image most influenced the prediction — useful for verifying the model is actually focusing on the animal (ears, face, fur texture) rather than background artifacts.

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model building, training, MobileNetV2 transfer learning
- **OpenCV, Pillow, NumPy, Pandas** — image I/O and preprocessing
- **Matplotlib, Seaborn** — EDA and result visualization
- **scikit-learn** — train/test splitting, evaluation metrics
- **Flask** — web app for serving predictions

---

## 📌 Future Improvements

- Multi-class extension (beyond binary cat/dog)
- Convert to TensorFlow Lite for mobile inference
- Add batch prediction / REST API endpoint (JSON in, JSON out) alongside the HTML form
- Containerize with Docker for easier deployment

---

## 📄 License

This project is open for educational and personal use. Add a license of your choice (MIT recommended) if publishing publicly.
