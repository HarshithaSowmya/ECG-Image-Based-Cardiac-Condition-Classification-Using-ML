import joblib
import numpy as np

from preprocessing.ecg_signal_extractor import process_ecg


# -----------------------------
# Load saved models
# -----------------------------

pca = joblib.load("models/pca.pkl")
model = joblib.load("models/ecg_model.pkl")


# -----------------------------
# Label mapping
# -----------------------------

labels = {
    0: "NORMAL",
    1: "AHB",
    2: "MI",
    3: "PMI"
}


# -----------------------------
# Test ECG
# -----------------------------

image_path = "dataset/MI/MI(1).jpg"

features = process_ecg(image_path)

if features is None:
    raise RuntimeError("ECG preprocessing failed.")


# Convert to 2D
features = features.reshape(1, -1)

print("Original features:", features.shape)


# -----------------------------
# PCA
# -----------------------------

features_pca = pca.transform(features)

print("PCA features:", features_pca.shape)


# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(features_pca)[0]

print("\n==============================")
print("ECG PREDICTION")
print("==============================")

print("Image:", image_path)
print("Predicted class:", labels[prediction])