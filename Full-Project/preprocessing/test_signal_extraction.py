from skimage.io import imread
from skimage import color
from skimage.transform import resize
from skimage.filters import threshold_otsu, gaussian
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Load image
# -----------------------------
image_path = "dataset/MI/MI(1).jpg"

image = imread(image_path)

# RGB → grayscale
gray = color.rgb2gray(image)

# Same size as reference
gray = resize(gray, (1572, 2213))


# -----------------------------
# Extract Lead 1
# -----------------------------
lead = gray[300:600, 150:643]

print("Lead shape:", lead.shape)


# -----------------------------
# Gaussian filtering
# -----------------------------
blurred = gaussian(lead, sigma=0.7)


# -----------------------------
# Otsu threshold
# -----------------------------
threshold = threshold_otsu(blurred)

binary = blurred < threshold

print("Threshold:", threshold)
print("Binary shape:", binary.shape)


# -----------------------------
# Resize binary image
# -----------------------------
binary = resize(binary, (300, 450))

print("Resized binary:", binary.shape)


# -----------------------------
# Find contours
# -----------------------------
contours = measure.find_contours(binary, 0.8)

print("Number of contours:", len(contours))

if len(contours) == 0:
    raise RuntimeError("No ECG contour detected.")

# Select largest contour
contour = max(contours, key=lambda x: x.shape[0])

print("Largest contour shape:", contour.shape)


# -----------------------------
# Resize contour
# -----------------------------
signal = resize(contour, (255, 2))

print("Signal shape:", signal.shape)


# -----------------------------
# Min-Max scaling
# -----------------------------
scaler = MinMaxScaler()

scaled = scaler.fit_transform(signal)

print("Scaled signal shape:", scaled.shape)
print("Minimum:", scaled.min())
print("Maximum:", scaled.max())


# -----------------------------
# Take X coordinate
# -----------------------------
one_d_signal = scaled[:, 0]

print("1-D signal shape:", one_d_signal.shape)


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12, 4))

plt.plot(one_d_signal)

plt.title("Extracted 1-D ECG Signal - Lead 1")
plt.xlabel("Sample")
plt.ylabel("Normalized amplitude")

plt.tight_layout()
plt.show()