import os
import numpy as np

from skimage.io import imread
from skimage import color, measure
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# ECG lead coordinates from the reference pipeline
# --------------------------------------------------

LEAD_COORDINATES = {
    "Lead_1":  (300, 600, 150, 643),
    "Lead_2":  (300, 600, 646, 1135),
    "Lead_3":  (300, 600, 1140, 1625),
    "Lead_4":  (300, 600, 1630, 2125),

    "Lead_5":  (600, 900, 150, 643),
    "Lead_6":  (600, 900, 646, 1135),
    "Lead_7":  (600, 900, 1140, 1625),
    "Lead_8":  (600, 900, 1630, 2125),

    "Lead_9":  (900, 1200, 150, 643),
    "Lead_10": (900, 1200, 646, 1135),
    "Lead_11": (900, 1200, 1140, 1625),
    "Lead_12": (900, 1200, 1630, 2125),
}


def extract_lead_signal(lead):
    """
    Convert one ECG lead image into a
    255-point normalized 1-D signal.
    """

    # Gaussian smoothing
    blurred = gaussian(lead, sigma=0.7)

    # Otsu threshold
    threshold = threshold_otsu(blurred)

    # ECG waveform is darker than background
    binary = blurred < threshold

    # Standard size used by reference pipeline
    binary = resize(binary, (300, 450))

    # Find contours
    contours = measure.find_contours(binary, 0.8)

    if not contours:
        return None

    # Select largest contour
    contour = max(contours, key=lambda x: x.shape[0])

    # Resize contour to 255 x 2
    signal = resize(contour, (255, 2))

    # Min-Max scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(signal)

    # Use X coordinate as 1-D signal
    one_d_signal = scaled[:, 0]

    return one_d_signal


REFERENCE_SIZE = (1572, 2213)   # (height, width) used by the reference pipeline


def to_reference_gray(image):
    """
    Decoded image array (gray / RGB / RGBA) -> grayscale float image
    resized to the reference size (1572 x 2213).
    """

    if image.ndim == 3:
        # drop alpha channel if present (RGBA PNGs)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        gray = color.rgb2gray(image)
    else:
        gray = image

    return resize(gray, REFERENCE_SIZE)


def process_ecg_array(image):
    """
    Same as process_ecg(), but takes an already decoded image array.

    Returns:
        features : np.ndarray of shape (3060,) or None if any lead failed
        failed_leads : list of lead names where no waveform contour was found
    """

    gray = to_reference_gray(image)

    all_features = []
    failed_leads = []

    for lead_name, (r1, r2, c1, c2) in LEAD_COORDINATES.items():

        signal = extract_lead_signal(gray[r1:r2, c1:c2])

        if signal is None:
            failed_leads.append(lead_name)
            continue

        all_features.extend(signal)

    if failed_leads:
        return None, failed_leads

    return np.array(all_features), failed_leads


def process_ecg(image_path):
    """
    Process one ECG image.

    Returns:
        3060 features = 12 leads x 255 points
    """

    features, _ = process_ecg_array(imread(image_path))

    return features


# --------------------------------------------------
# Test with one image
# --------------------------------------------------

if __name__ == "__main__":

    image_path = "dataset/MI/MI(1).jpg"

    features = process_ecg(image_path)

    if features is None:
        print("Signal extraction failed.")
    else:
        print("Image:", image_path)
        print("Feature shape:", features.shape)
        print("Expected shape: (3060,)")
        print("Minimum:", features.min())
        print("Maximum:", features.max())