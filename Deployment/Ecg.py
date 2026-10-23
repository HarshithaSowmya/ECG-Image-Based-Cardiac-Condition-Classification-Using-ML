"""ECG image preprocessing and inference for the bundled PCA + classifier artifacts.

The trained PCA expects 3,060 features: 255 normalized samples from each of 12 ECG leads.
This pipeline builds those features in memory; it does not read or append temporary CSVs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import color, measure
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import resize


LABELS = {
    0: "Abnormal heartbeat",
    1: "Myocardial infarction",
    2: "Normal",
    3: "History of myocardial infarction",
}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024


class ECGInputError(ValueError):
    """Raised when an uploaded file cannot be processed as a supported ECG image."""


@dataclass
class ECGResult:
    label: str
    class_id: Any
    image: np.ndarray
    grayscale: np.ndarray
    leads: list[np.ndarray]
    lead_signals: np.ndarray
    reduced_features: np.ndarray
    probabilities: dict[str, float] | None = None


class ECG:
    """Image preprocessing and prediction using the project's original trained artifacts."""

    def __init__(self, model_path: str | Path | None = None, pca_path: str | Path | None = None):
        base = Path(__file__).resolve().parent
        self.model_path = Path(model_path) if model_path else base / "Heart_Disease_Prediction_using_ECG.pkl"
        self.pca_path = Path(pca_path) if pca_path else base / "PCA_ECG.pkl"
        self._model = None
        self._pca = None

    @staticmethod
    def getImage(uploaded_file) -> np.ndarray:
        """Read and validate an uploaded image without assuming RGB/RGBA input."""
        name = getattr(uploaded_file, "name", "upload.png")
        suffix = Path(name).suffix.lower()
        if suffix and suffix not in SUPPORTED_EXTENSIONS:
            raise ECGInputError(f"Unsupported image format '{suffix}'. Use PNG, JPG, JPEG, BMP, TIFF, or WEBP.")
        raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        if not raw:
            raise ECGInputError("The uploaded file is empty.")
        if len(raw) > MAX_IMAGE_BYTES:
            raise ECGInputError("The image is larger than 15 MB. Please upload a smaller image.")
        try:
            from io import BytesIO
            with Image.open(BytesIO(raw)) as im:
                im.verify()
            with Image.open(BytesIO(raw)) as im:
                if im.width < 500 or im.height < 300:
                    raise ECGInputError(f"Image is too small ({im.width} × {im.height}). Upload a readable ECG page at least 500 × 300 pixels.")
                if im.width > 10000 or im.height > 10000:
                    raise ECGInputError("Image dimensions are unusually large. Please resize the ECG image and try again.")
                aspect = im.width / im.height
                expected_aspect = 2213 / 1572
                if not (expected_aspect * 0.82 <= aspect <= expected_aspect * 1.22):
                    raise ECGInputError("The image proportions do not resemble the supported full-page ECG layout. Upload an upright, uncropped ECG page rather than a single lead or screenshot crop.")
                # Keep grayscale and RGB files valid; normalize transparency on a white background.
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    rgba = im.convert("RGBA")
                    background = Image.new("RGB", rgba.size, "white")
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    im = background
                else:
                    im = im.convert("RGB")
                return np.asarray(im)
        except ECGInputError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ECGInputError("This file is not a readable image. Try exporting the ECG as PNG or JPEG.") from exc

    @staticmethod
    def GrayImgae(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image.astype(np.float32)
            if gray.max(initial=0) > 1:
                gray = gray / 255.0
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            gray = color.rgb2gray(image[..., :3])
        else:
            raise ECGInputError("Unsupported image channels. Expected a grayscale, RGB, or RGBA image.")
        # The bundled lead crop coordinates were designed for this canonical canvas.
        return resize(gray, (1572, 2213), preserve_range=True, anti_aliasing=True).astype(np.float32)

    @staticmethod
    def DividingLeads(image: np.ndarray) -> list[np.ndarray]:
        if image.ndim not in (2, 3) or image.shape[0] < 1480 or image.shape[1] < 2125:
            raise ECGInputError("The ECG layout does not match the supported 12-lead page template. Expected a full-page ECG with the leads arranged in the project's 4-column layout.")
        crops = [
            image[300:600,150:643], image[300:600,646:1135], image[300:600,1140:1625], image[300:600,1630:2125],
            image[600:900,150:643], image[600:900,646:1135], image[600:900,1140:1625], image[600:900,1630:2125],
            image[900:1200,150:643], image[900:1200,646:1135], image[900:1200,1140:1625], image[900:1200,1630:2125],
            image[1250:1480,150:2125],
        ]
        if any(crop.size == 0 or min(crop.shape[:2]) < 20 for crop in crops):
            raise ECGInputError("One or more ECG lead regions are empty. Check the image orientation and page layout.")
        return crops

    @staticmethod
    def _lead_to_signal(lead: np.ndarray) -> np.ndarray:
        """Convert a cropped lead to a stable 255-value trace using the original contour approach."""
        if lead.ndim == 2:
            gray = lead.astype(np.float32)
            if gray.max(initial=0) > 1:
                gray /= 255.0
        else:
            gray = color.rgb2gray(lead[..., :3])
        smooth = gaussian(gray, sigma=0.7, preserve_range=True)
        try:
            binary = smooth < threshold_otsu(smooth)
        except ValueError as exc:
            raise ECGInputError("A lead region has no usable contrast. Upload a clear, high-contrast ECG image.") from exc
        binary = resize(binary.astype(np.float32), (300, 450), preserve_range=True, anti_aliasing=True) > 0.5
        contours = measure.find_contours(binary.astype(np.float32), 0.8)
        if not contours:
            raise ECGInputError("Could not extract an ECG waveform from one of the lead regions. Check image quality and layout.")
        # The source implementation selected the largest contour, then sampled 255 points.
        contour = max(contours, key=len)
        sampled = resize(contour, (255, 2), preserve_range=True, anti_aliasing=True)
        x = sampled[:, 0].astype(np.float32)
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-8:
            raise ECGInputError("A waveform lead has insufficient variation for feature extraction.")
        return ((x - lo) / (hi - lo)).astype(np.float32)

    def PreprocessingLeads(self, Leads):
        """Return display-friendly preprocessed lead images; all computation is in memory."""
        output = []
        for lead in Leads:
            gray = color.rgb2gray(lead[..., :3]) if lead.ndim == 3 else lead.astype(np.float32)
            if gray.max(initial=0) > 1:
                gray = gray / 255.0
            blurred = gaussian(gray, sigma=1)
            output.append(resize((blurred < threshold_otsu(blurred)).astype(np.float32), (300, 450), preserve_range=True) > 0.5)
        return output

    def SignalExtraction_Scaling(self, Leads):
        """Extract a (12, 255) matrix of normalized lead traces."""
        return np.vstack([self._lead_to_signal(lead) for lead in Leads[:12]]).astype(np.float32)

    def CombineConvert1Dsignal(self, signals: np.ndarray) -> np.ndarray:
        """Flatten twelve leads in deterministic lead order to the model's 3,060 features."""
        arr = np.asarray(signals, dtype=np.float32)
        if arr.shape != (12, 255):
            raise ECGInputError(f"Internal feature shape mismatch: expected 12 × 255, received {arr.shape}.")
        return arr.reshape(1, 3060)

    def _load_artifacts(self):
        if self._pca is None:
            if not self.pca_path.is_file():
                raise RuntimeError(f"PCA artifact not found: {self.pca_path.name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._pca = joblib.load(self.pca_path)
        if self._model is None:
            if not self.model_path.is_file():
                raise RuntimeError(f"Classifier artifact not found: {self.model_path.name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = joblib.load(self.model_path)
        expected = getattr(self._pca, "n_features_in_", None)
        if expected is not None and expected != 3060:
            raise RuntimeError(f"The bundled PCA expects {expected} features, but this pipeline generates 3060. Model artifacts are incompatible.")
        return self._pca, self._model

    def DimensionalReduciton(self, test_final):
        pca, _ = self._load_artifacts()
        features = np.asarray(test_final, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != getattr(pca, 'n_features_in_', 3060):
            raise ECGInputError(f"Feature schema mismatch. The PCA expects {getattr(pca, 'n_features_in_', 'an unknown number of')} numeric features.")
        return pca.transform(features)

    def ModelLoad_predict(self, final_df):
        _, model = self._load_artifacts()
        features = np.asarray(final_df)
        expected = getattr(model, 'n_features_in_', None)
        if expected is not None and features.shape[1] != expected:
            raise RuntimeError(f"Classifier expects {expected} features but received {features.shape[1]}.")
        prediction = model.predict(features)[0]
        try:
            class_id = int(prediction)
        except (TypeError, ValueError):
            class_id = prediction
        label = LABELS.get(class_id, f"Model class: {class_id}")
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(features)[0]
                classes = getattr(model, 'classes_', range(len(probs)))
                probabilities = {LABELS.get(int(c), str(c)): float(p) for c, p in zip(classes, probs)}
            except (ValueError, AttributeError, TypeError):
                probabilities = None
        return label, class_id, probabilities

    def predict_image(self, uploaded_file) -> ECGResult:
        image = self.getImage(uploaded_file)
        gray = self.GrayImgae(image)
        leads = self.DividingLeads(gray)
        signals = self.SignalExtraction_Scaling(leads)
        combined = self.CombineConvert1Dsignal(signals)
        reduced = self.DimensionalReduciton(combined)
        label, class_id, probabilities = self.ModelLoad_predict(reduced)
        return ECGResult(label, class_id, image, gray, leads, signals, reduced, probabilities)
