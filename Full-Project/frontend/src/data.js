// Content taken from the Review-II deck and the training outputs (training/final_model.py).

export const LEAD_NAMES = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"];

export const THEMES = {
  "Coral Pulse": ["#ff4d6d", "#ffb020", "#3b82f6"],
  "Ocean Wave": ["#12b5a6", "#3b82f6", "#7c5cff"],
  "Violet Night": ["#7c5cff", "#ec4899", "#12b5a6"],
};

export const CLASS_INFO = {
  NORMAL: { title: "Normal ECG", description: "The model classified this ECG as Normal." },
  AHB: { title: "Abnormal Heartbeat", description: "The model classified this ECG as having an abnormal heartbeat." },
  MI: { title: "Myocardial Infarction", description: "The model classified this ECG as belonging to the MI class." },
  PMI: { title: "History of MI", description: "The model classified this ECG as belonging to the history-of-MI class." },
};

export const DATASET = [
  { cls: "NORMAL", images: 284, meaning: "Normal ECG" },
  { cls: "MI", images: 239, meaning: "Myocardial infarction" },
  { cls: "AHB", images: 233, meaning: "Abnormal heartbeat" },
  { cls: "PMI", images: 172, meaning: "History of MI" },
];

export const OBJECTIVES = [
  ["ECG image processing", "Process ECG images and prepare them for analysis."],
  ["12-lead extraction", "Divide the ECG image into individual leads."],
  ["Waveform features", "Extract meaningful waveform information from each lead."],
  ["Dimensionality reduction", "Reduce redundant features using PCA."],
  ["ML classification", "Train and evaluate multiple ML classifiers."],
  ["Application", "Integrate the trained model into a web application."],
];

export const STAGES = [
  ["ECG image", "Input", "The user uploads a full-page 12-lead ECG image (JPG, JPEG or PNG)."],
  ["Preprocessing", "Gray · Resize · Gauss · Otsu", "Convert to grayscale (intensity), resize to a standard size, smooth with a Gaussian filter to reduce noise, then apply Otsu thresholding to separate the waveform from the background. The result is a binary waveform."],
  ["12-lead division", "Per-lead crop", "The page is divided into the 12 individual leads (I, II, III, aVR, aVL, aVF, V1–V6)."],
  ["Waveform extraction", "Contour-based", "For each lead: grayscale, Gaussian filtering, thresholding, contour detection, then the largest relevant ECG contour is identified."],
  ["Feature extraction", "255 samples × 12 leads", "The contour is resized to a fixed representation of 255 normalised samples per lead. Twelve leads give 12 × 255 = 3060 features per ECG."],
  ["Min-Max scaling", "Normalise features", "Features are scaled to a common range before dimensionality reduction."],
  ["PCA", "3060 → 190 components", "Removes redundant information, lowers computational complexity, and keeps the important variance."],
  ["Classification", "SVM", "Logistic Regression, K-Nearest Neighbors, Random Forest and SVM were compared with 5-fold cross-validation on the training split. The final SVM predicts the class."],
];

export const CLASSIFIERS = [
  ["Logistic Regression", "Linear baseline"],
  ["K-Nearest Neighbors", "Distance-based"],
  ["Random Forest", "Tree ensemble"],
  ["SVM", "Margin-based · final model"],
];

export const CLASSES = ["NORMAL", "AHB", "MI", "PMI"];

export const REPORT = [
  { cls: "NORMAL", Precision: 0.9, Recall: 0.96, "F1-score": 0.93, Support: 57 },
  { cls: "AHB", Precision: 0.91, Recall: 0.85, "F1-score": 0.88, Support: 47 },
  { cls: "MI", Precision: 1.0, Recall: 1.0, "F1-score": 1.0, Support: 48 },
  { cls: "PMI", Precision: 0.91, Recall: 0.88, "F1-score": 0.9, Support: 34 },
];

// rows = true class, columns = predicted class (order: NORMAL, AHB, MI, PMI)
export const CONFUSION = [
  [55, 2, 0, 0],
  [4, 40, 0, 3],
  [0, 0, 48, 0],
  [2, 2, 0, 30],
];

export const TEAM = ["P. Nagarjuna", "P. Nandini", "R. Susan", "P. Harshitha Sowmya"];

export const REFERENCES = [
  "Mhamdi, L., Dammak, O., Cottin, F., & Ben Dhaou, I. (2022). Artificial Intelligence for Cardiac Diseases Diagnosis and Prediction Using ECG Images on Embedded Systems. Biomedicines, 10, 2013.",
  "Uchiyama, R., Okada, Y., Kakizaki, R., & Tomioka, S. (2022). End-to-End Convolutional Neural Network Model to Detect and Localize Myocardial Infarction Using 12-Lead ECG Images without Preprocessing. Bioengineering, 9, 430.",
  "Ashtaiwi, A., Khalifa, T., & Alirr, O. (2024). Enhancing Heart Disease Diagnosis Through ECG Image Vectorization-Based Classification. Heliyon, 10, e37574.",
  "Kailasanathan, N. K., Ezhilarasan, G., Selvarajan, S., Dhanaraj, R. K., Pamucar, D., & Shankar, N. (2025). Heart disease prediction with a feature-sensitized interpretable framework for the Internet of Medical Things sensors. Frontiers in Digital Health, 7. doi: 10.3389/fdgth.2025.1612915",
  "Chang, V., Bhavani, V. R., Xu, A. Q., & Hossain, M. A. (2022). An artificial intelligence model for heart disease detection using machine learning algorithms. Healthcare Analytics, 2, 100016. doi: 10.1016/j.health.2022.100016",
  "Dataset: https://www.kaggle.com/datasets/kanishkarathore1604/ecg-image",
];

export const FUTURE_SCOPE = [
  "Larger, more diverse ECG datasets", "Patient-level external validation", "Improved waveform extraction",
  "Explainable AI / Grad-CAM", "Comparison with CNNs and Transformers", "Real-time ECG acquisition",
  "Mobile / wearable deployment", "Clinical validation with professionals", "Integration of patient information",
];
