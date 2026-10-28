"""
Calibrate the data-driven validation thresholds from YOUR training features.

Run from the project root (after create_dataset.py / prepare_data.py /
final_model.py):

    python training/calibrate_validation.py

Writes models/validation_thresholds.json, which the API loads at start-up.

How the OOD (out-of-distribution) thresholds are chosen
-------------------------------------------------------
PCA is re-fitted on 4/5 of the data and the held-out 1/5 is scored, 5 times.
Scoring only held-out samples matters: samples PCA was fitted on always look
"too normal", which would make the thresholds too tight and reject real ECGs.
"""
import json
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from preprocessing.ecg_validator import ood_scores     # noqa: E402

FEATURES_CSV = "training/ecg_features.csv"
OUT = "models/validation_thresholds.json"
PERCENTILE = 99.5      # keep 99.5% of genuine held-out ECGs
MARGIN = 1.10          # +10% head-room for scanner / JPEG variation

df = pd.read_csv(FEATURES_CSV)
cols = [c for c in df.columns if c.startswith("feature_")]
X = df[cols].to_numpy()
y = df["label"].to_numpy()

rec, mah = [], []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    p = PCA(n_components=0.95, svd_solver="full").fit(X[tr])
    for i in te:
        r, m = ood_scores(p, X[i])
        rec.append(r)
        mah.append(m)
rec, mah = np.array(rec), np.array(mah)

print("Held-out genuine ECGs (5-fold):")
print(f"  recon error : median {np.median(rec):.3f}  p99.5 {np.percentile(rec, PERCENTILE):.3f}  max {rec.max():.3f}")
print(f"  mahalanobis : median {np.median(mah):.3f}  p99.5 {np.percentile(mah, PERCENTILE):.3f}  max {mah.max():.3f}")

# Reconstruction error is the discriminating gate (non-ECG photos score far above
# genuine ECGs). Mahalanobis distance barely separates them, so it is only a loose
# backstop against extreme outliers: genuine maximum + 15%.
thr = {
    "max_recon_error": float(np.percentile(rec, PERCENTILE) * MARGIN),
    "max_mahalanobis": float(mah.max() * 1.15),
}

# ---- confidence threshold: accuracy vs coverage on the untouched test split ----
pca = joblib.load("models/pca.pkl")
model = joblib.load("models/ecg_model.pkl")
test = pd.read_csv("training/test_pca.csv")
Xt, yt = test.drop(columns=["label"]).to_numpy(), test["label"].to_numpy()
proba = model.predict_proba(Xt)
pred = model.predict(Xt)                      # same label the API reports
conf = proba[np.arange(len(pred)), np.searchsorted(model.classes_, pred)]  # prob. of THAT label
print("\nConfidence gate (test split):")
print("  min_conf  kept   accuracy-of-kept")
chosen = 0.50
for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
    k = conf >= t
    acc = (pred[k] == yt[k]).mean() if k.any() else float("nan")
    print(f"  {t:7.2f}  {k.mean():5.1%}  {acc:6.1%}")
thr["min_confidence"] = chosen

with open(OUT, "w") as f:
    json.dump(thr, f, indent=2)
print("\nSaved", OUT, thr)
