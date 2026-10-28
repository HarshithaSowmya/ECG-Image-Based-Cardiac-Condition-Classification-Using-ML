"""
Automated checks that need no real ECG images.  Run from project root:

    python -m pytest validation/test_validator.py -q      (or)      python validation/test_validator.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "validation"))

import cv2, joblib, numpy as np, pandas as pd
from skimage import data
import synthetic as S
from preprocessing.ecg_validator import (ECGValidator, ValidationConfig, check_distribution,
                                         NOT_ECG, NOT_12_LEAD, INVALID_FILE, IMAGE_TOO_SMALL, OK)

pca = joblib.load(os.path.join(ROOT, "models", "pca.pkl"))
cfg = ValidationConfig.load(os.path.join(ROOT, "models", "validation_thresholds.json"))
V = ECGValidator(pca, cfg)
enc = lambda im, ext=".png": cv2.imencode(ext, im)[1].tobytes()
bgr = lambda a: cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def test_file_level():
    assert V.validate_bytes(b"").code == INVALID_FILE
    assert V.validate_bytes(b"%PDF-1.4 not an image").code == INVALID_FILE
    assert V.validate_bytes(enc(S.make_ecg(1), ".jpg")[:4000]).code == INVALID_FILE      # truncated
    assert V.validate_bytes(enc(np.full((80, 100, 3), 255, np.uint8))).code == IMAGE_TOO_SMALL


def test_non_ecg_images_rejected():
    for n in ["astronaut", "coffee", "camera", "chelsea", "page", "text", "brick", "grass", "coins", "rocket", "cell", "moon"]:
        im = getattr(data, n)()
        im = bgr(im) if im.ndim == 3 else im
        r = V.validate_bytes(enc(im)); assert not r.valid, n
    for n in ["neg_white", "neg_noise", "neg_gradient", "neg_text_page", "neg_line_chart", "neg_grid_only"]:
        r = V.validate_bytes(enc(getattr(S, n)())); assert not r.valid, n


def test_non_ecg_rejected_by_feature_gate_alone():
    """Stage 4 must catch non-ECG photos even if stages 1-3 were bypassed."""
    from preprocessing.ecg_signal_extractor import process_ecg_array
    for n in ["astronaut", "coffee", "camera", "chelsea", "page", "text", "gravel", "rocket"]:
        f, _ = process_ecg_array(getattr(data, n)())
        _, code, _w = check_distribution(pca, f, cfg)
        assert code is not None, n


def test_genuine_training_features_pass_feature_gate():
    df = pd.read_csv(os.path.join(ROOT, "training", "ecg_features.csv"))
    X = df[[c for c in df.columns if c.startswith("feature_")]].to_numpy()
    rej = sum(check_distribution(pca, x, cfg)[1] is not None for x in X)
    assert rej / len(X) < 0.02, f"{rej}/{len(X)} genuine ECG feature rows rejected"


def test_ecg_style_image_passes_image_checks():
    """Synthetic 12-lead ECG, several degradations: stages 1-3 must accept."""
    loose = ValidationConfig.load(os.path.join(ROOT, "models", "validation_thresholds.json"))
    loose.max_recon_error = loose.max_mahalanobis = None            # synthetic != real ECG statistics
    v = ECGValidator(pca, loose)
    E = S.make_ecg(3)
    for name, im in {"clean": E, "jpeg": S.jpeg(E, 30), "half": S.rescale(E, .5), "rot2": S.rotate(E, 2),
                     "dim": S.dim(E, .55), "gray": S.gray3(E), "blur": S.blur(E, 9)}.items():
        r = v.validate_bytes(enc(im)); assert r.code == OK, (name, [c for c in r.checks if not c.passed])


def test_any_12_lead_layout_is_recognised():
    """3x4, 6x2 and 12x1 pages (with/without rhythm strip) are recognised as 12-lead ECGs."""
    for kind in ("3x4", "6x2", "12x1"):
        for rhythm in (True, False):
            r = V.validate_bytes(enc(S.make_ecg_layout(kind, rhythm=rhythm)))
            assert r.twelve_lead, (kind, rhythm, r.code)


def test_ecg_without_12_leads_is_rejected():
    for n in (1, 2, 5, 8):
        r = V.validate_bytes(enc(S.make_ecg_rows(n)))
        assert r.code == NOT_12_LEAD and not r.valid and not r.twelve_lead, (n, r.code)


def test_stacked_line_charts_are_not_ecg():
    for n in (1, 3, 4):
        r = V.validate_bytes(enc(S.neg_multipanel_chart(n)))
        assert r.code == NOT_ECG, (n, r.code)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)
