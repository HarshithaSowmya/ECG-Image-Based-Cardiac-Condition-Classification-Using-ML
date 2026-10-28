"""
ECG input validation.

Decides, BEFORE the classifier is allowed to answer, whether an uploaded file is
an ECG image this model can legitimately handle. Four gates, cheapest first:

  Stage 1  FILE       real JPEG/PNG, decodable, sane size and dimensions
  Stage 2  ECG-NESS   ECG-paper grid + waveform traces in the 12-lead layout
  Stage 3  SIGNAL     the 12 leads were actually extracted and are not flat
  Stage 4  FEATURES   extracted features look like the TRAINING data
                      (PCA reconstruction error + Mahalanobis distance)

A later gate (in the API) also rejects low-confidence predictions.

Every check is reported (name, passed, value, threshold) so you can see WHY an
image was rejected and tune thresholds.
"""
from __future__ import annotations

import io
import warnings
import json
import os
from dataclasses import dataclass, field, asdict, replace
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from preprocessing import ecg_layout
from preprocessing import ecg_region
from preprocessing.ecg_signal_extractor import (
    LEAD_COORDINATES,
    process_ecg_array,
    to_reference_gray,
)

Image.MAX_IMAGE_PIXELS = 60_000_000      # decompression-bomb guard


# ----------------------------------------------------------------------------
# Rejection codes (the frontend can switch on these)
# ----------------------------------------------------------------------------
INVALID_FILE = "INVALID_FILE"            # not a readable JPEG/PNG
IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
FILE_TOO_LARGE = "FILE_TOO_LARGE"
NOT_ECG = "NOT_ECG"                      # no ECG grid / traces at all
UNSUPPORTED_LAYOUT = "UNSUPPORTED_LAYOUT"  # (legacy, no longer produced)
NOT_12_LEAD = "NOT_12_LEAD"              # an ECG, but not a complete 12-lead page
POOR_SIGNAL = "POOR_SIGNAL"              # leads could not be extracted cleanly
OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"  # features unlike training data
OK = "OK"

MESSAGES = {
    INVALID_FILE: "This file is not a valid JPG/PNG image.",
    IMAGE_TOO_SMALL: "This image is too small to be a readable 12-lead ECG. If it is an ECG, "
                     "please upload a higher-resolution version.",
    FILE_TOO_LARGE: "The file is too large. Please upload an image under 4 MB.",
    NOT_ECG: "This is not an ECG image. Please upload a 12-lead ECG (waveforms on ECG grid paper).",
    NOT_12_LEAD: "This looks like an ECG, but it does not contain a complete 12-lead layout. "
                 "Please upload a full 12-lead ECG (not a single-lead strip or a partial image) and make sure it is upright, not sideways.",
    POOR_SIGNAL: "The ECG waveforms could not be read reliably from this image. "
                 "Please upload a sharper, well-lit, straight-on ECG image.",
    OUT_OF_DISTRIBUTION: "A 12-lead ECG was detected, but this image looks too different from the images "
                         "the model was trained on, so no result is shown (it would not be reliable).",
    OK: "ECG image accepted.",
}


@dataclass
class Check:
    name: str
    passed: bool
    value: float | str | None = None
    threshold: float | str | None = None
    detail: str = ""


@dataclass
class ValidationResult:
    valid: bool
    code: str
    message: str
    stage: str
    checks: list = field(default_factory=list)
    features: Optional[np.ndarray] = None      # 3060 features when stage 3 passed
    warnings: list = field(default_factory=list)
    twelve_lead: bool = False          # a complete 12-lead ECG was recognised
    preprocessing: Optional[dict] = None

    def to_dict(self):
        d = {
            "valid": self.valid,
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "twelve_lead": self.twelve_lead,
            "checks": [asdict(c) for c in self.checks],
            "warnings": self.warnings,
            "preprocessing": self.preprocessing,
        }
        return d


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class ValidationConfig:
    # stage 1
    max_file_bytes: int = 4 * 1024 * 1024
    min_side_px: int = 300
    # stage 2
    min_aspect: float = 1.15            # landscape; reference is 2213/1572 = 1.41
    max_aspect: float = 2.0
    work_width: int = 1000              # working width for grid analysis
    min_grid_score: float = 0.20        # autocorrelation peak, both axes
    min_trace_cells: int = 11           # of the 12 lead cells must contain a trace
    cell_min_coverage: float = 0.60     # share of cell columns containing trace pixels
    cell_dark_range: tuple = (0.005, 0.08)  # share of dark pixels in a cell
    # stage 3
    min_lead_std: float = 0.03          # flatter than this = dead lead
    max_flat_leads: int = 2
    # stage 4 (overwritten by models/validation_thresholds.json)
    max_recon_error: Optional[float] = None
    max_mahalanobis: Optional[float] = None
    # If stage 2 saw a waveform in ALL 12 lead boxes, a recon error up to
    # max_recon_error * soft_band is accepted with a warning (small / low-res
    # copies of genuine ECGs have noisier contours). Otherwise strict.
    soft_band: float = 1.45
    # prediction confidence gate (used by the API)
    min_confidence: float = 0.50

    @classmethod
    def load(cls, path="models/validation_thresholds.json"):
        cfg = cls()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg


# ----------------------------------------------------------------------------
# Stage 1 - file
# ----------------------------------------------------------------------------
def decode_image(data: bytes, cfg: ValidationConfig):
    """bytes -> (RGB uint8 array | None, checks, failure_code | None)"""
    checks = []

    ok = len(data) <= cfg.max_file_bytes and len(data) > 0
    checks.append(Check("file_size", ok, round(len(data) / 1048576, 2), f"<= {cfg.max_file_bytes/1048576:.0f} MB"))
    if not ok:
        return None, checks, FILE_TOO_LARGE if len(data) else INVALID_FILE

    # We trust the *content*, not the filename or the browser-supplied MIME type.
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()                       # detects truncated / corrupt files
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)      # phone photos: honour rotation tag
        im = im.convert("RGB")                # handles RGBA, CMYK, palette, gray
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        checks.append(Check("is_valid_image", False, "unreadable", "JPEG/PNG"))
        return None, checks, INVALID_FILE

    # Some downloaded .jpg files actually contain WebP. Decode those safely too;
    # processing uses the decoded RGB array and never trusts an extension.
    ok = fmt in ("JPEG", "PNG", "WEBP")
    checks.append(Check("image_format", ok, fmt, "JPEG, PNG, or decoded WebP"))
    if not ok:
        return None, checks, INVALID_FILE

    arr = np.asarray(im)
    h, w = arr.shape[:2]
    ok = min(h, w) >= cfg.min_side_px
    checks.append(Check("min_resolution", ok, f"{w}x{h}", f"short side >= {cfg.min_side_px}px"))
    if not ok:
        return None, checks, IMAGE_TOO_SMALL

    return arr, checks, None


# ----------------------------------------------------------------------------
# Stage 2 - does it look like a 12-lead ECG?
# ----------------------------------------------------------------------------
def _axis_periodicity(profile: np.ndarray, lo=3, hi=120, min_amp=0.003) -> float:
    """Strength (0..1) of the strongest periodic component of a 1-D profile."""
    p = profile.astype(np.float64)
    k = 41
    p = p - np.convolve(np.pad(p, k // 2, mode="edge"), np.ones(k) / k, mode="valid")  # high-pass
    p = np.clip(p, -4 * np.median(np.abs(p)) - 1e-6, 4 * np.median(np.abs(p)) + 1e-6)   # tame outliers
    p = p - p.mean()
    if p.std() < min_amp * 0.25:             # no real line contrast (flat / smooth gradient)
        return 0.0
    n = len(p)
    f = np.fft.rfft(p, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[:n]
    r = r / r[0]
    hi = min(hi, n // 3)
    seg = r[lo:hi]
    peaks = [seg[i] for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    return float(max(peaks)) if peaks else 0.0


def _masked_profile(g: np.ndarray, axis: int) -> np.ndarray:
    """Mean over the NON-dark pixels along an axis, so waveform / text pixels
    (and flat baselines that span a whole row) do not swamp the faint grid."""
    keep = g > 0.6 * np.percentile(g, 90)       # relative: works for dim photos too
    cnt = keep.sum(axis=axis)
    tot = (g * keep).sum(axis=axis)
    prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    idx = np.arange(len(prof))
    good = ~np.isnan(prof)
    if good.sum() < len(prof) * 0.5:
        return np.zeros(len(prof))
    return np.interp(idx, idx[good], prof[good])


def grid_score(rgb: np.ndarray, cfg: ValidationConfig) -> float:
    """
    ECG paper is a regular grid: faint lines repeating along BOTH axes.
    A few small rotations are tried so slightly tilted photos still pass.
    """
    h, w = rgb.shape[:2]
    s = cfg.work_width / w
    small = cv2.resize(rgb, (cfg.work_width, max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    best = 0.0
    hh, ww = g.shape
    for ang in (0, -1, 1, -2, 2, -3, 3):
        if ang:
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
            gr = cv2.warpAffine(g, M, (ww, hh), borderMode=cv2.BORDER_REPLICATE)
        else:
            gr = g
        col = _axis_periodicity(_masked_profile(gr, 0))
        row = _axis_periodicity(_masked_profile(gr, 1))
        best = max(best, min(col, row))
        if best > 0.6:
            break
    return best


def trace_cells(rgb: np.ndarray, cfg: ValidationConfig):
    """
    Count how many of the 12 lead cells (the exact crops the extractor uses)
    contain a waveform-like trace: a thin dark line crossing most of the cell.
    """
    gray = to_reference_gray(rgb)                      # 1572 x 2213, float 0..1
    gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    sub = gray[::4, ::4]
    lo, hi = np.percentile(sub, 0.1), np.percentile(sub, 90)   # 0.1%: thin 1-px traces are few pixels
    if hi - lo < 0.15 or lo > 0.45:                     # no contrast / no truly dark ink
        return 0, []
    norm = np.clip((gray - lo) / (hi - lo), 0, 1)
    dark = norm < 0.40
    good, stats = 0, []
    for name, (r1, r2, c1, c2) in LEAD_COORDINATES.items():
        cell = dark[r1:r2, c1:c2]
        frac = float(cell.mean())
        cover = float(cell.any(axis=0).mean())
        ok = cover >= cfg.cell_min_coverage and cfg.cell_dark_range[0] <= frac <= cfg.cell_dark_range[1]
        good += ok
        stats.append({"lead": name, "coverage": round(cover, 2), "dark": round(frac, 4), "ok": bool(ok)})
    return good, stats


def check_ecg_like(rgb: np.ndarray, cfg: ValidationConfig):
    checks = []
    h, w = rgb.shape[:2]
    aspect = w / h
    ok_aspect = cfg.min_aspect <= aspect <= cfg.max_aspect
    checks.append(Check("landscape_aspect_ratio", ok_aspect, round(aspect, 2),
                        f"{cfg.min_aspect}-{cfg.max_aspect}"))

    gs = grid_score(rgb, cfg)
    ok_grid = gs >= cfg.min_grid_score
    checks.append(Check("ecg_grid_detected", ok_grid, round(gs, 3), f">= {cfg.min_grid_score}"))

    cells, _ = trace_cells(rgb, cfg)
    ok_cells = cells >= cfg.min_trace_cells
    checks.append(Check("lead_cells_with_waveform", ok_cells, cells, f">= {cfg.min_trace_cells} of 12"))

    if ok_aspect and ok_grid and ok_cells:
        return checks, None
    if not ok_grid and not ok_cells:
        return checks, NOT_ECG
    if not ok_aspect and not ok_grid:
        return checks, NOT_ECG
    # one signal present, the other missing: ECG-like only if BOTH grid and
    # a substantial number of waveform cells are seen, otherwise it is not an ECG
    if ok_grid and cells >= 6:
        return checks, UNSUPPORTED_LAYOUT
    return checks, NOT_ECG


# ----------------------------------------------------------------------------
# Stage 2b - re-alignment for ECGs from other sources (different page margins)
# ----------------------------------------------------------------------------
# Where the ECG grid frame sits on a Mendeley page, as fractions of the image.
# (measured on real dataset images: identical on all of them)
REF_FRAME = (0.026, 0.176, 0.990, 0.969)       # x0, y0, x1, y1


def detect_grid_frame(rgb: np.ndarray):
    """Bounding box (x0, y0, x1, y1 in pixels) of the pink/red ECG grid, or None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, sat, val = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    pink = ((h < 12) | (h > 160)) & (sat > 18) & (val > 120)
    H, W = pink.shape

    def span(profile, n):
        k = max(3, n // 100) | 1
        p = np.convolve(profile, np.ones(k) / k, "same")
        idx = np.where(p > 0.35 * np.percentile(p, 95))[0]
        return (int(idx[0]), int(idx[-1])) if len(idx) else None

    sx, sy = span(pink.mean(0), W), span(pink.mean(1), H)
    if sx is None or sy is None or pink.mean() < 0.005:
        return None
    x0, x1 = sx
    y0, y1 = sy
    w, hh = x1 - x0, y1 - y0
    if w < 0.3 * W or hh < 0.3 * H or not (1.2 <= w / max(hh, 1) <= 2.4):
        return None
    return x0, y0, x1, y1


def is_mendeley_template(frame, shape, tol=0.03):
    """True when the ECG grid frame sits where it does on every Mendeley page."""
    h, w = shape[:2]
    x0, y0, x1, y1 = frame
    got = (x0 / w, y0 / h, x1 / w, y1 / h)
    return all(abs(a - b) <= tol for a, b in zip(got, REF_FRAME))


def register_to_reference(rgb: np.ndarray):
    """
    Place the detected ECG grid area onto a white 2213x1572 page at the position
    it has on a Mendeley page, so the fixed lead crops line up. None if no frame.
    """
    box = detect_grid_frame(rgb)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1 + 1, x0:x1 + 1]
    # Suppress a strongly coloured grid: use the channel in which the grid is
    # brightest (red channel for pink/red paper) - black traces stay dark.
    ch = int(np.argmax(crop.reshape(-1, 3).mean(0)))
    crop = np.repeat(crop[..., ch:ch + 1], 3, axis=2)
    RH, RW = 1572, 2213
    tx0, ty0, tx1, ty1 = (int(REF_FRAME[0] * RW), int(REF_FRAME[1] * RH),
                          int(REF_FRAME[2] * RW), int(REF_FRAME[3] * RH))
    canvas = np.full((RH, RW, 3), 255, np.uint8)
    canvas[ty0:ty1, tx0:tx1] = cv2.resize(crop, (tx1 - tx0, ty1 - ty0), interpolation=cv2.INTER_AREA)
    return canvas


# ----------------------------------------------------------------------------
# Stage 3 - signal extraction quality
# ----------------------------------------------------------------------------
def check_signal(features, failed_leads, cfg: ValidationConfig):
    checks = []
    ok = features is not None
    checks.append(Check("all_12_leads_extracted", ok, 12 - len(failed_leads), "12"))
    if not ok:
        return checks, POOR_SIGNAL
    if features.shape != (3060,) or not np.isfinite(features).all():
        checks.append(Check("features_finite_3060", False, str(features.shape), "(3060,)"))
        return checks, POOR_SIGNAL
    stds = features.reshape(12, 255).std(axis=1)
    flat = int((stds < cfg.min_lead_std).sum())
    ok = flat <= cfg.max_flat_leads
    checks.append(Check("flat_leads", ok, flat, f"<= {cfg.max_flat_leads}"))
    return checks, None if ok else POOR_SIGNAL


# ----------------------------------------------------------------------------
# Stage 4 - are the features like the training data?
# ----------------------------------------------------------------------------
def transform_features(pca, values):
    """Preserve the numerical PCA input without a pandas runtime dependency.

    The saved PCA also stores training column names. Column order remains the
    same; suppress only sklearn's warning about unlabeled ndarray input.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)
        return pca.transform(np.asarray(values, dtype=np.float64))


def ood_scores(pca, features: np.ndarray):
    """
    recon_error : distance between the features and their PCA reconstruction
                  (large = pattern the PCA never saw)
    mahalanobis : distance of the PCA coordinates from the training mean, in
                  units of the training spread (large = extreme sample)
    """
    x = features.reshape(1, -1)
    z = transform_features(pca, x)
    recon = pca.inverse_transform(z)
    recon_error = float(np.sqrt(((np.asarray(x) - recon) ** 2).sum()))
    mahal = float(np.sqrt((z[0] ** 2 / pca.explained_variance_).sum()))
    return recon_error, mahal


def check_distribution(pca, features, cfg: ValidationConfig, strong_layout=False):
    """Returns (checks, failure_code | None, warnings)."""
    checks, warnings = [], []
    if pca is None or cfg.max_recon_error is None or cfg.max_mahalanobis is None:
        checks.append(Check("feature_distribution", True, "skipped", "run training/calibrate_validation.py"))
        return checks, None, warnings
    rec, mah = ood_scores(pca, features)
    limit = cfg.max_recon_error * (cfg.soft_band if strong_layout else 1.0)
    ok1 = rec <= limit
    ok2 = mah <= cfg.max_mahalanobis
    checks.append(Check("pca_reconstruction_error", ok1, round(rec, 3), f"<= {limit:.3f}"))
    checks.append(Check("mahalanobis_distance", ok2, round(mah, 3), f"<= {cfg.max_mahalanobis:.3f}"))
    if ok1 and ok2 and rec > cfg.max_recon_error:
        warnings.append("LOW_IMAGE_QUALITY: this image is lower quality than the training images "
                        "(small or compressed), so the result may be less reliable.")
    return checks, (None if (ok1 and ok2) else OUT_OF_DISTRIBUTION), warnings


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
class ECGValidator:
    def __init__(self, pca=None, config: Optional[ValidationConfig] = None):
        self.pca = pca
        self.cfg = config or ValidationConfig.load()

    def _fail(self, code, stage, checks, twelve_lead=False):
        return ValidationResult(False, code, MESSAGES[code], stage, checks, twelve_lead=twelve_lead)

    def validate_bytes(self, data: bytes, rotation: int = 0) -> ValidationResult:
        checks = []

        rgb, c, code = decode_image(data, self.cfg)
        checks += c
        if code:
            return self._fail(code, "file", checks)

        return self.validate_array(rgb, checks, rotation=rotation)

    def _generic_path(self, rgb):
        """
        The page does not look like a Mendeley page. Look for a 12-lead ECG of ANY
        layout: find the ECG area, the waveform rows, and rebuild the reference
        geometry. Returns (canvas | None, checks, failure_code | None, warnings).
        """
        cfg = self.cfg
        frame = detect_grid_frame(rgb)
        area = rgb[frame[1]:frame[3] + 1, frame[0]:frame[2] + 1] if frame else rgb
        gs = grid_score(area, cfg)
        lay = ecg_layout.analyze(rgb, frame)
        light = lay.paper_light
        h_, w_ = rgb.shape[:2]
        aspect = w_ / h_
        checks = [
            Check("page_orientation", 0.5 <= aspect <= 4.5, round(aspect, 2), "0.5-4.5"),
            Check("light_paper_background", light, light, "True"),
            Check("ecg_grid_detected", gs >= 0.25, round(gs, 3), ">= 0.25"),
            Check("waveform_rows_found", lay.kind is not None, lay.n_rows, "3-4, 6-7 or 12-13 rows"),
            Check("rows_with_heartbeats", lay.n_spiky >= 1, lay.n_spiky, ">= half of the rows"),
        ]
        evidence = (light and gs >= 0.25 and lay.n_spiky >= max(1, 0.3 * lay.n_rows)
                    and 0.5 <= aspect <= 4.5)
        if not evidence:
            return None, checks, NOT_ECG, []
        checks.append(Check("page_upright", lay.upright, lay.upright, "True (rotate sideways images)"))
        if lay.kind is None or not lay.upright:
            return None, checks, NOT_12_LEAD, []
        # a portrait page can only be a 6x2 / 12x1 layout; a portrait "3x4" is a sideways page
        if aspect < 1.0 and lay.kind == "3x4":
            checks.append(Check("sideways_page", False, round(aspect, 2), "rotate the image upright"))
            return None, checks, NOT_12_LEAD, []
        canvas = ecg_layout.canonicalize(lay)
        if canvas is None:
            return None, checks, NOT_12_LEAD, []
        note = ("REALIGNED: this ECG has a different page layout from the training images "
                f"(detected {lay.kind} layout) and was automatically re-arranged, so the result may be less reliable.")
        return canvas, checks, None, [note]

    def validate_array(self, rgb: np.ndarray, checks=None, rotation: int = 0) -> ValidationResult:
        checks = list(checks or [])
        warnings = []
        generic = False
        metadata = None

        # (a) the page looks like the training pages: the fixed lead boxes hold waveforms
        #     AND it is either the exact Mendeley template or a proper 12-lead layout
        c, code = check_ecg_like(rgb, self.cfg)
        native_ok = False
        if code is None:
            frame = detect_grid_frame(rgb)
            if frame and is_mendeley_template(frame, rgb.shape):
                native_ok = True
            else:
                lay = ecg_layout.analyze(rgb, frame)
                native_ok = lay.kind is not None and lay.paper_light and lay.upright
                generic = native_ok            # not the exact training template: lenient quality band
        if native_ok and not rotation:
            checks += c
            # Exact training pages retain their original feature geometry.
            h, w = rgb.shape[:2]
            reference = cv2.resize(rgb, (2213, 1572), interpolation=cv2.INTER_AREA)
            rows = [ecg_layout.Row(y, 150, 2125, 1., 2., True) for y in (450, 750, 1050)]
            lay = ecg_layout.Layout('3x4', 3, 3, rows, ecg_layout._working_gray(reference), paper_light=True, n_spiky=3)
            region = ecg_region.Region((0, 0, w, h), 0, rgb, lay, 1.)
            metadata = ecg_region.describe(rgb, region)
        else:
            # Isolate waveform panels before layout validation; notes are never
            # fed into the classifier. Rotation is local to this request.
            region = ecg_region.locate(rgb, grid_score, self.cfg)
            if region is None:
                _, c2, code2, _ = self._generic_path(rgb)
                checks += [Check("ecg_region_located", False, 0, "one complete ECG panel")] + c2
                return self._fail(code2 or NOT_12_LEAD, "ecg_check", checks)
            if rotation:
                region.rgb = np.rot90(region.rgb, 2).copy()
                lay = ecg_layout.analyze(region.rgb)
                cleaned = None
                if lay.kind is None:
                    lay, cleaned, _ = ecg_region.broad_layout(region.rgb)
                if lay is None or lay.kind is None:
                    return self._fail(NOT_12_LEAD, "ecg_check", checks)
                region.layout, region.cleaned = lay, cleaned
                region.rotation = (region.rotation + 180) % 360
            metadata = ecg_region.describe(rgb, region)
            checks += [Check("ecg_region_located", True, list(region.box), "ECG panel, excluding adjacent notes"),
                       Check("orientation_corrected", True, region.rotation, "degrees counterclockwise"),
                       Check("twelve_lead_layout", True, region.layout.kind, "3x4, 6x2 or 12x1"),
                       Check("lead_regions_extracted", metadata['lead_count'] == 12, metadata['lead_count'], "12")]
            lay = replace(region.layout, gray=region.cleaned) if region.cleaned is not None else region.layout
            canvas = ecg_layout.canonicalize(lay)
            if canvas is None:
                failed = self._fail(POOR_SIGNAL, "signal", checks, twelve_lead=True)
                failed.preprocessing = metadata
                return failed
            warnings = ["REALIGNED: ECG panel isolated and arranged into the model's reference layout. Review the orientation and assumed lead labels; classification may be less reliable for this source."]
            rgb, generic = canvas, True

        try:
            features, failed = process_ecg_array(rgb)
        except Exception as e:                      # extractor must never crash the API
            checks.append(Check("extraction", False, type(e).__name__, "no error"))
            failed = self._fail(POOR_SIGNAL, "signal", checks, twelve_lead=True)
            failed.preprocessing, failed.warnings = metadata, warnings
            return failed
        c, code = check_signal(features, failed, self.cfg)
        checks += c
        if code:
            failed = self._fail(code, "signal", checks, twelve_lead=True)
            failed.preprocessing, failed.warnings = metadata, warnings
            return failed

        cells = next((c.value for c in checks if c.name == "lead_cells_with_waveform"), 0)
        c, code, warns = check_distribution(self.pca, features, self.cfg, strong_layout=(generic or cells == 12))
        checks += c
        if code:
            failed = self._fail(code, "features", checks, twelve_lead=True)
            failed.preprocessing, failed.warnings = metadata, warnings + warns
            return failed

        if any(c.name == 'image_format' and c.value == 'WEBP' for c in checks):
            warnings.append('FORMAT_NORMALIZED: The uploaded file contains WebP image data despite its filename; it was decoded to RGB before processing.')
        return ValidationResult(True, OK, MESSAGES[OK], "passed", checks, features, warnings + warns,
                                twelve_lead=True, preprocessing=metadata)
