"""
Generic 12-lead ECG layout detection (works on pages that do NOT look like the
Mendeley training pages).

Idea: a 12-lead ECG is printed as a small number of evenly spaced horizontal
rows of waveform on grid paper:
    3 rows (+ optional rhythm strip)  -> 3 x 4 layout   (most common)
    6 rows (+ optional rhythm strip)  -> 6 x 2 layout
    12 rows (+ optional rhythm strip) -> 12 x 1 layout
We find those rows, check each really contains an ECG-like waveform (long trace
with sharp QRS spikes), and rebuild the Mendeley reference geometry from them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

# Mendeley reference geometry used by the extractor (row/col boxes of the 12 cells)
REF_ROWS = [(300, 600), (600, 900), (900, 1200)]
REF_COLS = [(150, 643), (646, 1135), (1140, 1625), (1630, 2125)]
# (row, col) of every lead in the reference 3x4 layout
REF_POS = {"I": (0, 0), "aVR": (0, 1), "V1": (0, 2), "V4": (0, 3),
           "II": (1, 0), "aVL": (1, 1), "V2": (1, 2), "V5": (1, 3),
           "III": (2, 0), "aVF": (2, 1), "V3": (2, 2), "V6": (2, 3)}
# lead order of the common alternatives
ORDER_6x2 = [["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"]]
ORDER_12x1 = [["I"], ["II"], ["III"], ["aVR"], ["aVL"], ["aVF"], ["V1"], ["V2"], ["V3"], ["V4"], ["V5"], ["V6"]]
ORDER_3x4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]


@dataclass
class Row:
    center: float            # y of baseline (pixels in the working image)
    x0: int
    x1: int                  # horizontal extent of the trace
    coverage: float          # share of the page width covered by trace
    spikiness: float         # number of heart-beat-like tall strokes
    ecg_like: bool


@dataclass
class Layout:
    kind: Optional[str]                  # '3x4' | '6x2' | '12x1' | None
    n_rows: int                          # waveform rows found (ECG-like ones)
    n_bands: int                         # all candidate rows before the ECG-like filter
    rows: list = field(default_factory=list)
    gray: Optional[np.ndarray] = None    # channel-suppressed working image (uint8)
    note: str = ""
    paper_light: bool = False            # bright paper background (ECG printouts are light)
    upright: bool = True                 # baselines run horizontally (page is not sideways)
    n_spiky: int = 0                     # rows with heart-beat-like spikes


def _working_gray(rgb: np.ndarray):
    """Brightest colour channel: coloured grid lines vanish, black ink stays dark."""
    ch = int(np.argmax(rgb.reshape(-1, 3).mean(0)))
    return rgb[..., ch]


def _trace_mask(gray: np.ndarray):
    g = cv2.GaussianBlur(gray, (0, 0), 0.6).astype(np.float32) / 255.0
    sub = g[::2, ::2]
    lo, hi = np.percentile(sub, 0.05), np.percentile(sub, 90)
    if hi - lo < 0.2 or lo > 0.6:
        return None
    dark = ((g - lo) / (hi - lo)) < 0.62
    H, W = dark.shape
    d8 = dark.astype(np.uint8)

    # 1. cut long vertical lines (lead separators, page borders)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(H * 0.36))))
    vert = cv2.dilate(cv2.morphologyEx(d8, cv2.MORPH_OPEN, vk), np.ones((3, 3), np.uint8))
    d8 = d8 & (1 - vert)

    # 2. drop small components (text, labels, calibration pulse): keep wide pieces only
    n, lab, st, _ = cv2.connectedComponentsWithStats(d8, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_WIDTH] >= 0.03 * W
    return keep[lab]


def _orientation_upright(gray: np.ndarray) -> bool:
    """Upright ECG: ink forms long HORIZONTAL runs (baselines). Sideways: long vertical runs."""
    g = cv2.GaussianBlur(gray, (0, 0), 0.6).astype(np.float32) / 255.0
    sub = g[::2, ::2]
    lo, hi = np.percentile(sub, 0.05), np.percentile(sub, 90)
    if hi - lo < 0.2:
        return True
    d8 = (((g - lo) / (hi - lo)) < 0.62).astype(np.uint8)
    H, W = d8.shape
    hm = cv2.morphologyEx(d8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(W * 0.06)), 1))).sum()
    vm = cv2.morphologyEx(d8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, int(H * 0.06))))).sum()
    return bool(hm >= 1.3 * vm)


def _find_rows(mask: np.ndarray):
    """Horizontal bands of trace = peaks of the row-density profile."""
    H, W = mask.shape
    prof = mask.sum(1).astype(np.float32)
    k = max(3, int(H * 0.006)) | 1
    prof = np.convolve(prof, np.hanning(k * 2 + 1) / np.hanning(k * 2 + 1).sum(), "same")
    if prof.max() <= 0:
        return []
    peaks, props = find_peaks(prof, height=0.12 * prof.max(), distance=max(5, int(H * 0.05)),
                              prominence=0.06 * prof.max())
    return _clean_rows([float(p) for p in peaks], [float(h) for h in props["peak_heights"]])


def _clean_rows(centers, heights):
    """Remove spurious rows (footer text, side panels): near-duplicates and odd end rows."""
    c, h = list(centers), list(heights)
    for _ in range(20):
        if len(c) < 3:
            break
        d = np.diff(c)
        pitch = float(np.median(d))
        # (1) two rows much closer than the normal spacing: keep the stronger one
        i = int(np.argmin(d))
        if d[i] < 0.5 * pitch:
            drop = i if h[i] < h[i + 1] else i + 1
            del c[drop], h[drop]
            continue
        # (2) end rows whose spacing does not fit the rest
        if d[0] < 0.55 * pitch or d[0] > 1.7 * pitch:
            del c[0], h[0]
            continue
        if d[-1] < 0.55 * pitch or d[-1] > 1.7 * pitch:
            del c[-1], h[-1]
            continue
        break
    return c


def _row_stats(mask: np.ndarray, center: float, pitch: float):
    """coverage of the row and number of QRS-like tall strokes (heart beats)."""
    H, W = mask.shape
    y0, y1 = max(0, int(center - pitch * 0.5)), min(H, int(center + pitch * 0.5))
    band = mask[y0:y1]
    has = band.any(0)
    xs = np.where(has)[0]
    if len(xs) < 30:
        return None
    first = np.argmax(band, axis=0)
    last = band.shape[0] - 1 - np.argmax(band[::-1], axis=0)
    mid = (first + last) / 2.0
    ext = np.where(has, last - first + 1, 0)
    med = np.median(ext[has])
    # tall but NOT a full-row-height stroke (those are lead separators, not heart beats)
    tall = has & (ext >= max(3 * med, 0.08 * pitch, 6)) & (ext <= 0.92 * pitch)
    # group neighbouring tall columns into single beats
    beats, prev = 0, -999
    for x in np.where(tall)[0]:
        if x - prev > 0.012 * W:
            beats += 1
        prev = x
    # a perfectly straight, beat-less line (page rule, border, underline) is not a waveform
    if beats < 2 and (np.percentile(mid[has], 95) - np.percentile(mid[has], 5)) < 0.04 * pitch:
        return None
    return int(xs[0]), int(xs[-1]), float(has.mean()), beats


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Undo a small tilt (<= 3 degrees): choose the angle with the sharpest row profile."""
    h, w = gray.shape
    small = cv2.resize(gray, (min(w, 700), max(20, int(h * min(w, 700) / w))), interpolation=cv2.INTER_AREA)
    bg = float(np.median(small))
    inv = np.clip(bg - small.astype(np.float32), 0, None)       # ink = positive, paper = 0
    best, best_ang = -1.0, 0.0
    for ang in (0, -0.5, 0.5, -1, 1, -1.5, 1.5, -2, 2, -2.5, 2.5, -3, 3):
        M = cv2.getRotationMatrix2D((small.shape[1] / 2, small.shape[0] / 2), ang, 1.0)
        r = cv2.warpAffine(inv, M, (small.shape[1], small.shape[0]), borderValue=0)
        score = float(np.var(r.sum(1)))
        if score > best * 1.02 or (best_ang == 0.0 and best < 0):
            best, best_ang = score, ang
    if best_ang == 0.0:
        return gray
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=float(np.median(gray)))


def analyze(rgb: np.ndarray, frame=None) -> Layout:
    """
    rgb   : full image
    frame : optional (x0,y0,x1,y1) of the ECG grid area; the whole image if None
    """
    if frame is not None:
        x0, y0, x1, y1 = frame
        rgb = rgb[y0:y1 + 1, x0:x1 + 1]
    # normalise size for speed/consistency
    h, w = rgb.shape[:2]
    s = min(1.0, 1400.0 / w)                     # shrink big images, never upscale small ones
    if s < 1.0:
        rgb = cv2.resize(rgb, (1400, max(50, int(h * s))), interpolation=cv2.INTER_AREA)
    gray = _working_gray(rgb)
    gray = _deskew(gray)
    # ECG paper = a large, evenly lit, light area (also true for a dim photo of it)
    g32 = cv2.resize(gray, (160, max(8, int(160 * gray.shape[0] / gray.shape[1]))), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    med = float(np.median(g32))
    uniform = float((np.abs(g32 - med) <= 0.12).mean())
    light = bool(med >= 0.75 or (med >= 0.35 and uniform >= 0.6))
    upright = True
    mask = _trace_mask(gray)
    if mask is None:
        return Layout(None, 0, 0, gray=gray, note="no dark waveform ink", paper_light=light, upright=upright)
    H, W = mask.shape
    centers = _find_rows(mask)
    rows = []
    pitch = float(np.median(np.diff(centers))) if len(centers) > 1 else H / 3.0
    for c in centers:
        st = _row_stats(mask, c, pitch)
        if st is None:
            continue
        x0_, x1_, cov, beats = st
        span = (x1_ - x0_) / W
        # a waveform row spans most of the page width; spikiness = number of beats
        if span >= 0.5 and cov >= 0.35:
            rows.append(Row(c, x0_, x1_, float(span), float(beats), beats >= 2))
    good = rows
    spiky = [r for r in rows if r.ecg_like]
    lay = Layout(None, len(rows), len(centers), rows, gray, paper_light=light, n_spiky=len(spiky), upright=upright)
    lay.note = f"{len(spiky)}/{len(rows)} rows have heart-beat-like spikes"

    # layout from the number of waveform rows (+1 = optional rhythm strip)
    n = len(good)
    if n in (3, 4):
        lay.kind = "3x4"
    elif n in (6, 7):
        lay.kind = "6x2"
    elif n in (12, 13):
        lay.kind = "12x1"
    # waveform rows of a real 12-lead page run (nearly) the full page width; rows 'found' in a
    # sideways page do not
    if lay.kind and np.mean([r.coverage for r in good]) < 0.84:
        lay.kind = None
        lay.note = 'rows do not span the page (sideways image?)'
    # most rows must really look like ECG traces
    if lay.kind and len(spiky) < 0.5 * n:
        lay.kind = None
        lay.note = f"rows do not look like ECG waveforms ({len(spiky)}/{n} have beats)"
    # rows must be evenly spaced
    if lay.kind and len(good) > 2:
        d = np.diff([r.center for r in good])
        # rhythm strip may be spaced a little differently: ignore the last gap
        core = d[:-1] if n in (4, 7, 13) else d
        if len(core) >= 2 and np.std(core) / np.mean(core) > 0.30:
            lay.kind = None
            lay.note = "rows are not evenly spaced"
    return lay


def canonicalize(lay: Layout) -> Optional[np.ndarray]:
    """
    Rebuild the Mendeley reference page (2213 x 1572, RGB) from a detected layout:
    every lead is cut out of its row and pasted into the reference cell that the
    extractor/model expect for that lead.
    """
    if lay.kind is None or lay.gray is None:
        return None
    order = {"3x4": ORDER_3x4, "6x2": ORDER_6x2, "12x1": ORDER_12x1}[lay.kind]
    rows = lay.rows[:len(order)]
    if len(rows) < len(order):
        return None
    pitch = float(np.median(np.diff([r.center for r in rows]))) if len(rows) > 1 else lay.gray.shape[0] / 3
    H, W = lay.gray.shape
    canvas = np.full((1572, 2213), 255, np.uint8)
    for row, leads in zip(rows, order):
        y0, y1 = max(0, int(row.center - pitch / 2)), min(H, int(row.center + pitch / 2))
        span = row.x1 - row.x0
        for j, lead in enumerate(leads):
            if len(leads) == 1:                       # 12x1: take the first quarter of the row
                xa, xb = row.x0, row.x0 + span / 4
            else:
                xa, xb = row.x0 + span * j / len(leads), row.x0 + span * (j + 1) / len(leads)
            piece = lay.gray[y0:y1, int(xa):max(int(xb), int(xa) + 2)]
            if piece.size == 0:
                return None
            r, c = REF_POS[lead]
            (ry0, ry1), (cx0, cx1) = REF_ROWS[r], REF_COLS[c]
            canvas[ry0:ry1, cx0:cx1] = cv2.resize(piece, (cx1 - cx0, ry1 - ry0), interpolation=cv2.INTER_AREA)
    return np.repeat(canvas[..., None], 3, axis=2)
