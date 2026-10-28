"""
Synthetic test images for the validator.

IMPORTANT: these are only for smoke-testing the validation logic. They are
NOT a replacement for running validation/evaluate_validation.py on your real
Mendeley images, which is how the thresholds should be confirmed.
"""
import numpy as np
import cv2

H, W = 1572, 2213
# (row1,row2,col1,col2) of the 12 lead cells - same as the extractor
CELLS = [
    (300, 600, 150, 643), (300, 600, 646, 1135), (300, 600, 1140, 1625), (300, 600, 1630, 2125),
    (600, 900, 150, 643), (600, 900, 646, 1135), (600, 900, 1140, 1625), (600, 900, 1630, 2125),
    (900, 1200, 150, 643), (900, 1200, 646, 1135), (900, 1200, 1140, 1625), (900, 1200, 1630, 2125),
]
LEADS = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"]


def _beat(n, rng, amp=1.0):
    t = np.linspace(0, 1, n)
    g = lambda c, w, a: a * np.exp(-((t - c) ** 2) / (2 * w ** 2))
    return (g(.18, .025, .12) + g(.36, .008, -.12) + g(.40, .010, 1.0 * amp)
            + g(.44, .010, -.25) + g(.68, .05, .25 * rng.uniform(.5, 1.5)))


def _trace(width, rng, beats=None):
    beats = beats or rng.integers(2, 5)
    per = width // beats
    y = np.concatenate([_beat(per, rng, rng.uniform(.6, 1.4)) for _ in range(beats)])
    y = np.pad(y, (0, width - len(y)), mode="edge")
    return y + rng.normal(0, .004, width)


def _paper(h, w, grid_bgr=(200, 190, 255), bg=(250, 250, 250), minor=12, major=60):
    img = np.full((h, w, 3), bg, np.uint8)
    for x in range(0, w, minor):
        cv2.line(img, (x, 0), (x, h), tuple(int(.5 * c + .5 * 255) for c in grid_bgr), 1)
    for y in range(0, h, minor):
        cv2.line(img, (0, y), (w, y), tuple(int(.5 * c + .5 * 255) for c in grid_bgr), 1)
    for x in range(0, w, major):
        cv2.line(img, (x, 0), (x, h), grid_bgr, 2)
    for y in range(0, h, major):
        cv2.line(img, (0, y), (w, y), grid_bgr, 2)
    return img


def make_ecg(seed=0, grid=(200, 190, 255), cells=CELLS, labels=True):
    rng = np.random.default_rng(seed)
    img = _paper(H, W, grid_bgr=grid)
    cv2.putText(img, f"Patient {seed:04d}   25 mm/s   10 mm/mV", (160, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
    for i, (r1, r2, c1, c2) in enumerate(cells):
        w = c2 - c1
        y = _trace(w, rng)
        base = (r1 + r2) // 2 + 40
        ys = (base - y * 110).astype(int)
        pts = np.stack([np.arange(w) + c1, ys], 1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], False, (15, 15, 15), 3, cv2.LINE_AA)
        if labels:
            cv2.putText(img, LEADS[i % 12], (c1 + 5, r1 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (20, 20, 20), 3, cv2.LINE_AA)
    # rhythm strip
    w = 1975
    y = _trace(w, rng, beats=10)
    pts = np.stack([np.arange(w) + 150, (1365 - y * 110).astype(int)], 1).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts], False, (15, 15, 15), 3, cv2.LINE_AA)
    return img


# ---------- degradations that real internet / phone images go through ----------
def jpeg(img, q=40):
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def rescale(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 245))


def dim(img, f=.6):  # badly lit photo
    return (img.astype(np.float32) * f).astype(np.uint8)


def gray3(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def blur(img, k=7):
    return cv2.GaussianBlur(img, (k, k), 0)


# ---------- negatives ----------
def neg_white():  return np.full((H, W, 3), 255, np.uint8)
def neg_noise(seed=0): return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)
def neg_gradient():
    g = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1)); return np.stack([g] * 3, 2)
def neg_grid_only(): return _paper(H, W)                      # empty ECG paper
def neg_text_page():
    img = np.full((H, W, 3), 255, np.uint8)
    for i in range(30):
        cv2.putText(img, "The quick brown fox jumps over the lazy dog 0123456789", (120, 80 + i * 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def neg_line_chart(seed=0):
    """single line chart with a grid (e.g. a stock chart)"""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 255, np.uint8)
    for x in range(100, W - 100, 100): cv2.line(img, (x, 100), (x, H - 100), (200, 200, 200), 1)
    for y in range(100, H - 100, 100): cv2.line(img, (100, y), (W - 100, y), (200, 200, 200), 1)
    ys = (H / 2 + np.cumsum(rng.normal(0, 6, W - 200))).clip(120, H - 120)
    pts = np.stack([np.arange(W - 200) + 100, ys], 1).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def neg_other_layout_ecg(seed=0):
    """A real-looking ECG on grid paper, but 2 columns x 6 rows (not the 4x3 layout)."""
    cells = []
    for r in range(6):
        for c in range(2):
            r1 = 150 + r * 200; c1 = 150 + c * 1000
            cells.append((r1, r1 + 200, c1, c1 + 950))
    rng = np.random.default_rng(seed)
    img = _paper(H, W)
    for (r1, r2, c1, c2) in cells:
        w = c2 - c1; y = _trace(w, rng)
        pts = np.stack([np.arange(w) + c1, ((r1 + r2) // 2 + 30 - y * 80).astype(int)], 1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], False, (15, 15, 15), 3, cv2.LINE_AA)
    return img


# ---------- other 12-lead page layouts (for the generic layout detector) ----------
def make_ecg_layout(kind="6x2", seed=0, rhythm=True, w=1600, h=1000, grid=(200, 190, 255)):
    """Synthetic 12-lead ECG on grid paper laid out as 3x4, 6x2 or 12x1 (+ optional rhythm strip)."""
    rng = np.random.default_rng(seed)
    img = _paper(h, w, grid_bgr=grid, minor=8, major=40)
    rows = {"3x4": 3, "6x2": 6, "12x1": 12}[kind] + (1 if rhythm else 0)
    pitch = (h - 80) / rows
    for r in range(rows):
        base = 40 + pitch * (r + .6)
        n = w - 120
        y = _trace(n, rng, beats=int(rng.integers(8, 14)) if (rhythm and r == rows - 1) else int(rng.integers(3, 6)))
        ys = (base - y * pitch * .45).astype(int)
        pts = np.stack([np.arange(n) + 60, ys], 1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], False, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.putText(img, "lead", (62, int(base - pitch * .4)), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def make_ecg_rows(n_rows, seed=0, w=1600, h=900):
    """n rows of ECG on grid paper (to test 'not 12 lead' rejection: n=1,2,5,8...)."""
    rng = np.random.default_rng(seed)
    img = _paper(h, w, minor=8, major=40)
    pitch = (h - 80) / n_rows
    for r in range(n_rows):
        base = 40 + pitch * (r + .6)
        y = _trace(w - 120, rng, beats=int(rng.integers(6, 12)))
        pts = np.stack([np.arange(w - 120) + 60, (base - y * pitch * .45).astype(int)], 1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], False, (15, 15, 15), 2, cv2.LINE_AA)
    return img


def neg_multipanel_chart(n=4, seed=0, w=1600, h=900):
    """n stacked line charts (sine + noise) with a regular grid - looks like rows of waveform."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 255, np.uint8)
    for x in range(0, w, 40): cv2.line(img, (x, 0), (x, h), (215, 215, 215), 1)
    for y in range(0, h, 40): cv2.line(img, (0, y), (w, y), (215, 215, 215), 1)
    pitch = (h - 80) / n
    for r in range(n):
        base = 40 + pitch * (r + .6)
        t = np.arange(w - 120)
        y = np.sin(t / (40 + 15 * r)) * .8 + rng.normal(0, .05, len(t))
        pts = np.stack([t + 60, (base - y * pitch * .35).astype(int)], 1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 0), 2, cv2.LINE_AA)
    return img
