"""Locate ECG panels in mixed text/ECG scans, orient them and expose reviewable crops.

No OCR or diagnosis is performed. Lead names are assigned from standard printed
layout order; the UI explicitly asks the user to review these assignments.
"""
from __future__ import annotations
import base64
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.signal import find_peaks
from preprocessing import ecg_layout


@dataclass
class Region:
    box: tuple
    rotation: int
    rgb: np.ndarray
    layout: object
    score: float
    cleaned: np.ndarray | None = None


def data_image(rgb, width=1000):
    h, w = rgb.shape[:2]
    if w > width:
        rgb = cv2.resize(rgb, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii') if ok else None


def _runs(flags):
    padded = np.pad(flags.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def panel_boxes(rgb):
    """Split only on near-black page gutters, never on text or individual traces."""
    h, w = rgb.shape[:2]
    scale = min(1., 1200 / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gutter = (gray < 70).mean(0) > .82
    cuts = [0] + [round((a + b) / 2 / scale) for a, b in _runs(gutter) if .18 * gray.shape[1] < (a + b) / 2 < .82 * gray.shape[1]] + [w]
    boxes = [(0, 0, w, h)]
    for a, b in zip(cuts, cuts[1:]):
        if b - a < .23 * w:
            continue
        panel = rgb[:, a:b]
        light = np.max(panel, axis=2) > 100
        ys = np.where(light.mean(1) > .65)[0]
        xs = np.where(light.mean(0) > .65)[0]
        if len(ys) and len(xs):
            # Exclude binding edges and black scanner borders.
            x0, x1 = a + int(xs[0]), a + int(xs[-1]) + 1
            y0, y1 = int(ys[0]), int(ys[-1]) + 1
            pad_x, pad_y = max(1, round((x1-x0)*.012)), max(1, round((y1-y0)*.012))
            box = (x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y)
            if box not in boxes:
                boxes.append(box)
    return boxes[1:] if len(cuts) > 2 else boxes


def broad_layout(rgb):
    """Fallback for monochrome grids and broad waves: remove straight paper rules
    before finding row envelopes. This does not require narrow, normal-looking QRS.
    """
    h, w = rgb.shape[:2]
    scale = 1200 / w
    a = cv2.resize(rgb, (max(30, round(w*scale)), max(30, round(h*scale))))
    gray = ecg_layout._working_gray(a)
    H, W = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    kernel = max(15, round(W * .025))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((kernel, 1), np.uint8))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, kernel), np.uint8))
    lines = cv2.dilate(vert | horiz, np.ones((3, 3), np.uint8))
    ink = cv2.subtract(bw, lines)
    # Scanner edges and text must not dominate row finding.
    margin = max(1, round(W * .015))
    ink[:, :margin] = ink[:, -margin:] = 0
    ink[:max(1, H//50)] = ink[-max(1, H//50):] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = np.zeros(n, bool)
    keep[1:] = (stats[1:, cv2.CC_STAT_WIDTH] >= .009*W) & (stats[1:, cv2.CC_STAT_HEIGHT] >= 3)
    mask = keep[labels]
    profile = cv2.GaussianBlur(mask.sum(1).astype(np.float32)[:, None], (1, 0), max(2, H*.025)).ravel()
    if profile.max() <= 0:
        return None, None, 0
    centers, _ = find_peaks(profile, prominence=profile.max()*.10, distance=max(8, round(H*.14)))
    if len(centers) not in (3, 4):
        return None, None, 0
    pitch = float(np.median(np.diff(centers)))
    if np.std(np.diff(centers))/pitch > .24:
        return None, None, 0
    rows, cell_count = [], 0
    x0, x1 = round(.025*W), round(.975*W)
    for c in centers:
        y0, y1 = max(0, round(c-pitch*.48)), min(H, round(c+pitch*.48))
        band = mask[y0:y1, x0:x1]
        cov = float(band.any(0).mean())
        if cov < .32:
            return None, None, 0
        rows.append(ecg_layout.Row(float(c), x0, x1, cov, 2., True))
    # Each of the 12 assumed cells needs non-straight trace evidence after grid removal.
    for row in rows[:3]:
        y0, y1 = max(0, round(row.center-pitch*.48)), min(H, round(row.center+pitch*.48))
        for j in range(4):
            xa, xb = round(x0+(x1-x0)*j/4), round(x0+(x1-x0)*(j+1)/4)
            cell = mask[y0:y1, xa:xb]
            ys, xs = np.where(cell)
            coverage = cell.any(0).mean()
            variation = np.percentile(ys, 95)-np.percentile(ys, 5) if len(ys) else 0
            cell_count += bool(coverage >= .22 and variation >= max(5, pitch*.08) and .001 <= cell.mean() <= .20)
    light = float(np.median(gray)) >= 110
    if cell_count != 12 or not light:
        return None, None, cell_count
    # Smooth line charts on graph paper also have broad envelopes. Require
    # compact label/calibration ink around the starts of individual lead cells.
    # This is structural evidence, not OCR or a claim that labels were read.
    small = [(x,y,cw,ch) for x,y,cw,ch,area in stats[1:]
             if 2 <= cw <= W*.035 and max(3,pitch*.025) <= ch <= pitch*.3 and .12 <= cw/ch <= 5]
    label_cells = 0
    for row in rows[:3]:
        for j in range(4):
            start = x0 + (x1-x0)*j/4
            hits = sum(start-W*.015 <= x <= start+W*.08 and abs(y+ch/2-row.center) <= pitch*.5
                       for x,y,cw,ch in small)
            label_cells += hits >= 3
    if label_cells < 8:
        return None, None, cell_count
    clean = np.where(mask, 0, 255).astype(np.uint8)
    # Use the original grayscale for spatial crops; grid-suppressed masks for signal extraction.
    lay = ecg_layout.Layout('3x4', len(rows), len(rows), rows, gray, 'Grid-suppressed row envelopes', light, True, len(rows))
    return lay, clean, cell_count


def _start_score(rgb, lay):
    """Calibration/lead-name ink lies on the left of standard printed leads.
    A heuristic only: expose 180-degree correction in the UI for user review.
    """
    g = lay.gray
    h, w = g.shape
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    left = bw[:, round(w*.025):round(w*.10)].mean(1)
    right = bw[:, round(w*.90):round(w*.975)].mean(1)
    return float(np.std(left) - np.std(right)) / 255


def regular_grid(rgb):
    """Reject sparse form/table rules: ECG paper has many regularly spaced lines."""
    from preprocessing.ecg_validator import _masked_profile
    h,w=rgb.shape[:2]
    small=cv2.resize(rgb,(800,max(30,round(h*800/w))))
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    for axis in (0,1):
        p=_masked_profile(gray,axis)
        p=p-np.convolve(np.pad(p,20,mode='edge'),np.ones(41)/41,'valid')
        peaks,_=find_peaks(-p,prominence=max(.001,float(np.std(p))*.7),distance=3)
        gaps=np.diff(peaks)
        if len(gaps)<12:
            continue
        support=max(int((np.abs(gaps-pitch)<=max(2,pitch*.16)).sum()) for pitch in gaps if pitch>=5) if np.any(gaps>=5) else 0
        if support>=12 and support/len(gaps)>=.48:
            return True
    return False


def _trimmed_box(box,k,h,w,trim_y=.05,trim_x=.015):
    x0,y0,x1,y1=box
    if k%2:
        dx,dy=round((x1-x0)*trim_y),round((y1-y0)*trim_x)
    else:
        dx,dy=round((x1-x0)*trim_x),round((y1-y0)*trim_y)
    return x0+dx,y0+dy,x1-dx,y1-dy


def locate(rgb, grid_score, cfg):
    best = None
    for box in panel_boxes(rgb):
        x0, y0, x1, y1 = box
        panel = rgb[y0:y1, x0:x1]
        if not regular_grid(panel):
            continue
        for k in (0, 1, 3, 2):
            rotated = np.rot90(panel, k).copy()
            h, w = rotated.shape[:2]
            if not .55 <= w / h <= 4.5:
                continue
            # Bound working dimensions before the legacy layout detector.
            sh,sw=rotated.shape[:2]
            work=cv2.resize(rotated,(1200,max(30,round(sh*1200/sw))))
            lay = ecg_layout.analyze(work)
            cleaned = None
            chosen_box=box
            if lay.kind is None:
                # Never merge an already clear incomplete layout into three
                # broad envelopes (for example an eight-row partial ECG).
                if lay.n_rows >= 5 and lay.n_spiky >= lay.n_rows * .5:
                    continue
                # Remove only a narrow page margin (binding holes and headings),
                # then locate all four waveform envelopes in the remaining panel.
                margin_y,margin_x=round(h*.05),round(w*.015)
                trimmed=rotated[margin_y:h-margin_y,margin_x:w-margin_x]
                lay, cleaned, cells = broad_layout(trimmed)
                if lay is not None:
                    rotated=trimmed
                    chosen_box=_trimmed_box(box,k,h,w)
            if lay is None or lay.kind is None or not lay.paper_light:
                continue
            if lay.kind == '3x4' and rotated.shape[1] < rotated.shape[0]:
                continue
            gs = grid_score(rotated, cfg)
            # Keep a real grid requirement; text pages and freehand charts must not pass.
            if gs < .18:
                continue
            score = 3 + min(gs, .8) + _start_score(rotated, lay)
            # Prefer isolated panels over a composite containing notes.
            score += .15 * (1 - (x1-x0)*(y1-y0)/(rgb.shape[0]*rgb.shape[1]))
            candidate = Region(chosen_box, (k*90)%360, rotated, lay, score, cleaned)
            if best is None or candidate.score > best.score:
                best = candidate
    return best


def describe(rgb, region):
    x0,y0,x1,y1 = region.box
    annotated = rgb.copy()
    thickness = max(2, round(max(rgb.shape[:2]) / 400))
    cv2.rectangle(annotated, (x0,y0), (x1-1,y1-1), (0,180,140), thickness)
    order = {'3x4':ecg_layout.ORDER_3x4,'6x2':ecg_layout.ORDER_6x2,'12x1':ecg_layout.ORDER_12x1}[region.layout.kind]
    lay=region.layout
    rows=lay.rows[:len(order)]
    pitch=float(np.median(np.diff([r.center for r in rows])))
    H,W=lay.gray.shape
    leads=[]
    for row,names in zip(rows,order):
        ya,yb=max(0,round(row.center-pitch*.5)),min(H,round(row.center+pitch*.5))
        for j,name in enumerate(names):
            span=row.x1-row.x0
            xa=round(row.x0+span*j/len(names)); xb=round(row.x0+span*(j+1)/len(names))
            leads.append({'name':name,'box':[xa,ya,xb,yb],'image':data_image(lay.gray[ya:yb,xa:xb],360)})
    # Boundaries are original-image pixels; lead boxes refer to the upright working image.
    return {'box':[x0,y0,x1,y1], 'original_size':[rgb.shape[1],rgb.shape[0]],
            'rotation_ccw':region.rotation,'layout':lay.kind,'lead_count':len(leads),
            'annotated_image':data_image(annotated),'upright_image':data_image(region.rgb),
            'leads':leads,'label_method':'Standard layout order (not OCR); review lead labels and orientation before analysis.'}
