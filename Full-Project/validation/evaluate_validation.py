"""
Measure the validator on REAL images. Run from the project root:

    python validation/evaluate_validation.py --dataset dataset --sample 60     (quick test first)
    python validation/evaluate_validation.py --dataset dataset                 (all images)
    python validation/evaluate_validation.py --dataset dataset --negatives my_non_ecg_images --internet my_internet_ecgs

--dataset    your Mendeley folder with NORMAL/ AHB/ MI/ PMI/ sub-folders.
             EVERY image here is a genuine ECG -> any rejection is a FALSE REJECTION.
--negatives  (optional) folder of images that are NOT ECGs (photos, screenshots,
             documents, charts...). Every one should be rejected.
--internet   (optional) folder of ECG images downloaded from the internet. These
             are shown for information: they may be rejected because their layout
             or style differs from the training data.

If --negatives is omitted, built-in sample photos + synthetic negatives are used.
Writes validation/report.csv with one row per image.
"""
import argparse
import os
import sys
import warnings
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))

import joblib                                                   # noqa: E402
from preprocessing.ecg_validator import ECGValidator, ValidationConfig   # noqa: E402

warnings.filterwarnings("ignore")
EXT = (".jpg", ".jpeg", ".png")
_V = None


def _init():
    global _V
    warnings.filterwarnings("ignore")
    pca = joblib.load(os.path.join(ROOT, "models", "pca.pkl"))
    cfg = ValidationConfig.load(os.path.join(ROOT, "models", "validation_thresholds.json"))
    _V = ECGValidator(pca, cfg)


def _one(args):
    p, group = args
    with open(p, "rb") as fh:
        r = _V.validate_bytes(fh.read())
    failed = ";".join(c.name for c in r.checks if not c.passed)
    return {"group": group, "file": p, "valid": r.valid, "code": r.code, "stage": r.stage,
            "failed_checks": failed, **{c.name: c.value for c in r.checks}}


def list_images(folder):
    for dp, _, fs in os.walk(folder):
        for f in sorted(fs):
            if f.lower().endswith(EXT):
                yield os.path.join(dp, f)


def run(validator, paths, group, rows, workers):
    jobs = [(p, group) for p in paths]
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        for i, row in enumerate(ex.map(_one, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"  [{group}] {i}/{len(jobs)} done", flush=True)


def builtin_negatives(tmp):
    import synthetic as S
    from skimage import data
    os.makedirs(tmp, exist_ok=True)
    out = []
    for n in ["astronaut", "coffee", "camera", "chelsea", "page", "text", "brick", "grass", "coins", "rocket"]:
        im = getattr(data, n)()
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.ndim == 3 else im
        out.append((n, im))
    for n in ["neg_white", "neg_noise", "neg_text_page", "neg_line_chart", "neg_grid_only"]:
        out.append((n, getattr(S, n)()))
    paths = []
    for n, im in out:
        p = os.path.join(tmp, n + ".png"); cv2.imwrite(p, im); paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--negatives")
    ap.add_argument("--internet")
    ap.add_argument("--sample", type=int, help="only check N random dataset images (quick test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    v = None
    rows = []

    ds = list(list_images(a.dataset)) if os.path.isdir(a.dataset) else []
    if a.sample and len(ds) > a.sample:
        random.seed(0); ds = random.sample(ds, a.sample)
    print(f"Dataset images to check: {len(ds)}  (workers: {a.workers})")
    run(v, ds, "dataset", rows, a.workers)

    neg = list(list_images(a.negatives)) if a.negatives else builtin_negatives(os.path.join(ROOT, "validation", "_neg"))
    run(v, neg, "negative", rows, a.workers)
    if a.internet:
        run(v, list(list_images(a.internet)), "internet_ecg", rows, a.workers)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "validation", "report.csv"), index=False)

    print("\n=========== RESULT ===========")
    for g, d in df.groupby("group"):
        acc = d["valid"].mean()
        if g == "dataset":
            print(f"\n[dataset]  accepted {d['valid'].sum()}/{len(d)} = {acc:.1%}   (want ~100%; rest are FALSE REJECTIONS)")
        elif g == "negative":
            print(f"\n[non-ECG]  rejected {(~d['valid']).sum()}/{len(d)} = {1-acc:.1%}   (want 100%)")
        else:
            print(f"\n[internet ECGs] accepted {d['valid'].sum()}/{len(d)} = {acc:.1%}")
        print("  outcome codes:", dict(Counter(d["code"])))
        bad = d[~d["valid"]] if g != "negative" else d[d["valid"]]
        if len(bad):
            print("  " + ("false rejections" if g == "dataset" else "unexpected" if g == "negative" else "rejected") + " (first 15):")
            for _, r in bad.head(15).iterrows():
                print(f"    {r['file']}  -> {r['code']}  failed: {r['failed_checks']}")
    print("\nFull per-image report: validation/report.csv")
    print("If genuine dataset images are rejected, look at the 'failed_checks' column and "
          "relax that threshold in ValidationConfig (preprocessing/ecg_validator.py).")


if __name__ == "__main__":
    main()
