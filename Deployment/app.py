from __future__ import annotations
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import base64
import hashlib
from pathlib import Path

import altair as alt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st

from Ecg import ECG, ECGInputError, SUPPORTED_EXTENSIONS, LABELS  # noqa: F401

st.set_page_config(
    page_title="CardioLens | ECG Intelligence",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
BG_PATH = BASE_DIR / "assets" / "background.jpg"


@st.cache_data(show_spinner=False)
def load_background(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode()


@st.cache_resource(show_spinner=False)
def get_engine() -> ECG:
    """One shared ECG object, so the PCA and classifier pickles load only once."""
    return ECG()


# DividingLeads() crops in this order: row by row across the 4-column page layout.
LEAD_NAMES = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"]

BG64 = load_background(str(BG_PATH))
BG_URL = f"url('data:image/jpeg;base64,{BG64}')" if BG64 else "linear-gradient(120deg,#12233c,#7a1f3d)"

# ----------------------------------------------------------------------------
# Sidebar: navigation + personalisation
# ----------------------------------------------------------------------------
THEMES = {
    "Coral Pulse": ("#ff4d6d", "#ffb020", "#3b82f6"),
    "Ocean Wave": ("#12b5a6", "#3b82f6", "#7c5cff"),
    "Violet Night": ("#7c5cff", "#ec4899", "#12b5a6"),
}
PAGES = ["🫀  ECG Analysis", "📘  About the Project", "⚠️  Limits & Responsible Use"]

with st.sidebar:
    st.markdown(
        "<div class='brand'>♥ CardioLens</div><div class='brand-sub'>ECG intelligence workspace</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**NAVIGATE**")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="page")
    st.divider()
    st.markdown("**MAKE IT YOURS**")
    theme_name = st.radio("Colour theme", list(THEMES.keys()), key="theme")
    bpm = st.slider("Heartbeat speed (BPM)", 40, 160, 72, help="Only changes the header animation.")
    st.divider()
    st.markdown("**MODEL PIPELINE**")
    st.caption("Image validation → 12-lead extraction → signal normalization → PCA → trained classifier")
    st.markdown(
        "<div class='side-note'>Local processing · The uploaded image is not sent to an external service by this app.</div>",
        unsafe_allow_html=True,
    )

A1, A2, A3 = THEMES[theme_name]
BEAT = round(120 / bpm, 2)
PALETTE = [A1, A3, A2, "#12b5a6", "#7c5cff", "#ec4899", "#84cc16", "#f97316"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&display=swap');
:root{--bg-img:__BG__;--a1:__A1__;--a2:__A2__;--a3:__A3__;--ink:#1b1d3a;--muted:#5f6485;--line:#e6e8f2;--beat:__BEAT__s}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
h1,h2,h3,.display{font-family:'Bricolage Grotesque','DM Sans',sans-serif}

/* Full-page image background (set on .stApp itself, so Streamlit's scrolling is untouched) */
.stApp{background-color:#0c1426;
  background-image:linear-gradient(180deg,rgba(8,12,30,.15),rgba(8,12,30,.45)),var(--bg-img);
  background-size:cover,cover;background-position:center,center;background-repeat:no-repeat,no-repeat;background-attachment:fixed,fixed}
header[data-testid="stHeader"]{background:transparent}

/* Frosted content sheet over the image */
.block-container{max-width:1280px;margin-top:1.2rem;margin-bottom:1.5rem;padding:1.6rem 2rem 1.2rem;
  background:rgba(255,255,255,.90);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border:1px solid rgba(255,255,255,.55);border-radius:26px;box-shadow:0 30px 70px -30px rgba(0,0,0,.65);color:var(--ink)}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(18,16,48,.93),rgba(28,22,70,.92) 55%,rgba(14,40,64,.93));
  backdrop-filter:blur(10px);border-right:1px solid rgba(255,255,255,.12)}
[data-testid="stSidebar"] *{color:#eceaff}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#cfcdf3}
.brand{font-family:'Bricolage Grotesque',sans-serif;font-size:1.55rem;font-weight:800;color:white}
.brand-sub{color:#b9b5ea;font-size:.82rem;margin-top:.15rem}
.side-note{font-size:.76rem;color:#aaa6dd;line-height:1.4}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.5rem .7rem;border-radius:12px;transition:background .15s ease;width:100%}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(255,255,255,.1)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,__A1__cc,__A3__cc)}

/* Hero over the same image */
.hero{position:relative;overflow:hidden;border-radius:22px;padding:2.2rem 2.4rem 1.2rem;margin-bottom:1.3rem;color:white;
  background:linear-gradient(100deg,rgba(10,14,36,.88) 0%,rgba(10,14,36,.55) 55%,rgba(10,14,36,.15) 100%),var(--bg-img) center 45%/cover;
  box-shadow:0 18px 40px -20px __A1__}
.hero h1{color:white;font-size:clamp(1.9rem,3.2vw,2.8rem);line-height:1.1;margin:.5rem 0 .7rem;font-weight:800}
.hero p{color:#e4e2ff;max-width:620px;margin:0;font-size:1.02rem;line-height:1.55}
.chips{margin-top:1.1rem;display:flex;gap:.5rem;flex-wrap:wrap}
.chip{display:inline-block;padding:.34rem .8rem;border-radius:999px;font-size:.78rem;font-weight:600;color:white;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.3)}
.ecg{display:block;width:100%;height:64px;margin-top:1rem}
.ecg path{fill:none;stroke:#fff;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1000;stroke-dashoffset:1000;animation:sweep var(--beat) linear infinite;filter:drop-shadow(0 0 7px __A1__)}
@keyframes sweep{to{stroke-dashoffset:0}}
@media (prefers-reduced-motion:reduce){.ecg path{animation:none;stroke-dashoffset:0}}

/* Tabs as pills */
.stTabs [data-baseweb="tab-list"]{gap:.45rem;border-bottom:0;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{height:2.6rem;padding:0 1.1rem;border-radius:999px;background:#fff;border:1px solid var(--line);font-weight:700;color:var(--ink)}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,var(--a1),var(--a3));color:white !important;border-color:transparent}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}

/* Cards */
.metric{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.1rem 1.25rem;min-height:112px;border-top:5px solid var(--a1);
  box-shadow:0 8px 24px -14px rgba(40,30,100,.35);transition:transform .2s ease,box-shadow .2s ease}
.metric:hover{transform:translateY(-3px);box-shadow:0 14px 28px -14px rgba(40,30,100,.45)}
.metric.m2{border-top-color:var(--a2)} .metric.m3{border-top-color:var(--a3)} .metric.m4{border-top-color:#12b5a6}
.metric-label{color:var(--muted);font-size:.8rem;font-weight:700;letter-spacing:.04em}
.metric-value{font-family:'Bricolage Grotesque',sans-serif;font-size:1.8rem;font-weight:800;margin:.2rem 0}
.subtle{color:var(--muted);font-size:.93rem;line-height:1.5}
.section-title{font-family:'Bricolage Grotesque',sans-serif;font-size:1.25rem;font-weight:800;margin:.5rem 0 .2rem;color:var(--ink)}
.panel{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.1rem 1.3rem;margin-bottom:1rem;box-shadow:0 8px 24px -16px rgba(40,30,100,.35)}
.panel ul{margin:.3rem 0 0 1.1rem;padding:0;color:var(--ink)} .panel li{margin:.25rem 0}
.stage{background:linear-gradient(120deg,__A1__1f,__A3__1f);border:1px solid var(--line);border-radius:18px;padding:1.2rem 1.5rem}
.stage h3{margin:.1rem 0 .4rem;font-size:1.35rem}
.obj{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;min-height:128px;transition:transform .2s ease}
.obj:hover{transform:translateY(-3px)}
.obj .n{display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:50%;color:#fff;font-weight:800;background:linear-gradient(135deg,var(--a1),var(--a3));margin-bottom:.4rem}
.obj b{display:block;font-size:1rem}
.check{display:inline-block;margin:.2rem .3rem .2rem 0;padding:.3rem .75rem;border-radius:999px;font-size:.8rem;font-weight:600;background:#e9f9f4;color:#0f7a63;border:1px solid #c6efe3}
.result{border-radius:22px;padding:1.5rem 1.7rem;color:white;background:linear-gradient(120deg,var(--a1),var(--a3));box-shadow:0 18px 36px -18px var(--a1)}
.result-label{font-size:.82rem;font-weight:700;letter-spacing:.05em;opacity:.92}
.result-value{font-family:'Bricolage Grotesque',sans-serif;font-size:2.1rem;font-weight:800;margin:.3rem 0}
.rchip{display:inline-block;margin-right:.45rem;padding:.25rem .7rem;border-radius:999px;background:rgba(255,255,255,.22);font-size:.82rem;font-weight:600}
.note{background:#fff6e6;border:1px solid #ffe0a6;border-radius:14px;padding:.9rem 1.1rem;color:#7a5200;font-size:.9rem;margin-top:1rem}
.person{background:#fff;border:1px solid var(--line);border-radius:16px;padding:.9rem 1.1rem;display:flex;align-items:center;gap:.8rem}
.avatar{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;background:linear-gradient(135deg,var(--a1),var(--a3))}
.foot{color:#7b7fa0;font-size:.82rem;text-align:center;padding:1.6rem 0 .4rem}

/* Widgets */
.stButton>button{border-radius:12px;font-weight:700;min-height:2.8rem;transition:transform .15s ease}
.stButton>button:hover{transform:translateY(-1px)}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,var(--a1),var(--a3));border:0;color:white}
[data-testid="stFileUploader"] section{border:2px dashed var(--a1);border-radius:18px;background:#fff}
@media(max-width:700px){.block-container{padding:1rem;margin-top:.4rem}.hero{padding:1.3rem}}
</style>
"""
for token, value in {"__A1__": A1, "__A2__": A2, "__A3__": A3, "__BEAT__": str(BEAT), "__BG__": BG_URL}.items():
    CSS = CSS.replace(token, value)
st.markdown(CSS, unsafe_allow_html=True)


def stretch(fn, *args, **kwargs):
    """Full-width widget that works on both old (use_container_width) and new (width='stretch') Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def metric(label: str, value: str, sub: str, cls: str = "") -> None:
    html(f"<div class='metric {cls}'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='subtle'>{sub}</div></div>")


# ----------------------------------------------------------------------------
# Hero (shown on every page)
# ----------------------------------------------------------------------------
ECG_PATH = (
    "M0 40 H70 L80 40 L90 28 L100 40 H130 L140 46 L152 4 L164 66 L176 40 H220 "
    "Q236 22 252 40 H330 L340 40 L350 28 L360 40 H390 L400 46 L412 4 L424 66 L436 40 H480 "
    "Q496 22 512 40 H600"
)
HERO_TEXT = {
    PAGES[0]: ("Understand the signal.<br>Explore the pattern.", "Upload a supported 12-lead ECG image and run the project's trained preprocessing and classification pipeline, with every step visible."),
    PAGES[1]: ("About the project", "ECG Image-Based Cardiac Condition Classification Using ML: from a photographed or scanned ECG page to a four-class prediction, using a lightweight classical-ML pipeline."),
    PAGES[2]: ("Limits &amp; responsible use", "What this prototype can and cannot do, and why its output must never replace a clinician."),
}
h1, sub = HERO_TEXT[page]
html(
    f"<div class='hero'><span class='chip'>Cardiovascular AI · Research prototype</span><h1>{h1}</h1><p>{sub}</p>"
    "<div class='chips'><span class='chip'>12-lead images</span><span class='chip'>PCA + SVM</span><span class='chip'>4 cardiac classes</span></div>"
    f"<svg class='ecg' viewBox='0 0 600 70' preserveAspectRatio='none'><path pathLength='1000' d='{ECG_PATH}'/></svg></div>"
)

# ============================================================================
# PAGE: ABOUT THE PROJECT  (content from the Review-II deck)
# ============================================================================
if page == PAGES[1]:
    t_over, t_data, t_pipe, t_res, t_team = st.tabs(["Overview", "Dataset", "Pipeline", "Results", "Team & references"])

    # ---- Overview -----------------------------------------------------------
    with t_over:
        html(
            "<div class='panel'><div class='section-title'>Problem statement</div>"
            "<p class='subtle' style='font-size:1rem'>ECG reports hold important information about cardiac activity, but extracting it from "
            "ECG <i>images</i> is hard because of image noise, background grids and irrelevant pixels. This project processes the image "
            "automatically, extracts the waveform of every lead, turns it into numbers, and classifies it.</p></div>"
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric("Dataset", "928", "ECG images, 4 classes")
        with c2: metric("Features", "3060 → 190", "12 leads × 255 samples, then PCA", "m2")
        with c3: metric("Best model", "SVM", "Chosen from 5 classifiers", "m3")
        with c4: metric("Test accuracy", "93.01%", "On a 186-image test set", "m4")

        html("<div class='section-title'>Objectives</div>")
        OBJ = [
            ("ECG image processing", "Process ECG images and prepare them for analysis."),
            ("12-lead extraction", "Divide the ECG image into individual leads."),
            ("Waveform features", "Extract meaningful waveform information from each lead."),
            ("Dimensionality reduction", "Reduce redundant features using PCA."),
            ("ML classification", "Train and evaluate multiple ML classifiers."),
            ("Application", "Integrate the trained model into a web application."),
        ]
        cols = st.columns(3)
        for i, (t, d) in enumerate(OBJ):
            with cols[i % 3]:
                html(f"<div class='obj'><span class='n'>{i + 1}</span><b>{t}</b><span class='subtle'>{d}</span></div>")
                st.write("")

        html("<div class='section-title'>Why this approach? The research gap</div>")
        g1, g2 = st.columns([1.2, 1])
        with g1:
            html(
                "<div class='panel'><ul><li>Limited dataset size and diversity in existing work</li>"
                "<li>Lack of independent / external validation</li><li>Deep learning is computationally expensive</li>"
                "<li>Complex preprocessing in some methods; the grid and background interfere with features</li>"
                "<li>Limited explainability in some models</li></ul></div>"
            )
        with g2:
            html(
                f"<div class='stage' style='border-left:8px solid {A1}'><h3>Our direction</h3><p class='subtle' style='font-size:1rem'>"
                "An efficient, lightweight and ECG-signal-focused classification system: classical ML on compact waveform features, "
                "instead of an end-to-end CNN.</p></div>"
            )
        html("<br><div class='section-title'>Key contributions</div>")
        st.markdown(
            "- 12-lead ECG processing from images\n- Waveform-based feature extraction\n- 3060 → 190 feature reduction with PCA\n"
            "- Classical ML classification\n- FastAPI + React deployment (this Streamlit app is the interactive prototype)"
        )

    # ---- Dataset ------------------------------------------------------------
    with t_data:
        html("<div class='section-title'>Four-class ECG cardiac-condition dataset</div><p class='subtle'>928 ECG images · 2213 × 1572 px · 12-lead information. Hover or pick a class below.</p>")
        ds = pd.DataFrame(
            [
                {"Class": "NORMAL", "Images": 284, "Meaning": "Normal ECG"},
                {"Class": "MI", "Images": 239, "Meaning": "Myocardial infarction"},
                {"Class": "AHB", "Images": 233, "Meaning": "Abnormal heartbeat"},
                {"Class": "PMI", "Images": 172, "Meaning": "History of MI"},
            ]
        )
        d1, d2 = st.columns([1, 1], gap="large")
        with d1:
            donut = (
                alt.Chart(ds)
                .mark_arc(innerRadius=70, outerRadius=130, cornerRadius=6)
                .encode(
                    theta="Images:Q",
                    color=alt.Color("Class:N", scale=alt.Scale(domain=list(ds["Class"]), range=[A1, A3, A2, "#12b5a6"]), legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Class", "Meaning", "Images"],
                )
                .properties(height=320)
            )
            stretch(st.altair_chart, donut)
        with d2:
            pick = st.radio("Class", list(ds["Class"]), horizontal=True, label_visibility="collapsed")
            row = ds[ds["Class"] == pick].iloc[0]
            share = row["Images"] / ds["Images"].sum() * 100
            html(
                f"<div class='stage' style='border-left:8px solid {A1}'><h3>{row['Class']} · {row['Meaning']}</h3>"
                f"<p class='subtle' style='font-size:1rem'>{int(row['Images'])} images, {share:.1f}% of the dataset.</p></div>"
            )
            st.write("")
            stretch(st.dataframe, ds, hide_index=True)
            st.caption("Dataset: Kaggle “ECG Image” by kanishkarathore1604.")

    # ---- Pipeline -----------------------------------------------------------
    with t_pipe:
        html("<div class='section-title'>From ECG image to class label</div><p class='subtle'>Pick a stage to see what happens to the image.</p>")
        STAGES = [
            ("ECG image", "Input", "The user uploads a full-page 12-lead ECG image (JPG, JPEG or PNG)."),
            ("Preprocessing", "Gray · Resize · Gauss · Otsu", "Convert to grayscale (intensity), resize to a standard size, smooth with a Gaussian filter to reduce noise, then apply Otsu thresholding to separate the waveform from the background. The result is a binary waveform."),
            ("12-lead division", "Per-lead crop", "The page is divided into the 12 individual leads (I, II, III, aVR, aVL, aVF, V1–V6)."),
            ("Waveform extraction", "Contour-based", "For each lead: grayscale, Gaussian filtering, thresholding, contour detection, then the largest relevant ECG contour is identified."),
            ("Feature extraction", "255 samples × 12 leads", "The contour is resized to a fixed representation of 255 normalised samples per lead. Twelve leads give 12 × 255 = 3060 features per ECG."),
            ("Min-Max scaling", "Normalise features", "Features are scaled to a common range before dimensionality reduction."),
            ("PCA", "3060 → 190 components", "Removes redundant information, lowers computational complexity, and keeps the important variance."),
            ("Classification", "SVM", "Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest and SVM were compared with 5-fold cross-validation. The final SVM predicts the class."),
        ]
        names = [f"{i + 1}. {s[0]}" for i, s in enumerate(STAGES)]
        idx = st.select_slider("Stage", options=list(range(len(STAGES))), format_func=lambda i: names[i], label_visibility="collapsed")
        s_title, s_sub, s_desc = STAGES[idx]
        st.progress((idx + 1) / len(STAGES))
        html(f"<div class='stage' style='border-left:8px solid {PALETTE[idx % len(PALETTE)]}'><h3>{names[idx]}</h3><b style='color:{A1}'>{s_sub}</b><p class='subtle' style='font-size:1rem;margin-top:.4rem'>{s_desc}</p></div>")
        st.write("")
        html("<div class='section-title'>Classifiers compared</div>")
        cl = st.columns(5)
        for c, (n, d) in zip(cl, [("Logistic Regression", "Linear baseline"), ("K-Nearest Neighbors", "Distance-based"), ("Decision Tree", "Rule-based splits"), ("Random Forest", "Tree ensemble"), ("SVM", "Margin-based · final model")]):
            with c:
                html(f"<div class='obj' style='min-height:96px'><b>{n}</b><span class='subtle'>{d}</span></div>")
        html("<br><div class='section-title'>Deployed architecture</div>")
        a1, a2, a3 = st.columns(3)
        with a1: html("<div class='panel'><b>Frontend · React</b><ul><li>ECG image upload</li><li>Result view</li></ul></div>")
        with a2: html("<div class='panel'><b>Backend · FastAPI</b><ul><li>Image preprocessing</li><li>Lead extraction</li><li>Feature extraction</li></ul></div>")
        with a3: html("<div class='panel'><b>ML · PCA + SVM</b><ul><li>PCA transformation</li><li>SVM prediction</li></ul></div>")

    # ---- Results ------------------------------------------------------------
    with t_res:
        html("<div class='section-title'>Final SVM performance</div><p class='subtle'>Test-set results from the project evaluation (186 images).</p>")
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric("Accuracy", "93.01%", "Overall", "")
        with m2: metric("Precision", "93.00%", "Macro", "m2")
        with m3: metric("Recall", "92.46%", "Macro", "m3")
        with m4: metric("Macro F1", "92.67%", "Macro", "m4")

        CLASSES = ["NORMAL", "AHB", "MI", "PMI"]
        report = pd.DataFrame(
            {"Class": CLASSES, "Precision": [0.90, 0.91, 1.00, 0.91], "Recall": [0.96, 0.85, 1.00, 0.88], "F1-score": [0.93, 0.88, 1.00, 0.90], "Support": [57, 47, 48, 34]}
        )
        cm = np.array([[55, 2, 0, 0], [4, 40, 0, 3], [0, 0, 48, 0], [2, 2, 0, 30]])

        r1, r2 = st.columns([1, 1.1], gap="large")
        with r1:
            html("<div class='section-title'>Per-class report</div>")
            metric_pick = st.radio("Metric", ["Precision", "Recall", "F1-score"], horizontal=True, label_visibility="collapsed")
            bars = (
                alt.Chart(report)
                .mark_bar(cornerRadiusEnd=8, size=28)
                .encode(
                    x=alt.X(f"{metric_pick}:Q", scale=alt.Scale(domain=[0, 1.05])),
                    y=alt.Y("Class:N", sort=CLASSES, title=None),
                    color=alt.Color("Class:N", scale=alt.Scale(domain=CLASSES, range=[A1, A3, A2, "#12b5a6"]), legend=None),
                    tooltip=["Class", "Precision", "Recall", "F1-score", "Support"],
                )
                .properties(height=220)
            )
            stretch(st.altair_chart, bars)
            stretch(st.dataframe, report, hide_index=True)
        with r2:
            html("<div class='section-title'>Confusion matrix</div>")
            cm_df = pd.DataFrame([{"Actual": CLASSES[i], "Predicted": CLASSES[j], "Count": int(cm[i, j])} for i in range(4) for j in range(4)])
            heat = alt.Chart(cm_df).mark_rect(cornerRadius=6).encode(
                x=alt.X("Predicted:N", sort=CLASSES, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Actual:N", sort=CLASSES),
                color=alt.Color("Count:Q", scale=alt.Scale(range=["#f4f1ff", A3]), legend=None),
                tooltip=["Actual", "Predicted", "Count"],
            )
            text = alt.Chart(cm_df).mark_text(fontSize=16, fontWeight="bold").encode(
                x=alt.X("Predicted:N", sort=CLASSES), y=alt.Y("Actual:N", sort=CLASSES), text="Count:Q",
                color=alt.condition(alt.datum.Count > 20, alt.value("white"), alt.value("#1b1d3a")),
            )
            stretch(st.altair_chart, (heat + text).properties(height=300))
            st.caption("Rows are the true class, columns the predicted class. MI is separated perfectly; most errors are AHB ↔ NORMAL/PMI.")
        html(
            "<div class='note'><b>Read these numbers carefully:</b> they come from one project dataset and one evaluation protocol, so they should not be compared "
            "directly with accuracies in published papers. External, patient-level validation is still required.</div>"
        )

    # ---- Team & references ----------------------------------------------------
    with t_team:
        html("<div class='section-title'>Team</div><p class='subtle'>· Under the guidance of <b>Dr. Chattu Kannaiah Sir</b></p>")
        team = ["P. Nagarjuna", "P. Nandini", "R. Susan", "P. Harshitha Sowmya"]
        cols = st.columns(4)
        for c, name in zip(cols, team):
            with c:
                initials = "".join(w[0] for w in name.replace(".", " ").split()[:2])
                html(f"<div class='person'><div class='avatar'>{initials}</div><b>{name}</b></div>")
        html("<br><div class='section-title'>References</div>")
        st.markdown(
            "1. Mhamdi, L., Dammak, O., Cottin, F., & Ben Dhaou, I. (2022). Artificial Intelligence for Cardiac Diseases Diagnosis and Prediction Using ECG Images on Embedded Systems. *Biomedicines*, 10, 2013.\n"
            "2. Uchiyama, R., Okada, Y., Kakizaki, R., & Tomioka, S. (2022). End-to-End Convolutional Neural Network Model to Detect and Localize Myocardial Infarction Using 12-Lead ECG Images without Preprocessing. *Bioengineering*, 9, 430.\n"
            "3. Ashtaiwi, A., Khalifa, T., & Alirr, O. (2024). Enhancing Heart Disease Diagnosis Through ECG Image Vectorization-Based Classification. *Heliyon*, 10, e37574.\n"
            "4. Kailasanathan, N. K., Ezhilarasan, G., Selvarajan, S., Dhanaraj, R. K., Pamucar, D., & Shankar, N. (2025). Heart disease prediction with a feature-sensitized interpretable framework for the Internet of Medical Things sensors. *Frontiers in Digital Health*, 7. doi: 10.3389/fdgth.2025.1612915\n"
            "5. Chang, V., Bhavani, V. R., Xu, A. Q., & Hossain, M. A. (2022). An artificial intelligence model for heart disease detection using machine learning algorithms. *Healthcare Analytics*, 2, 100016. doi: 10.1016/j.health.2022.100016\n"
            "6. Dataset: https://www.kaggle.com/datasets/kanishkarathore1604/ecg-image"
        )

# ============================================================================
# PAGE: LIMITS
# ============================================================================
elif page == PAGES[2]:
    html(
        "<div class='panel'><div class='section-title'>Research prototype, not a medical device</div>"
        "<p class='subtle' style='font-size:1rem'>This project classifies a particular ECG image representation using bundled PCA and classifier artifacts. "
        "It is not a diagnosis, cannot rule out heart disease, and must not be used to make treatment decisions.</p></div>"
    )
    l1, l2 = st.columns(2, gap="large")
    with l1:
        html(
            "<div class='panel'><div class='section-title'>Known limitations</div><ul>"
            "<li>Limited dataset size and diversity</li><li>ECG image quality affects waveform extraction</li>"
            "<li>Contour extraction may fail on noisy or unclear images</li><li>Trained only on four predefined classes</li>"
            "<li>External patient-level validation is required</li></ul></div>"
        )
        html(
            "<div class='panel'><div class='section-title'>Compatibility</div><p class='subtle'>The lead crop coordinates assume a full-page ECG image in the project's "
            "expected layout. Screenshots, single-lead strips, angled photos or other layouts may not work.</p>"
            "<div class='section-title'>Model files</div><p class='subtle'>Pickled scikit-learn models are version-sensitive. If loading fails, install a compatible "
            "Python and scikit-learn version or re-export the model. A loading failure is not an ECG result.</p></div>"
        )
    with l2:
        html("<div class='section-title'>Future scope</div>")
        FUT = [
            "Larger, more diverse ECG datasets", "Patient-level external validation", "Improved waveform extraction",
            "Explainable AI / Grad-CAM", "Comparison with CNNs and Transformers", "Real-time ECG acquisition",
            "Mobile / wearable deployment", "Clinical validation with professionals", "Integration of patient information",
        ]
        done = []
        for i, f in enumerate(FUT):
            if st.checkbox(f, key=f"fut{i}", value=False):
                done.append(f)
        st.caption("Roadmap: data → methods → deployment → clinical validation. (Tick items to explore; nothing is saved.)")
    html("<div class='note'>If you have symptoms or concerns, consult a qualified healthcare professional. Seek emergency care for possible heart-attack symptoms.</div>")

# ============================================================================
# PAGE: ECG ANALYSIS
# ============================================================================
else:
    c1, c2, c3 = st.columns(3)
    with c1: metric("Input", "Image", "PNG, JPG, TIFF and more")
    with c2: metric("Expected layout", "12-lead", "Supported full-page template", "m2")
    with c3: metric("Output", "Class label", "Confidence only if the model supports it", "m3")

    html("<div class='section-title'>Start a new analysis</div><p class='subtle'>Choose a clear, upright ECG page. The image is validated before the model runs.</p>")
    _missing = [n for n in ("Heart_Disease_Prediction_using_ECG.pkl", "PCA_ECG.pkl") if not (BASE_DIR / n).is_file()]
    if _missing:
        st.warning("Model file(s) not found next to app.py: " + ", ".join(_missing) + ". The analysis cannot run until they are added.")
    uploaded = st.file_uploader(
        "Upload ECG image",
        type=[x.lstrip(".") for x in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=False,
        help="Maximum 15 MB. Use a full-page ECG image in the expected 12-lead layout.",
    )
    upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded else None

    if uploaded:
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            html("<div class='section-title'>Image preview</div><p class='subtle'>Confirm the image is upright, complete and readable.</p>")
            stretch(st.image, uploaded)
            st.caption(f"File: {uploaded.name} · {uploaded.size / 1024:.1f} KB")
        with right:
            html(
                "<div class='section-title'>Validation checklist</div><p class='subtle'>These checks run before the model.</p>"
                "<span class='check'>✓ Supported extension</span><span class='check'>✓ Non-empty, within size limit</span>"
                "<span class='check'>✓ Readable image</span><span class='check'>✓ RGB, RGBA or grayscale</span>"
                "<span class='check'>✓ Page layout checked while processing</span><br><br>"
            )
            run = stretch(st.button, "Run ECG analysis", type="primary")
            clear = stretch(st.button, "Clear previous result")
            if clear:
                for key in ("ecg_result", "ecg_filename", "ecg_upload_hash"):
                    st.session_state.pop(key, None)
                st.rerun()

        if run:
            with st.spinner("Validating image and running the trained model pipeline…"):
                try:
                    result = get_engine().predict_image(uploaded)
                    st.session_state["ecg_result"] = result
                    st.session_state["ecg_filename"] = uploaded.name
                    st.session_state["ecg_upload_hash"] = upload_hash
                    if hasattr(st, "toast"):
                        st.toast("Analysis complete", icon="✅")
                except ECGInputError as exc:
                    st.error(f"Input validation failed: {exc}")
                except Exception as exc:
                    st.error("The model pipeline could not complete. No prediction has been generated.")
                    st.exception(exc)

        result = st.session_state.get("ecg_result") if st.session_state.get("ecg_upload_hash") == upload_hash else None

        if result is not None:
            st.markdown("---")
            html("<div class='section-title'>Analysis results</div><p class='subtle'>Output from the bundled classifier for the most recent successful upload.</p>")
            fname = st.session_state.get("ecg_filename", "uploaded ECG")
            html(
                f"<div class='result'><div class='result-label'>MODEL CLASSIFICATION · {fname}</div>"
                f"<div class='result-value'>{result.label}</div><span class='rchip'>Class ID: {result.class_id}</span></div>"
            )

            if result.probabilities:
                html("<br><div class='section-title'>Model-reported class probabilities</div>")
                prob_df = (
                    pd.DataFrame([{"Class": str(k), "Probability": float(v)} for k, v in result.probabilities.items()])
                    .sort_values("Probability", ascending=False)
                    .reset_index(drop=True)
                )
                chart = (
                    alt.Chart(prob_df)
                    .mark_bar(cornerRadiusEnd=8, size=26)
                    .encode(
                        x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, max(1.0, prob_df["Probability"].max())])),
                        y=alt.Y("Class:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                        color=alt.Color("Class:N", scale=alt.Scale(range=PALETTE), legend=None),
                        tooltip=["Class", alt.Tooltip("Probability:Q", format=".2%")],
                    )
                    .properties(height=max(120, 46 * len(prob_df)))
                )
                stretch(st.altair_chart, chart)
                st.caption("Shown only because the loaded classifier exposes predict_proba. These values are not clinically calibrated unless separately validated.")
            else:
                st.info("This classifier does not expose class probabilities, so no confidence score is displayed.")

            html("<br><div class='section-title'>Explore each lead</div><p class='subtle'>Slide through the extracted lead regions. The waveform can be zoomed and panned.</p>")
            n_leads = min(len(result.leads), result.lead_signals.shape[0], len(LEAD_NAMES))
            li = st.select_slider("Lead", options=list(range(n_leads)), value=0, format_func=lambda i: f"Lead {LEAD_NAMES[i]}")
            lead_title = f"Lead {LEAD_NAMES[li]}"
            la, lb = st.columns([1, 1.4], gap="large")
            with la:
                stretch(st.image, result.leads[li], clamp=True, caption=lead_title)
            with lb:
                try:
                    sig = np.asarray(result.lead_signals[li], dtype=float).ravel()
                    sig_df = pd.DataFrame({"Sample": np.arange(len(sig)), "Amplitude": sig})
                    line = (
                        alt.Chart(sig_df)
                        .mark_line(color=A1, strokeWidth=2.2, interpolate="monotone")
                        .encode(x="Sample:Q", y=alt.Y("Amplitude:Q", scale=alt.Scale(zero=False, reverse=True), title="Amplitude (normalized)"), tooltip=["Sample", alt.Tooltip("Amplitude:Q", format=".3f")])
                        .properties(height=280)
                        .interactive()
                    )
                    stretch(st.altair_chart, line)
                    st.caption(f"{lead_title}: the 255-sample normalized trace the model receives, taken from the waveform contour. Axis is flipped so peaks point up.")
                except Exception:
                    st.info("A waveform could not be drawn for this lead.")

            with st.expander("Inspect preprocessing", expanded=False):
                a, b = st.columns(2)
                with a:
                    st.markdown("**Canonical grayscale image**")
                    stretch(st.image, result.grayscale, clamp=True)
                with b:
                    st.markdown("**All extracted lead regions**")
                    fig = Figure(figsize=(10, 7))
                    axes = fig.subplots(4, 3)
                    for i, ax in enumerate(axes.flat):
                        if i < n_leads:
                            ax.imshow(result.leads[i], cmap="gray" if result.leads[i].ndim == 2 else None)
                            ax.set_title(f"Lead {LEAD_NAMES[i]}", fontsize=9)
                        ax.axis("off")
                    fig.tight_layout()
                    st.pyplot(fig)
                st.write(f"Feature vector shape: `{result.lead_signals.shape[0]} leads × {result.lead_signals.shape[1]} samples`")
                st.write(f"PCA output shape: `{result.reduced_features.shape}`")

            report_df = pd.DataFrame([{"file": fname, "predicted_class": result.label, "class_id": result.class_id}])
            st.download_button("Download result CSV", data=report_df.to_csv(index=False).encode("utf-8"), file_name="ecg_analysis_result.csv", mime="text/csv")
            html(
                "<div class='note'><b>Important:</b> This is an experimental machine-learning output, not a medical diagnosis. "
                "If you have symptoms or concerns, consult a qualified healthcare professional. Seek emergency care for possible heart-attack symptoms.</div>"
            )
    else:
        html(
            "<div class='panel'><div class='section-title'>Waiting for an ECG image</div>"
            "<p class='subtle'>Upload a supported file to see the preview, validation and analysis. No prediction is generated until you run the analysis. "
            "Want to know how it works? Open <b>About the Project</b> in the sidebar.</p></div>"
        )

html("<div class='foot'>CardioLens · ECG image classification research prototype · Not for clinical decision-making</div>")