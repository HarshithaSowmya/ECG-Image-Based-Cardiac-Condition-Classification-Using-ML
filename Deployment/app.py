from __future__ import annotations
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import io
import hashlib
import pandas as pd
# import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from Ecg import ECG, ECGInputError, SUPPORTED_EXTENSIONS, LABELS

st.set_page_config(page_title="CardioLens | ECG Intelligence", page_icon="♥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#17243a;--muted:#718096;--line:#e6ebf2;--teal:#0e9f91;--bg:#f5f8fc}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 85% 0%,rgba(14,159,145,.08),transparent 32%),var(--bg);color:var(--ink)}
.block-container{padding-top:1.5rem;max-width:1440px}
[data-testid="stSidebar"]{background:#101d32;border-right:1px solid #25344a}
[data-testid="stSidebar"] *{color:#e7eef8}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#aebbd0}
.hero{padding:2rem 2.2rem;border-radius:24px;background:linear-gradient(120deg,#12233c 0%,#1d3b56 60%,#0e766f 100%);color:white;position:relative;overflow:hidden;margin-bottom:1.2rem}
.hero:after{content:'♥';position:absolute;right:4%;top:-45px;font-size:220px;line-height:1;color:rgba(255,255,255,.055);font-family:Arial}
.eyebrow{font-size:.74rem;letter-spacing:.15em;text-transform:uppercase;color:#77e1d4;font-weight:700}
.hero h1{font-family:Manrope,sans-serif;color:white;font-size:clamp(1.8rem,3vw,2.65rem);line-height:1.15;margin:.65rem 0}
.hero p{color:#d0dce9;max-width:680px;margin:0;font-size:1rem}
.pill{display:inline-block;padding:.35rem .7rem;border:1px solid rgba(255,255,255,.2);border-radius:999px;color:#e1fbf6;font-size:.75rem;margin-top:1rem}
.section-title{font-family:Manrope,sans-serif;font-size:1.2rem;font-weight:800;color:#17243a;margin:.6rem 0 .2rem}
.subtle{color:#718096;font-size:.9rem}
.metric{border:1px solid var(--line);background:white;border-radius:17px;padding:1.1rem 1.2rem;min-height:110px;box-shadow:0 5px 18px rgba(21,43,74,.035)}
.metric-label{color:#718096;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
.metric-value{font-family:Manrope,sans-serif;font-size:1.7rem;font-weight:800;color:#17243a;margin-top:.25rem}
.panel{background:white;border:1px solid var(--line);border-radius:20px;padding:1.3rem 1.4rem;box-shadow:0 5px 18px rgba(21,43,74,.035);margin-bottom:1rem}
.note{background:#eef8f7;border:1px solid #ccece7;border-radius:13px;padding:.9rem 1rem;color:#285b5a;font-size:.88rem}
.result{border-radius:18px;padding:1.25rem 1.4rem;background:linear-gradient(120deg,#e8f8f5,#f7fbff);border:1px solid #ccece7}
.result-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;color:#397a73;font-weight:700}
.result-value{font-family:Manrope,sans-serif;font-size:1.65rem;font-weight:800;color:#163c4b;margin:.3rem 0}
footer{color:#8794a6;font-size:.8rem;text-align:center;padding:2rem 0 1rem}
.stButton>button{border-radius:11px;font-weight:700;min-height:2.8rem}
[data-testid="stFileUploader"]{border:1px dashed #9bbab9;border-radius:16px;background:#f9fdfd;padding:.7rem}
@media(max-width:700px){.block-container{padding:1rem}.hero{padding:1.4rem}.panel{padding:1rem}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-size:1.35rem;font-weight:800;color:white'>♥ CardioLens</div><div style='color:#9fb0c6;font-size:.8rem;margin-top:.2rem'>ECG intelligence workspace</div>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("WORKSPACE", ["ECG Analysis", "How it works", "About & limitations"], label_visibility="visible")
    st.divider()
    st.markdown("**MODEL PIPELINE**")
    st.caption("Image validation → 12-lead extraction → signal normalization → PCA → trained classifier")
    st.markdown("<div style='font-size:.75rem;color:#91a4bd'>Local processing · Uploaded image is not sent to an external service by this app.</div>", unsafe_allow_html=True)

st.markdown("<div class='hero'><div class='eyebrow'>CARDIOVASCULAR AI · RESEARCH PROTOTYPE</div><h1>Understand the signal.<br>Explore the pattern.</h1><p>Upload a supported 12-lead ECG image to run the project's trained preprocessing and classification pipeline with transparent validation and results.</p><span class='pill'>● Image-first analysis</span> &nbsp; <span class='pill'>● Real model inference</span></div>", unsafe_allow_html=True)

if page == "How it works":
    st.markdown("<div class='section-title'>From ECG image to model output</div><p class='subtle'>The current artifacts were trained for a specific ECG page layout and feature representation.</p>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, n, title, desc in zip(cols, ["01", "02", "03", "04"], ["Validate", "Extract", "Reduce", "Classify"], ["Check image format, readability, size, and expected page geometry.", "Crop the 12 standard lead regions and extract normalized signal traces.", "Apply the bundled fitted PCA artifact to the 3,060-feature vector.", "Run the bundled classifier and show its returned class label."]):
        with col:
            st.markdown(f"<div class='panel'><div style='color:#0e9f91;font-weight:800;font-size:1.6rem'>{n}</div><div class='section-title'>{title}</div><p class='subtle'>{desc}</p></div>", unsafe_allow_html=True)
    st.info("Images that do not match the supported 12-lead page layout may be rejected. Supporting a different layout reliably requires a verified layout detector or a model trained for that representation.")
elif page == "About & limitations":
    st.markdown("<div class='section-title'>Scope and responsible use</div>", unsafe_allow_html=True)
    st.markdown("""<div class='panel'>
    <b>Research prototype, not a medical device</b><br><br>
    This project classifies a particular ECG image representation using bundled PCA and classifier artifacts. It is not a diagnosis, cannot rule out heart disease, and must not be used to make treatment decisions.<br><br>
    <b>Compatibility</b><br>
    The existing lead crop coordinates assume a full-page ECG image arranged in the project's expected layout. Arbitrary ECG screenshots, single-lead strips, photos at an angle, or different layouts may not be compatible.<br><br>
    <b>Model artifact compatibility</b><br>
    Pickled scikit-learn models are version-sensitive. If loading fails, install a compatible Python/scikit-learn environment or re-export the model in a supported environment; do not interpret that failure as an ECG result.
    </div>""", unsafe_allow_html=True)
else:
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown("<div class='metric'><div class='metric-label'>Input mode</div><div class='metric-value'>Image</div><div class='subtle'>PNG, JPG, TIFF and more</div></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric'><div class='metric-label'>Expected layout</div><div class='metric-value'>12-lead</div><div class='subtle'>Supported full-page template</div></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric'><div class='metric-label'>Output</div><div class='metric-value'>Class label</div><div class='subtle'>Confidence only if model supports it</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Start a new analysis</div><p class='subtle'>Choose a clear, upright ECG page. Your input is validated before inference.</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload ECG image", type=[x.lstrip('.') for x in sorted(SUPPORTED_EXTENSIONS)], accept_multiple_files=False, help="Maximum 15 MB. Use a full-page ECG image in the expected 12-lead layout.")
    upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded else None
    if uploaded:
        left,right = st.columns([1,1.25], gap="large")
        with left:
            st.markdown("<div class='panel'><div class='section-title'>Image preview</div><p class='subtle'>Confirm the image is upright, complete, and readable.</p>", unsafe_allow_html=True)
            st.image(uploaded, use_container_width=True)
            st.caption(f"File: {uploaded.name} · {uploaded.size / 1024:.1f} KB")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='panel'><div class='section-title'>Validation checklist</div><p class='subtle'>File-level checks run before model inference.</p>", unsafe_allow_html=True)
            st.markdown("- Supported image extension\n- Non-empty file and size limit\n- Readable image and reasonable dimensions\n- RGB, RGBA, or grayscale channels\n- Expected ECG page layout checked during processing")
            run = st.button("Run ECG analysis", type="primary", use_container_width=True)
            clear = st.button("Clear previous result", use_container_width=True)
            if clear:
                for key in ("ecg_result", "ecg_filename", "ecg_upload_hash"):
                    st.session_state.pop(key, None)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        if run:
            with st.spinner("Validating image and running the trained model pipeline…"):
                try:
                    result = ECG().predict_image(uploaded)
                    st.session_state["ecg_result"] = result
                    st.session_state["ecg_filename"] = uploaded.name
                    st.session_state["ecg_upload_hash"] = upload_hash
                    st.success("Inference completed. Review the model output below.")
                except ECGInputError as exc:
                    st.error(f"Input validation failed: {exc}")
                except Exception as exc:
                    st.error("The model pipeline could not complete. No prediction has been generated.")
                    st.exception(exc)
        result = st.session_state.get("ecg_result") if st.session_state.get("ecg_upload_hash") == upload_hash else None
        if result is not None:
            st.markdown("---")
            st.markdown("<div class='section-title'>Analysis results</div><p class='subtle'>Output generated by the bundled classifier for the most recent successful upload.</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='result'><div class='result-label'>Model classification · {st.session_state.get('ecg_filename','uploaded ECG')}</div><div class='result-value'>{result.label}</div><div style='color:#4c6873'>Class ID: {result.class_id}</div></div>", unsafe_allow_html=True)
            if result.probabilities:
                st.markdown("<div class='section-title'>Model-reported class probabilities</div>", unsafe_allow_html=True)
                prob_df = pd.DataFrame([{"Class": k, "Probability": v} for k,v in result.probabilities.items()]).sort_values("Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Class"), y="Probability")
                st.caption("These values are shown only because the loaded classifier exposes predict_proba; they are not clinically calibrated unless separately validated.")
            else:
                st.info("This classifier does not expose class probabilities; no confidence score is displayed.")
            with st.expander("Inspect preprocessing", expanded=False):
                a,b = st.columns(2)
                with a:
                    st.markdown("**Canonical grayscale image**")
                    st.image(result.grayscale, clamp=True, use_container_width=True)
                with b:
                    st.markdown("**Extracted lead regions**")
                    fig, axes = plt.subplots(4, 3, figsize=(10, 7))
                    for i, ax in enumerate(axes.flat):
                        ax.imshow(result.leads[i], cmap="gray" if result.leads[i].ndim == 2 else None)
                        ax.set_title(f"Lead region {i+1}", fontsize=9); ax.axis("off")
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                st.write(f"Feature vector shape: `{result.lead_signals.shape[0]} leads × {result.lead_signals.shape[1]} samples`")
                st.write(f"PCA output shape: `{result.reduced_features.shape}`")
            report = pd.DataFrame([{"file": st.session_state.get("ecg_filename", ""), "predicted_class": result.label, "class_id": result.class_id}])
            st.download_button("Download result CSV", data=report.to_csv(index=False).encode("utf-8"), file_name="ecg_analysis_result.csv", mime="text/csv")
            st.markdown("<div class='note'><b>Important:</b> This is an experimental machine-learning output, not a medical diagnosis. If you have symptoms or concerns, consult a qualified healthcare professional. Seek emergency care for possible heart-attack symptoms.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='panel'><div class='section-title'>Waiting for an ECG image</div><p class='subtle'>Upload a supported file to see the preview, validation, and analysis workflow. No predictions are generated until you submit an image.</p></div>", unsafe_allow_html=True)

st.markdown("<footer>CardioLens · ECG image classification research prototype · Not for clinical decision-making</footer>", unsafe_allow_html=True)
