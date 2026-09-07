"""
Everest Fleet — Exterior Damage Scorer (prototype, Streamlit version)

Runs vineetsarpal/yolov11n-car-damage (YOLOv11-Nano) against uploaded vehicle
photos, draws detected damage, and converts detections into a rough 0-10
exterior condition score using a keyword-based severity map.

Scope: EXTERIOR BODY / GLASS / LAMPS / TYRES-VISIBLE-IN-FRAME ONLY. This
model was not trained to assess interior condition or tyre tread wear —
those still need a separate check (see the Physical Condition Rubric).

The severity weights below are a starting guess. Run it once, look at the
"Raw detections" table, and adjust SEVERITY_KEYWORDS to match what the
model's real class names turn out to be.
"""

import streamlit as st
import pandas as pd
from PIL import Image
from huggingface_hub import list_repo_files, hf_hub_download
from ultralytics import YOLO

REPO_ID = "vineetsarpal/yolov11n-car-damage"

SEVERITY_KEYWORDS = {
    "crack": (5, "severe"),
    "shatter": (8, "critical"),
    "glass": (5, "severe"),
    "brokenlamp": (8, "critical"),
    "brokenheadlamp": (8, "critical"),
    "brokentaillamp": (8, "critical"),
    "lamp": (5, "severe"),
    "flattire": (8, "critical"),
    "flattyre": (8, "critical"),
    "tire": (3, "moderate"),
    "tyre": (3, "moderate"),
    "dent": (3, "moderate"),
    "scratch": (1, "minor"),
    "rust": (3, "moderate"),
    "bumper": (3, "moderate"),
    "missingpart": (5, "severe"),
}
DEFAULT_DEDUCTION = (2, "moderate")  # unrecognized class name


def severity_for(class_name: str):
    key = class_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    for kw, val in SEVERITY_KEYWORDS.items():
        if kw in key:
            return val
    return DEFAULT_DEDUCTION


@st.cache_resource(show_spinner="Downloading model weights...")
def load_model():
    files = list_repo_files(REPO_ID)
    weight_candidates = [f for f in files if f.endswith(".pt")]
    if not weight_candidates:
        raise RuntimeError(
            f"No .pt weights file found in {REPO_ID}. Files present: {files}"
        )
    weight_candidates.sort(key=lambda f: ("best" not in f.lower(), f))
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=weight_candidates[0])
    return YOLO(weights_path)


st.set_page_config(page_title="Exterior Damage Scorer", layout="wide")
st.title("Exterior Damage Scorer — prototype")
st.markdown(
    f"Upload a few exterior photos of one vehicle (front, rear, both sides, "
    f"close-ups of any damage). Runs [`{REPO_ID}`](https://huggingface.co/{REPO_ID}) "
    f"(YOLOv11-Nano) to detect exterior damage and estimate a rough condition score.\n\n"
    f"**Scope:** exterior body / glass / lamps / visible tyre only — "
    f"no interior or tyre-tread assessment."
)

uploaded_files = st.file_uploader(
    "Vehicle photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if st.button("Score this vehicle", type="primary", disabled=not uploaded_files):
    model = load_model()

    rows = []
    total_deduction = 0.0
    annotated_images = []

    with st.spinner("Running detection..."):
        for uf in uploaded_files:
            image = Image.open(uf).convert("RGB")
            result = model.predict(source=image, conf=0.25, verbose=False)[0]
            annotated_bgr = result.plot()
            annotated_images.append((uf.name, Image.fromarray(annotated_bgr[..., ::-1])))

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                deduction, band = severity_for(cls_name)
                weighted_deduction = deduction * conf
                total_deduction += weighted_deduction
                rows.append({
                    "photo": uf.name,
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "severity band": band,
                    "deduction": round(weighted_deduction, 1),
                })

    score = max(0.0, 10.0 - total_deduction)

    st.markdown(f"## Exterior condition score: {score:.1f} / 10")
    st.caption(
        f"Based on {len(rows)} detection(s) across {len(uploaded_files)} photo(s). "
        "This covers exterior body, glass, lamps and tyre visibility only — "
        "interior condition and tyre tread wear are not assessed by this model."
    )
    if not rows:
        st.info("No damage detected above the confidence threshold.")

    cols = st.columns(3)
    for i, (name, img) in enumerate(annotated_images):
        with cols[i % 3]:
            st.image(img, caption=name, use_container_width=True)

    if rows:
        st.markdown("### Raw detections (use this to calibrate SEVERITY_KEYWORDS)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
