import json
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw

DEBUG_DIR = Path("./debug")

st.set_page_config(layout="wide")


def list_trace_ids():
    return sorted(p.name for p in DEBUG_DIR.iterdir() if p.is_dir())


def load_image(trace_dir, name):
    path = trace_dir / f"{name}.png"
    return Image.open(path) if path.exists() else None


def load_json(trace_dir, name):
    path = trace_dir / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def draw_pts_overlay(raw_img, pts):
    img = raw_img.copy()
    draw = ImageDraw.Draw(img)
    if pts:
        poly = [tuple(p) for p in pts]
        draw.polygon(poly, outline="red", width=4)
    return img


trace_ids = list_trace_ids()

if "idx" not in st.session_state:
    st.session_state.idx = 0

col_prev, col_mid, col_next = st.columns([1, 4, 1])
with col_prev:
    if st.button("⬅ Prev") and st.session_state.idx > 0:
        st.session_state.idx -= 1
with col_next:
    if st.button("Next ➡") and st.session_state.idx < len(trace_ids) - 1:
        st.session_state.idx += 1
with col_mid:
    st.markdown(f"**Trace {st.session_state.idx + 1} / {len(trace_ids)}**: `{trace_ids[st.session_state.idx]}`")

trace_id = trace_ids[st.session_state.idx]
trace_dir = DEBUG_DIR / trace_id

raw = load_image(trace_dir, "00_raw")
pts_data = load_json(trace_dir, "01_sorted_pts")
warped = load_image(trace_dir, "02_warped")
ocr_result = load_json(trace_dir, "03_ocr_result")
edition_result = load_json(trace_dir, "04_edition_result")

# --- 3 Hauptspalten: raw+mask | warped | results ---
raw_col, warped_col, results_col = st.columns([1, 1, 1.5])

with raw_col:
    st.caption("Raw + sorted_pts")
    if raw:
        st.image(draw_pts_overlay(raw, pts_data["pts"] if pts_data else None))

with warped_col:
    st.caption("Warped")
    if warped:
        st.image(warped)
    else:
        st.info("Kein warped image")

with results_col:
    st.markdown("**OCR**")
    for i in range(2):
        img_col, res_col = st.columns(2)
        with img_col:
            crop = load_image(trace_dir, f"03_ocr_crop_{i:02d}")
            if crop:
                st.image(crop)
        with res_col:
            if ocr_result:
                st.json(ocr_result)

    st.markdown("**Edition**")
    for i in range(2):
        img_col, res_col = st.columns(2)
        with img_col:
            crop = load_image(trace_dir, f"04_edition_crop_{i:02d}")
            if crop:
                st.image(crop)
        with res_col:
            if edition_result:
                st.json(edition_result)

st.divider()

# --- Zweite Row: leer für später ---
st.container(height=200, border=True)