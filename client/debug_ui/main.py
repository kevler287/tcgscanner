import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from shared.tcg_config import TCGConfig

DEBUG_DIR = Path("./debug")
YUGIOH_CONFIG = TCGConfig.load("shared/yugioh.json")
font = ImageFont.load_default(size=25)

st.set_page_config(layout="wide")

def list_trace_ids():
    return sorted(p.name for p in DEBUG_DIR.iterdir() if p.is_dir())


def load_image(trace_dir, name):
    path = trace_dir / f"{name}.jpg"
    return Image.open(path) if path.exists() else None


def load_json(trace_dir, name):
    path = trace_dir / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def draw_pts_overlay(raw_img, pts):
    img = raw_img.copy()
    draw = ImageDraw.Draw(img)
    if pts:
        poly = [tuple(p) for p in pts]
        draw.polygon(poly, outline="red", width=6)
    return img

def resize_to_height(img, target_height=250):
    if img is None:
        return None
    ratio = target_height / img.height
    new_width = int(img.width * ratio)
    return img.resize((new_width, target_height))

def draw_crop_areas(warped_img, config):
    img = warped_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for name, (top_left, bottom_right) in config.ocr_fields.items():
        x1, y1 = top_left[0] * w, top_left[1] * h
        x2, y2 = bottom_right[0] * w, bottom_right[1] * h
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, y1 - font.size), name, fill="lime", font=font)

    for name, (top_left, bottom_right) in config.edition_areas.items():
        x1, y1 = top_left[0] * w, top_left[1] * h
        x2, y2 = bottom_right[0] * w, bottom_right[1] * h
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=3)
        draw.text((x1, y1 - font.size), name, fill="cyan", font=font)

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

raw_col, warped_col, results_col = st.columns([1, 1, 1.5])

with raw_col:
    st.caption("Raw frame + segmentation box")
    if raw:
        img = draw_pts_overlay(raw, pts_data["pts"] if pts_data else None)
        st.image(resize_to_height(img, 650))

with warped_col:
    st.caption("Warped")
    if warped:
        warped = draw_crop_areas(warped_img=warped, config=YUGIOH_CONFIG)
        st.image(resize_to_height(warped, 650))
    else:
        st.info("Kein warped image")

with results_col:
    st.header("OCR + Confidence")
    for i, field in enumerate(YUGIOH_CONFIG.ocr_fields.keys()):
        img_col, text_col, conf_col = st.columns(3)
        with img_col:
            st.caption(f"{field} crop")
            crop = load_image(trace_dir, f"03_ocr_crop_{i:02d}")
            if crop:
                st.image(crop)
        with text_col:
            st.caption("Text")
            if ocr_result:
                st.text(ocr_result["text"][field][0][0])
        with conf_col:
            st.caption("Confidence")
            if ocr_result:
                st.text(ocr_result["text"][field][0][1])

    st.header("First Edition Probability")
    for i, area in enumerate(YUGIOH_CONFIG.edition_areas.keys()):
        img_col, text_col = st.columns(2)
        with img_col:
            st.caption(f"{area} crop")
            crop = load_image(trace_dir, f"04_edition_crop_{i:02d}")
            if crop:
                st.image(crop)
        with text_col:
            st.caption("Probability")
            if edition_result:
                prob = edition_result["editions"][f"first_ed_{i}"]
                st.text(f"{prob:.0%}")

st.divider()