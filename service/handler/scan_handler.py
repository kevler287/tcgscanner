from fastapi import Request
import numpy as np

from service import debug_storage


def handle_frame(request: Request, img: np.ndarray):
    warped, sorted_pts = request.app.state.segmentor.segment_and_warp(img)
    trace_id = debug_storage.new_trace_id()

    if request.app.state.debug:
        debug_storage.save_frame(trace_id, "00_raw", img)
        debug_storage.save_json(trace_id, "01_sorted_pts", {
            "pts": sorted_pts.tolist() if sorted_pts is not None else None
        }) 
 
    if warped is None:
        return {
            "text": None,
            "pts": sorted_pts.tolist() if sorted_pts is not None else None
        }
    
    text, ocr_crops = request.app.state.ocr.extract(card_image=warped)
    edition_dets, ed_crops = request.app.state.ed_check.predict(frame=warped)

    if request.app.state.debug:
        debug_storage.save_frame(trace_id, "02_warped", warped)
        debug_storage.save_crops(trace_id, "03_ocr_crop", ocr_crops)
        debug_storage.save_json(trace_id, "03_ocr_result", {"text": text})
        debug_storage.save_crops(trace_id, "04_edition_crop", ed_crops)
        debug_storage.save_json(trace_id, "04_edition_result", {"editions": edition_dets})
    
    return {
        "text": text,
        "pts": sorted_pts.tolist() if sorted_pts is not None else None,
        "editions": edition_dets
    }