from fastapi import Request
import numpy as np

from helper import volume_writer


def handle_frame(request: Request, img: np.ndarray):
    warped, sorted_pts = request.app.state.segmentor.segment_and_warp(img)
    if sorted_pts is None:
        return {}
    
    trace_id = volume_writer.new_trace_id()
    if request.app.state.debug:
        volume_writer.save_frame(trace_id, "00_raw", img)
        volume_writer.save_json(trace_id, "01_sorted_pts", {"pts": sorted_pts.tolist()}) 
 
    if warped is None:
        return {}
    
    text, ocr_crops = request.app.state.ocr.extract(card_image=warped)
    edition_dets, ed_crops = request.app.state.ed_check.predict(frame=warped)

    set_crop = ocr_crops.get("set_code")
    if set_crop is not None and request.app.state.frame_counter % 10 == 0:
        volume_writer.save_crop(crop=set_crop)
    if request.app.state.debug:
        volume_writer.save_frame(trace_id, "02_warped", warped)
        volume_writer.save_crops(trace_id, "03_ocr_crop", ocr_crops.values())
        volume_writer.save_json(trace_id, "03_ocr_result", {"text": text})
        volume_writer.save_crops(trace_id, "04_edition_crop", ed_crops)
        volume_writer.save_json(trace_id, "04_edition_result", {"editions": edition_dets})

    request.app.state.frame_counter += 1
    
    return {
        "text": text,
        "editions": edition_dets
    }