from fastapi import Request
import numpy as np

from helper import volume_writer


def handle_frame(request: Request, img: np.ndarray):
    warped, sorted_pts = request.app.state.segmentor.segment_and_warp(img)
    if sorted_pts is None or warped is None:
        return {}
    
    text, ocr_crops = request.app.state.ocr.extract(card_image=warped)
    edition_dets, ed_crops = request.app.state.ed_check.predict(frame=warped)
    ed_prob_0 = edition_dets.get("first_ed_0")
    ed_prob_1 = edition_dets.get("first_ed_1")

    labels = {
        "set_code": text.get("set_code", ["", 0.0])[0],
        "first_edition": ed_prob_0 > 0.5 or ed_prob_1 > 0.5,
        "reviewed": False
    }

    if warped is not None and request.app.state.frame_counter % 10 == 0:
        volume_writer.collect(crop=warped, results=labels)

    request.app.state.frame_counter += 1
    
    return {
        "text": text,
        "editions": edition_dets
    }