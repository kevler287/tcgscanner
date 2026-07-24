from fastapi import Request
import numpy as np


def handle_frame(request: Request, img: np.ndarray):
    warped, sorted_pts = request.app.state.segmentor.segment_and_warp(img)
    
    if warped is None:
        return {
            "text": None,
            "pts": sorted_pts.tolist() if sorted_pts is not None else None
        }
    
    text = request.app.state.ocr.extract(card_image=warped)
    edition_dets = request.app.state.ed_check.predict(frame=warped)
    
    return {
        "text": text,
        "pts": sorted_pts.tolist() if sorted_pts is not None else None,
        "editions": edition_dets
    }