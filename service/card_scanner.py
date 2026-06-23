from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Response
from contextlib import asynccontextmanager
from shared.tcg_config import TCGConfig
from yolo.card_segmentation import CardSegmentor
from ocr.text_extraction import TextExtractor
import cv2
import numpy as np
import traceback

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    default_config = TCGConfig.load("shared/ygo_config.json")
    app.state.segmentor = CardSegmentor(model_path="yolo/v1.pt", tcg_config=default_config)
    app.state.ocr = TextExtractor(use_gpu=True, tcg_config=default_config)
    app.state.ready = True
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health/ready")
async def readiness(request: Request):
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Models loading")
    return {"status": "ready"}

@app.post("/configure")
async def configure(request: Request):
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Models loading")
    try:
        cfg_json = await request.json()
        new_config = TCGConfig(**cfg_json)
        request.app.state.segmentor.tcg_config = new_config
        request.app.state.ocr.tcg_config = new_config
    except:
        raise HTTPException(detail={"error": traceback.format_exc()}, status_code=503)


@app.post("/scan")
async def scan(request: Request, file: UploadFile = File(...)):
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Models loading")
    
    # Load image from multipart
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    warped, sorted_pts = request.app.state.segmentor.segment_and_warp(img)
    
    if warped is None:
        return {
            "text": None,
            "pts": sorted_pts.tolist() if sorted_pts is not None else None
        }
    
    text = request.app.state.ocr.extract(card_image=warped)
    
    return {
        "text": text,
        "pts": sorted_pts.tolist() if sorted_pts is not None else None
    }