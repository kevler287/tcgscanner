from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Response
from contextlib import asynccontextmanager
from ml.edition_checker import EditionClassifier
from shared.tcg_config import TCGConfig
from ml.card_segmentation import CardSegmentor
from ml.text_extraction import TextExtractor
from handler import scan_handler
import cv2
import numpy as np
import traceback

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    default_config = TCGConfig.load("shared/yugioh.json")
    app.state.segmentor = CardSegmentor(model_path="ml/v1.pt", tcg_config=default_config)
    app.state.ocr = TextExtractor(use_gpu=True, tcg_config=default_config)
    app.state.ed_check = EditionClassifier(model_path="ml/models_ed_check_1.1.3.pt", tcg_config=default_config)
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
    
    return scan_handler.handle_frame(request=request, img=img)