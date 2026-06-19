from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from service.yolo.card_segmentation import CardSegmentor
from service.ocr.text_extraction import TextExtractor

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.segmentor = CardSegmentor("yolo/best.pt")
    app.state.ocr = TextExtractor(use_gpu=True)
    app.state.ready = True
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/scan")
async def scan(request: Request):
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Models loading")
    