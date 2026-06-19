from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from service.card_configs.card_config import CardConfig
from service.yolo.card_segmentation import CardSegmentor
from service.ocr.text_extraction import TextExtractor

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    ygo_config = CardConfig.load("card_configs/ygo_card.json")
    app.state.segmentor = CardSegmentor(model_path="yolo/best.pt", ygo_config=ygo_config)
    app.state.ocr = TextExtractor(use_gpu=True, ygo_config=ygo_config)
    app.state.ready = True
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/scan")
async def scan(request: Request):
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Models loading")
    