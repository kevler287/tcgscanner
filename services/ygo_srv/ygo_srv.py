from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Body

import boot
from handler import product_handler

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once before the app starts accepting requests.
    catalog_df = boot.run_boot_job()
    app.state.catalog = catalog_df
    yield
    # Place shutdown/cleanup logic here if needed.


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/identify")
def identify_ygo_card(request: Request, set_code: str = Body(...)):
    return product_handler.identify_product(request=request, setcode=set_code)