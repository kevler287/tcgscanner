from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Body

from boot import price_setup
import boot.product_setup as product_setup
from service import catalog_query

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once before the app starts accepting requests.
    catalog_df = product_setup.run_job()
    price_df = price_setup.run_job()
    app.state.catalog = catalog_df
    app.state.price = price_df
    yield
    # Place shutdown/cleanup logic here if needed.


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def identify_ygo_card(request: Request, set_code: str = Body(..., embed=True)):
    return catalog_query.identify_product(request=request, setcode=set_code)

@app.get("/price/{product_id}")
def get_price(request: Request, product_id: str):
    return catalog_query.get_price_for_product(request=request, product_id=product_id)