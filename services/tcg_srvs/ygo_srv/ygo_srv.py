from contextlib import asynccontextmanager

from fastapi import FastAPI

import boot

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once before the app starts accepting requests.
    boot.run_boot_job()
    yield
    # Place shutdown/cleanup logic here if needed.


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}