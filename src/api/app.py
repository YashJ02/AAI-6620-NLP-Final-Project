"""FastAPI app entrypoint."""

from fastapi import FastAPI

from src.api.routes import router as pipeline_router

app = FastAPI(title="Blood Report Analyzer API")
app.include_router(pipeline_router)


@app.get('/health')
def health() -> dict:
    return {"status": "ok"}

