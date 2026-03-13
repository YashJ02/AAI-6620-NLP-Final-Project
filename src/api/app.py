"""FastAPI app entrypoint."""

from fastapi import FastAPI

app = FastAPI(title="Blood Report Analyzer API")


@app.get('/health')
def health() -> dict:
    return {"status": "ok"}

