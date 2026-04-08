"""Local cluster visualization API: serves bundled UI and JSON data from mounted file."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

_STATIC = Path(__file__).resolve().parent / "static"
_DATA = Path(os.environ.get("DATA_JSON", "/data/data.json"))

app = FastAPI(title="Campaign cluster viewer")

if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/api/data")
def get_data():
    if not _DATA.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {_DATA}. Mount run visualization dir to /data.",
        )
    with open(_DATA, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = _STATIC / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<body>Missing static/index.html</body>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
