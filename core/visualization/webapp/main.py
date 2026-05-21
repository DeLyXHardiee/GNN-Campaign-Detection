"""Local cluster visualization API: serves bundled UI and JSON data from mounted file."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from url_utils import urls_for_email_row

_STATIC = Path(__file__).resolve().parent / "static"
_DATA = Path(os.environ.get("DATA_JSON", "/data/data.json"))


def _campaign_email_count(campaign: dict) -> int:
    size = campaign.get("size")
    if size is not None:
        try:
            return int(size)
        except (TypeError, ValueError):
            pass
    return len(campaign.get("member_external_ids") or [])


def _sort_payload_campaigns(payload: dict) -> None:
    campaigns = payload.get("campaigns")
    if not isinstance(campaigns, list):
        return
    payload["campaigns"] = sorted(
        campaigns,
        key=_campaign_email_count,
        reverse=True,
    )


def _enrich_email_urls(data: dict) -> None:
    """Per-email ``urls`` from MISP attributes + body (refangs hxxp/hxxps)."""
    emails = data.get("emails")
    if not isinstance(emails, dict):
        return
    for row in emails.values():
        if isinstance(row, dict):
            row["urls"] = urls_for_email_row(row)


def _sort_solutions_campaigns(data: dict) -> dict:
    solutions = data.get("solutions")
    if not isinstance(solutions, dict):
        return data
    for payload in solutions.values():
        if isinstance(payload, dict):
            _sort_payload_campaigns(payload)
    _enrich_email_urls(data)
    return data


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
        data = json.load(f)
    return _sort_solutions_campaigns(data)


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = _STATIC / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<body>Missing static/index.html</body>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
