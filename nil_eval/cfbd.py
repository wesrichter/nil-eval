"""Minimal CollegeFootballData.com API client with on-disk caching.

The API key is read from the CFBD_API_KEY environment variable. (In a managed
environment where a proxy injects the key, any placeholder value works.)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.collegefootballdata.com"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
# Completed seasons never change; in-progress data refreshes after this many hours.
DEFAULT_TTL_HOURS = 12


def _cache_path(endpoint: str, params: dict) -> Path:
    key = endpoint + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    safe = endpoint.strip("/").replace("/", "_")
    return CACHE_DIR / f"{safe}_{digest}.json"


def get(endpoint: str, ttl_hours: float = DEFAULT_TTL_HOURS, **params) -> list | dict:
    """GET an endpoint, using the disk cache when fresh."""
    params = {k: v for k, v in params.items() if v is not None}
    path = _cache_path(endpoint, params)
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        return json.loads(path.read_text())

    headers = {"Accept": "application/json"}
    api_key = os.environ.get("CFBD_API_KEY", "proxy-injected")
    headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(4):
        resp = requests.get(f"{BASE_URL}/{endpoint.strip('/')}", params=params,
                            headers=headers, timeout=120)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        data = resp.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data
    resp.raise_for_status()
    return []
