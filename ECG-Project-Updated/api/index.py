"""Vercel ASGI entry point. All inference endpoints share one function."""
import sys
from pathlib import Path
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.main import app as ecg_api

# Mount rather than strip URLs with rewrites, preserving query strings,
# multipart uploads and the /api prefix in generated OpenAPI documentation.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount('/api', ecg_api)
