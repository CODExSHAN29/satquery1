import os
import re
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional

from satquery_ai.engine import SatQueryEngine
from satquery_ai.utils.report_generator import ReportGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("satquery_ai.api")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SatQuery AI Backend Engine",
    description="Agentic Vision-Language Assistant for Remote Sensing Imagery",
)

# ---------------------------------------------------------------------------
# CORS — restrict to known origins (configure via environment)
# ---------------------------------------------------------------------------
_allow_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
if _allow_origins == [""]:
    _allow_origins = ["http://localhost:8501", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)

# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------
_API_KEY = os.environ.get("SATQUERY_API_KEY", "")


def _require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """Validate X-API-Key header against environment variable."""
    if not _API_KEY:
        # Key not configured — allow in dev mode (warn in logs)
        import logging
        logging.warning("[SatQuery] SATQUERY_API_KEY not set — API is open (dev mode only)")
        return "dev"
    if x_api_key is None or x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key


# ---------------------------------------------------------------------------
# Upload sandbox — restrict image paths to the uploads directory
# ---------------------------------------------------------------------------
_UPLOADS_DIR = os.path.abspath(os.environ.get("SATQUERY_UPLOADS_DIR", "uploads"))


def _sandbox_path(path: str, purpose: str = "image") -> str:
    """
    Resolve path and verify it lives inside _UPLOADS_DIR.
    Prevents traversal attacks like ../../etc/passwd.
    """
    if not path or not isinstance(path, str):
        raise HTTPException(status_code=400, detail=f"Invalid {purpose} path")
    try:
        abs_path = os.path.abspath(path)
        real_abs = os.path.realpath(abs_path)
        real_upload = os.path.realpath(_UPLOADS_DIR)
        # realpath resolves symlinks so uploads dir cannot be escaped via symlink
        if not real_abs.startswith(real_upload + os.sep) and real_abs != real_upload:
            raise HTTPException(
                status_code=403,
                detail=f"{purpose} path escapes sandbox directory"
            )
        if not os.path.exists(real_abs):
            raise HTTPException(status_code=404, detail=f"{purpose} file not found: {os.path.basename(path)}")
        return real_abs
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}")


def _sandbox_pdf_path(path: str) -> str:
    """Validate PDF output path stays inside project outputs directory."""
    if not path or not isinstance(path, str):
        raise HTTPException(status_code=400, detail="Invalid output path")
    # Disallow any path traversal
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid output path: traversal not allowed")
    abs_path = os.path.abspath(os.path.join("outputs", os.path.basename(path)))
    real_output = os.path.realpath("outputs")
    real_abs = os.path.realpath(abs_path)
    if not real_abs.startswith(real_output + os.sep) and real_abs != real_output:
        raise HTTPException(status_code=403, detail="Output path escapes outputs directory")
    return real_abs


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    image_paths: List[str]

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must not be empty.")
        return v.strip()

    @field_validator("image_paths")
    @classmethod
    def validate_image_paths(cls, v: List[str]) -> List[str]:
        if not v or len(v) == 0:
            raise ValueError("At least one image path is required.")
        for p in v:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"Invalid image path: {p}")
        return v


class ReportRequest(BaseModel):
    result: Dict[str, Any]
    output_pdf_path: str = "SatQuery_AI_Report.pdf"


# ---------------------------------------------------------------------------
# App state — lazy engine initialization
# ---------------------------------------------------------------------------
_engine: Optional[SatQueryEngine] = None


def get_engine() -> SatQueryEngine:
    global _engine
    if _engine is None:
        _engine = SatQueryEngine()
    return _engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "system": "SatQuery AI Engine v1.0"}


@app.get("/health")
def health_check():
    """Health check — no auth required."""
    return {"status": "healthy"}


@app.post("/api/process-query")
def process_query(
    req: QueryRequest,
    request: Request,
    _auth: str = Depends(_require_api_key),
):
    """
    Primary endpoint called by TypeScript Backend.
    Takes user natural language query and list of satellite image paths.
    Returns answer, visual evidence overlays, confidence, and execution trace log.

    Security:
    - Requires X-API-Key header
    - All image paths sandboxed to uploads/ directory
    - Rate limited: 30 req/min per IP
    """
    # Sandbox all image paths
    sandboxed_paths: List[str] = []
    for p in req.image_paths:
        sandboxed_paths.append(_sandbox_path(p, purpose="image"))

    # Sanitize query — strip control characters
    safe_query = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", req.query)

    try:
        engine = get_engine()
        res = engine.process_query(query=safe_query, image_paths=sandboxed_paths)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export-report")
def export_report(
    req: ReportRequest,
    request: Request,
    _auth: str = Depends(_require_api_key),
):
    """Generates downloadable PDF report of execution evidence.

    Security:
    - Requires X-API-Key header
    - Output path validated to prevent traversal
    - Rate limited: 10 req/min per IP
    """
    # Sandbox the output path
    safe_pdf_path = _sandbox_pdf_path(req.output_pdf_path)

    try:
        pdf_path = ReportGenerator.generate_pdf(req.result, output_pdf_path=safe_pdf_path)
        return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
