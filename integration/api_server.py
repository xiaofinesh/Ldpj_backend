"""FastAPI-based HTTP API server for external IPC data access."""

from __future__ import annotations
import logging, threading
from typing import Any, Dict, Optional
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

app = FastAPI(title="Ldpj_backend API", version="2.6.2")

_refs: Dict[str, Any] = {"db_logger": None, "health_checker": None, "polling_engine": None,
                          "model": None, "fault_reporter": None, "api_key": "change-me"}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def _verify_key(key: Optional[str] = Security(api_key_header)) -> str:
    if key != _refs["api_key"]: raise HTTPException(status_code=403, detail="Invalid API key")
    return key


@app.get("/records", dependencies=[Depends(_verify_key)])
def get_records(start_time: Optional[str] = None, end_time: Optional[str] = None,
                cavity_id: Optional[int] = None, label: Optional[int] = None,
                limit: int = Query(100, ge=1, le=10000), offset: int = Query(0, ge=0)):
    db = _refs.get("db_logger")
    if not db: raise HTTPException(503, "Database not available")
    records = db.query_records(start_time=start_time, end_time=end_time,
                                cavity_id=cavity_id, label=label, limit=limit, offset=offset)
    return {"count": len(records), "records": records}


@app.get("/records/{record_id}", dependencies=[Depends(_verify_key)])
def get_record_detail(record_id: int):
    db = _refs.get("db_logger")
    if not db: raise HTTPException(503, "Database not available")
    # Use get_full_record so the response has decoded `pressures`/`angles`
    # lists and the BLOB columns are stripped — query_record_detail leaves
    # the BLOBs in place, which FastAPI cannot JSON-serialize.
    record = db.get_full_record(record_id)
    if not record: raise HTTPException(404, "Not found")
    return record


@app.get("/status", dependencies=[Depends(_verify_key)])
def get_status():
    return {"model_loaded": _refs["model"].loaded if _refs["model"] else False,
            "model_version": _refs["model"].version if _refs["model"] else "N/A",
            "plc_connected": _refs["polling_engine"].plc_connected if _refs["polling_engine"] else False}


@app.get("/health", dependencies=[Depends(_verify_key)])
def get_health():
    hc = _refs.get("health_checker")
    if not hc: raise HTTPException(503, "Health checker not available")
    return hc.run_all_checks()


class APIServer:
    def __init__(self, ipc_cfg: Dict[str, Any]):
        cfg = ipc_cfg.get("api_server", {})
        self._enabled = cfg.get("enabled", False)
        self._host = cfg.get("host", "0.0.0.0")
        self._port = cfg.get("port", 8000)
        _refs["api_key"] = cfg.get("api_key", "change-me")
        self._thread: Optional[threading.Thread] = None

    def set_references(self, **kwargs):
        _refs.update(kwargs)

    def start(self):
        if not self._enabled: return
        self._thread = threading.Thread(target=self._run, daemon=True, name="api-server")
        self._thread.start()
        logger.info("API server started on %s:%d", self._host, self._port)

    def stop(self):
        pass  # uvicorn doesn't support graceful stop from thread easily

    def _run(self):
        import uvicorn
        uvicorn.run(app, host=self._host, port=self._port, log_level="warning")
