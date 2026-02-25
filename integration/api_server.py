"""FastAPI-based HTTP API server for external IPC data access.

Provides endpoints for querying test records, system status, and health
reports.  Runs in a background thread via uvicorn.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app (module-level so routes can be registered at import time)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ldpj_backend API",
    description="漏液检测后端系统数据查询与状态接口",
    version="2.1.0",
)

# Shared references injected at startup
_refs: Dict[str, Any] = {
    "db_logger": None,
    "health_checker": None,
    "polling_engine": None,
    "model": None,
    "fault_reporter": None,
    "api_key": "change-me-in-production",
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_key(key: Optional[str] = Security(api_key_header)) -> str:
    if key != _refs["api_key"]:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/records", dependencies=[Depends(_verify_key)])
def get_records(
    start_time: Optional[str] = Query(None, description="ISO 8601 开始时间"),
    end_time: Optional[str] = Query(None, description="ISO 8601 结束时间"),
    cavity_id: Optional[int] = Query(None, description="舱室ID"),
    label: Optional[int] = Query(None, description="标签 (0=漏液, 1=正常)"),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    db = _refs.get("db_logger")
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        records = db.query_records(
            start_time=start_time,
            end_time=end_time,
            cavity_id=cavity_id,
            label=label,
            limit=limit,
            offset=offset,
        )
        return {"count": len(records), "records": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/records/{record_id}", dependencies=[Depends(_verify_key)])
def get_record_detail(record_id: int) -> Dict[str, Any]:
    db = _refs.get("db_logger")
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    record = db.query_record_detail(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.get("/status", dependencies=[Depends(_verify_key)])
def get_status() -> Dict[str, Any]:
    model = _refs.get("model")
    poller = _refs.get("polling_engine")
    db = _refs.get("db_logger")
    return {
        "system": "ldpj_backend",
        "version": "2.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {
            "loaded": model.loaded if model else False,
            "version": model.version if model else "N/A",
        },
        "plc": {
            "connected": poller.plc_connected if poller else False,
            "poll_stats": poller.stats if poller else {},
        },
        "database": {
            "record_count": db.count_records() if db else 0,
            "size_mb": db.get_db_size_mb() if db else 0,
        },
    }


@app.get("/health", dependencies=[Depends(_verify_key)])
def get_health() -> Dict[str, Any]:
    hc = _refs.get("health_checker")
    if hc is None:
        raise HTTPException(status_code=503, detail="Health checker not available")
    return hc.run_all_checks()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

class APIServer:
    """Manages the uvicorn server running in a background thread.

    Parameters
    ----------
    ipc_cfg : dict
        The full content of ``ipc.yaml``.
    """

    def __init__(self, ipc_cfg: Dict[str, Any]):
        api_cfg = ipc_cfg.get("api_server", {})
        self._enabled = api_cfg.get("enabled", False)
        self._host = api_cfg.get("host", "0.0.0.0")
        self._port = api_cfg.get("port", 8000)
        _refs["api_key"] = api_cfg.get("api_key", "change-me-in-production")
        self._thread: Optional[threading.Thread] = None

    def set_references(self, **kwargs: Any) -> None:
        _refs.update(kwargs)

    def start(self) -> None:
        if not self._enabled:
            logger.info("API server disabled by config")
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="api-server")
        self._thread.start()
        logger.info("API server started on %s:%d", self._host, self._port)

    def stop(self) -> None:
        # uvicorn doesn't have a clean shutdown from a thread; daemon=True
        # ensures it dies with the main process.
        pass

    def _run(self) -> None:
        import uvicorn
        uvicorn.run(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
