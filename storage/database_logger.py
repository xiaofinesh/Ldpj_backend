"""SQLite database logger for test records and raw data."""

from __future__ import annotations
import json, logging, os, sqlite3, threading, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.exceptions import StorageError

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS test_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, cavity_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL, pressure_data TEXT NOT NULL, angle_data TEXT,
    ai_data TEXT, position_data TEXT, features TEXT, label INTEGER,
    probability REAL, confidence REAL, model_version TEXT, duration_s REAL,
    point_count INTEGER, leak_valve_status INTEGER, end_angle REAL,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_records_timestamp ON test_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_records_cavity ON test_records(cavity_id);
CREATE INDEX IF NOT EXISTS idx_records_label ON test_records(label);
"""

# Migration for existing databases that lack new columns
_MIGRATIONS = [
    "ALTER TABLE test_records ADD COLUMN leak_valve_status INTEGER",
    "ALTER TABLE test_records ADD COLUMN end_angle REAL",
]


class DatabaseLogger:
    def __init__(self, db_path: str | Path = "ldpj_data.db"):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
            self._run_migrations()
            self._conn.commit()
            logger.info("Database initialised: %s", self._db_path)
        except Exception as exc:
            raise StorageError(f"Database init failed: {exc}") from exc

    def _run_migrations(self) -> None:
        """Add new columns to existing databases (idempotent)."""
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def log_record(self, cavity_id, pressures, angles, ai_values, positions,
                   features, label, probability, confidence, model_version,
                   duration_s, leak_valve_status=None, end_angle=None,
                   batch_id="") -> int:
        with self._lock:
            try:
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                cur = self._conn.execute(
                    "INSERT INTO test_records (batch_id,cavity_id,timestamp,pressure_data,"
                    "angle_data,ai_data,position_data,features,label,probability,confidence,"
                    "model_version,duration_s,point_count,leak_valve_status,end_angle"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (batch_id, cavity_id, ts, json.dumps(pressures),
                     json.dumps(angles) if angles else None,
                     json.dumps(ai_values) if ai_values else None,
                     json.dumps(positions) if positions else None,
                     json.dumps(features), label, probability, confidence,
                     model_version, round(duration_s, 3), len(pressures),
                     int(leak_valve_status) if leak_valve_status is not None else None,
                     round(end_angle, 2) if end_angle is not None else None))
                self._conn.commit()
                return cur.lastrowid
            except Exception as exc:
                raise StorageError(f"log_record failed: {exc}") from exc

    def query_records(self, start_time=None, end_time=None, cavity_id=None,
                      label=None, limit=100, offset=0) -> List[Dict[str, Any]]:
        with self._lock:
            clauses, params = [], []
            if start_time: clauses.append("timestamp >= ?"); params.append(start_time)
            if end_time: clauses.append("timestamp <= ?"); params.append(end_time)
            if cavity_id is not None: clauses.append("cavity_id = ?"); params.append(cavity_id)
            if label is not None: clauses.append("label = ?"); params.append(label)
            where = " AND ".join(clauses) if clauses else "1=1"
            sql = (f"SELECT id,batch_id,cavity_id,timestamp,label,probability,"
                   f"confidence,model_version,duration_s,point_count,created_at "
                   f"FROM test_records WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?")
            params.extend([limit, offset])
            try:
                cur = self._conn.execute(sql, params)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            except Exception as exc:
                raise StorageError(f"query_records failed: {exc}") from exc

    def query_record_detail(self, record_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            try:
                cur = self._conn.execute("SELECT * FROM test_records WHERE id = ?", (record_id,))
                row = cur.fetchone()
                if row is None: return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))
            except Exception as exc:
                raise StorageError(f"query_record_detail failed: {exc}") from exc

    def count_records(self) -> int:
        with self._lock:
            try: return self._conn.execute("SELECT COUNT(*) FROM test_records").fetchone()[0]
            except Exception: return 0

    def get_db_size_mb(self) -> float:
        try: return os.path.getsize(self._db_path) / (1024 * 1024)
        except Exception: return 0.0
