"""Unit tests for storage.database_logger module.

v2.5: schema uses leak_valve_status (single bool) and end_angle (REAL).
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from storage.database_logger import DatabaseLogger


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    logger = DatabaseLogger(path)
    yield logger
    logger.close()
    os.unlink(path)


class TestDatabaseLogger:
    def test_log_and_count(self, db):
        row_id = db.log_record(
            cavity_id=0,
            pressures=[100.0, 200.0, 300.0],
            angles=[1.0, 2.0, 3.0],
            ai_values=[10, 20, 30],
            positions=[0, 1, 2],
            features={"max": 300.0, "min": 100.0},
            label=1,
            probability=0.95,
            confidence=0.95,
            model_version="test_v1",
            duration_s=2.5,
            leak_valve_status=False,
            end_angle=276.5,
        )
        assert row_id == 1
        assert db.count_records() == 1

    def test_log_without_optional_fields(self, db):
        """leak_valve_status and end_angle are optional."""
        row_id = db.log_record(
            cavity_id=0,
            pressures=[100.0],
            angles=None,
            ai_values=None,
            positions=None,
            features={},
            label=-1,
            probability=0.0,
            confidence=0.0,
            model_version="unknown",
            duration_s=0.0,
        )
        assert row_id == 1

    def test_query_records(self, db):
        for i in range(5):
            db.log_record(
                cavity_id=i % 3,
                pressures=[float(i)],
                angles=None,
                ai_values=None,
                positions=None,
                features={},
                label=1 if i % 2 == 0 else 0,
                probability=0.5,
                confidence=0.5,
                model_version="test",
                duration_s=1.0,
            )
        all_records = db.query_records(limit=100)
        assert len(all_records) == 5

        # Filter by cavity
        c0 = db.query_records(cavity_id=0)
        assert all(r["cavity_id"] == 0 for r in c0)

        # Filter by label
        leaks = db.query_records(label=0)
        assert all(r["label"] == 0 for r in leaks)

    def test_query_detail_includes_v25_fields(self, db):
        row_id = db.log_record(
            cavity_id=2,
            pressures=[1.0, 2.0],
            angles=[0.5, 0.6],
            ai_values=[100, 200],
            positions=[0, 1],
            features={"test": 1.0},
            label=1,
            probability=0.9,
            confidence=0.9,
            model_version="v1",
            duration_s=1.0,
            leak_valve_status=True,
            end_angle=275.5,
        )
        detail = db.query_record_detail(row_id)
        assert detail is not None
        assert detail["cavity_id"] == 2
        assert '"test"' in detail["features"]
        # Verify v2.5 fields persisted
        assert detail["leak_valve_status"] == 1
        assert detail["end_angle"] == 275.5

    def test_db_size(self, db):
        size = db.get_db_size_mb()
        assert size >= 0


class TestV26CompressionRoundTrip:
    """v2.6: pressures/angles are stored as zlib BLOBs and decompressed on read."""

    def test_query_detail_decompresses_pressure_data(self, db):
        pressures = [600.0 + i * 0.1 for i in range(70)]
        angles = [i * 360.0 / 70 for i in range(70)]
        row_id = db.log_record(
            cavity_id=1, pressures=pressures, angles=angles,
            ai_values=None, positions=None, features={"hold_max": 600.0},
            label=1, probability=0.0, confidence=0.0,
            model_version="v2.6", duration_s=7.0,
        )
        detail = db.query_record_detail(row_id)
        # Legacy field comes back as a JSON string (decompressed by query)
        import json as _json
        recovered = _json.loads(detail["pressure_data"])
        assert len(recovered) == 70
        assert recovered[0] == pytest.approx(600.0, abs=1e-3)
        assert recovered[-1] == pytest.approx(600.0 + 69 * 0.1, abs=1e-3)
        # Same for angles
        recovered_angles = _json.loads(detail["angle_data"])
        assert len(recovered_angles) == 70

    def test_get_full_record_returns_lists(self, db):
        pressures = [100.0, 200.0, 300.0]
        angles = [10.0, 20.0, 30.0]
        row_id = db.log_record(
            cavity_id=2, pressures=pressures, angles=angles,
            ai_values=None, positions=None, features={},
            label=1, probability=0.0, confidence=0.0,
            model_version="v2.6", duration_s=0.3,
        )
        full = db.get_full_record(row_id)
        assert full is not None
        assert full["pressures"] == pytest.approx(pressures, rel=1e-5)
        assert full["angles"] == pytest.approx(angles, rel=1e-5)
        # Compressed BLOBs should be stripped from the user-facing dict
        assert "pressure_data_compressed" not in full
        assert "angle_data_compressed" not in full

    def test_legacy_pressure_data_column_is_empty(self, db):
        """v2.6 records leave the legacy text column as an empty placeholder
        (the actual data lives in pressure_data_compressed)."""
        row_id = db.log_record(
            cavity_id=1, pressures=[1.0, 2.0], angles=None,
            ai_values=None, positions=None, features={},
            label=1, probability=0.0, confidence=0.0,
            model_version="v2.6", duration_s=0.0,
        )
        # Read raw column directly to verify storage layout
        with db._lock:
            cur = db._conn.execute(
                "SELECT pressure_data, pressure_data_compressed FROM test_records WHERE id=?",
                (row_id,),
            )
            raw_text, raw_blob = cur.fetchone()
        assert raw_text == ""
        assert raw_blob is not None and isinstance(raw_blob, bytes)

    def test_quality_flags_persisted(self, db):
        """v2.6 perf-fix #4: quality_flags column round-trips through INSERT."""
        row_id = db.log_record(
            cavity_id=1, pressures=[100.0, 200.0, 300.0], angles=None,
            ai_values=None, positions=None, features={},
            label=1, probability=0.0, confidence=0.0,
            model_version="v2.6", duration_s=0.0,
            quality_flags=0xFF,  # all 8 defined bits set
        )
        with db._lock:
            cur = db._conn.execute(
                "SELECT quality_flags FROM test_records WHERE id=?", (row_id,)
            )
            (flags,) = cur.fetchone()
        assert flags == 0xFF

    def test_quality_flags_default_zero(self, db):
        """log_record without quality_flags arg writes 0 (no flags set)."""
        row_id = db.log_record(
            cavity_id=2, pressures=[1.0, 2.0], angles=None,
            ai_values=None, positions=None, features={},
            label=1, probability=0.0, confidence=0.0,
            model_version="v2.6", duration_s=0.0,
        )
        with db._lock:
            cur = db._conn.execute(
                "SELECT quality_flags FROM test_records WHERE id=?", (row_id,)
            )
            (flags,) = cur.fetchone()
        assert flags == 0

    def test_no_angles_means_null_blob(self, db):
        """When angles=None, the angle BLOB is NULL (not an empty BLOB)."""
        row_id = db.log_record(
            cavity_id=1, pressures=[1.0, 2.0, 3.0], angles=None,
            ai_values=None, positions=None, features={},
            label=1, probability=0.0, confidence=0.0,
            model_version="v2.6", duration_s=0.0,
        )
        full = db.get_full_record(row_id)
        assert full["pressures"] == pytest.approx([1.0, 2.0, 3.0], rel=1e-5)
        assert full["angles"] == []  # None decompressed to empty list
        # Direct check
        with db._lock:
            cur = db._conn.execute(
                "SELECT angle_data, angle_data_compressed FROM test_records WHERE id=?",
                (row_id,),
            )
            raw_text, raw_blob = cur.fetchone()
        assert raw_text is None
        assert raw_blob is None
