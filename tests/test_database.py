"""Unit tests for storage.database_logger module.

v2.1 修正: 测试新增的 leak_results_plc 和 health_statuses 字段.
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
            leak_results_plc=[False, False, False],
            health_statuses=[0.95, 0.94, 0.96],
        )
        assert row_id == 1
        assert db.count_records() == 1

    def test_log_without_new_fields(self, db):
        """Backward compat: new fields are optional."""
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

    def test_query_detail_includes_new_fields(self, db):
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
            leak_results_plc=[False, True],
            health_statuses=[0.99, 0.85],
        )
        detail = db.query_record_detail(row_id)
        assert detail is not None
        assert detail["cavity_id"] == 2
        assert '"test"' in detail["features"]
        # Verify new fields
        assert detail["leak_results_plc"] is not None
        assert "true" in detail["leak_results_plc"].lower()
        assert detail["health_statuses"] is not None
        assert "0.85" in detail["health_statuses"]

    def test_db_size(self, db):
        size = db.get_db_size_mb()
        assert size >= 0
