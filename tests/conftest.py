"""Shared pytest fixtures for Ldpj_backend tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def plc_cfg() -> dict:
    """Minimal PLC configuration for testing."""
    return {
        "connection": {"ip": "127.0.0.1", "rack": 0, "slot": 1, "reconnect_interval_s": 1},
        "polling": {"interval_ms": 50, "buffer_size": 100},
        "cabin_array": {
            "db_number": 9,
            "start_offset": 0,
            "cabin_count": 25,
            "cabin_size_bytes": 12,
        },
        "write_back": {"db_number": 9, "byte_offset": 200, "scale": 10, "base": 0},
        "fault_write": {"db_number": 9, "byte_offset": 202},
    }


@pytest.fixture
def runtime_cfg() -> dict:
    """Minimal runtime configuration for testing."""
    return {
        "logging": {"level": "DEBUG", "file": "/tmp/test_ldpj.log"},
        "threshold": 0.3,
        "feature_mode": "7d",
        "cycle_detection": {
            "start_pressure_drop": 50.0,
            "end_pressure_rise": 50.0,
            "min_collection_points": 5,
            "max_collection_points": 100,
            "max_collection_duration_s": 10,
            "collection_timeout_s": 15,
            "idle_pressure_min": 900.0,
        },
        "database": {"path": "/tmp/test_ldpj.db"},
        "loop_interval": 0.01,
    }


@pytest.fixture
def health_cfg() -> dict:
    return {
        "enabled": True,
        "check_interval_s": 5,
        "checks": {
            "plc_connection": {"enabled": True},
            "model_loaded": {"enabled": True},
            "disk_space": {"enabled": True, "min_free_mb": 10},
            "inference_latency": {"enabled": True, "max_ms": 1000},
            "polling_thread": {"enabled": True},
            "fsm_stuck": {"enabled": True, "max_stuck_duration_s": 30},
        },
    }


@pytest.fixture
def ipc_cfg() -> dict:
    return {
        "api_server": {"enabled": False, "host": "127.0.0.1", "port": 18000, "api_key": "test-key"},
        "alarm_pusher": {
            "enabled": False,
            "targets": [],
            "push_on_leak": False,
            "min_fault_level_to_push": "ERROR",
        },
    }
