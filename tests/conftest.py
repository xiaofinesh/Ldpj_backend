"""Shared pytest fixtures (v2.5)."""
from __future__ import annotations
import sys; from pathlib import Path; import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

@pytest.fixture
def plc_cfg():
    return {"connection":{"ip":"127.0.0.1","rack":0,"slot":1,"reconnect_interval_s":1},
            "polling":{"interval_ms":50,"buffer_size":100},
            "cabin_array":{"db_number":9,"start_offset":0,"cabin_count":26,
                           "cabin_size_bytes":20,"active_start":1,"active_end":25},
            "write_back":{"leak_result_ai_offset":12,"cabin_health_offset":14},
            "fault_write":{"db_number":9,"byte_offset":520}}

@pytest.fixture
def runtime_cfg():
    return {"logging":{"level":"DEBUG","file":"/tmp/test_ldpj.log"},"threshold":0.25,
            "feature_mode":"7d","no_bottle_threshold":50.0,
            "cycle_detection":{"start_angle":100.0,"end_angle":276.0,
                               "collection_points":36,"collection_interval_s":0.1,
                               "collection_timeout_s":8.0},
            "database":{"path":"/tmp/test_ldpj.db"},"loop_interval":0.01}

@pytest.fixture
def health_cfg():
    return {"enabled":True,"check_interval_s":5,"checks":{
        "plc_connection":{"enabled":True},"model_loaded":{"enabled":True},
        "disk_space":{"enabled":True,"min_free_mb":10},
        "inference_latency":{"enabled":True,"max_ms":1000},
        "polling_thread":{"enabled":True},
        "fsm_stuck":{"enabled":True,"max_stuck_duration_s":30}}}

@pytest.fixture
def ipc_cfg():
    return {"api_server":{"enabled":False,"host":"127.0.0.1","port":18000,"api_key":"test-key"},
            "alarm_pusher":{"enabled":False,"targets":[],"push_on_leak":False,
                            "min_fault_level_to_push":"ERROR"}}
