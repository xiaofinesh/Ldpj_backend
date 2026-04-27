"""Unit tests for health subsystem."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health.fault_codes import FAULT_CODES
from health.fault_reporter import FaultLevel, FaultReporter


class TestFaultCodes:
    def test_known_code(self):
        f = FAULT_CODES["F001"]
        assert f["level"] == "CRITICAL"
        assert f["plc_value"] == 1

    def test_all_codes_have_required_fields(self):
        for code, info in FAULT_CODES.items():
            assert "description" in info
            assert "level" in info
            assert "plc_value" in info
            assert info["level"] in ("INFO", "WARNING", "ERROR", "CRITICAL")


class TestFaultReporter:
    def test_raise_and_resolve(self):
        reporter = FaultReporter()
        reporter.raise_fault("F001", "PLC disconnected")
        assert "F001" in reporter.active_faults
        assert reporter.has_critical

        reporter.resolve_fault("F001")
        assert "F001" not in reporter.active_faults
        assert not reporter.has_critical

    def test_unknown_code_defaults_to_error(self):
        reporter = FaultReporter()
        reporter.raise_fault("F999", "unknown")
        ev = reporter.active_faults["F999"]
        assert ev.fault.level == FaultLevel.ERROR

    def test_duplicate_raise(self):
        reporter = FaultReporter()
        reporter.raise_fault("F004")
        reporter.raise_fault("F004")
        assert len(reporter.active_faults) == 1

    def test_callback(self):
        reporter = FaultReporter()
        events = []
        reporter.register_callback(lambda e: events.append(e))
        reporter.raise_fault("F005", "Disk full")
        assert len(events) == 1
        assert events[0].fault.code == "F005"

    def test_highest_plc_value_severity_based(self):
        """Most severe fault wins, regardless of numeric plc_value."""
        reporter = FaultReporter()
        reporter.raise_fault("F004")  # WARNING, plc_value=4
        reporter.raise_fault("F001")  # CRITICAL, plc_value=1
        # CRITICAL wins -> plc_value=1
        assert reporter.get_highest_plc_value() == 1

    def test_highest_plc_value_empty(self):
        reporter = FaultReporter()
        assert reporter.get_highest_plc_value() == 0

    def test_summary(self):
        reporter = FaultReporter()
        reporter.raise_fault("F003")
        s = reporter.summary()
        assert s["active_count"] == 1
        assert len(s["faults"]) == 1
