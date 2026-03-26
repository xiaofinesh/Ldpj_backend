"""Unit tests for health subsystem."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from health.fault_codes import FaultLevel, get_fault
from health.fault_reporter import FaultReporter


class TestFaultCodes:
    def test_known_code(self):
        fc = get_fault("F001")
        assert fc.code == "F001"
        assert fc.level == FaultLevel.CRITICAL
        assert fc.plc_value == 1

    def test_unknown_code(self):
        fc = get_fault("F999")
        assert fc.code == "F999"
        assert fc.level == FaultLevel.ERROR


class TestFaultReporter:
    def test_raise_and_resolve(self):
        reporter = FaultReporter()
        reporter.raise_fault("F001", "PLC disconnected")
        assert "F001" in reporter.active_faults
        assert reporter.has_critical

        reporter.resolve_fault("F001")
        assert "F001" not in reporter.active_faults
        assert not reporter.has_critical

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

    def test_highest_plc_value(self):
        reporter = FaultReporter()
        reporter.raise_fault("F004")  # WARNING, plc_value=4
        reporter.raise_fault("F001")  # CRITICAL, plc_value=1
        # Highest severity is CRITICAL
        assert reporter.get_highest_plc_value() == 1

    def test_summary(self):
        reporter = FaultReporter()
        reporter.raise_fault("F003")
        s = reporter.summary()
        assert s["active_count"] == 1
        assert len(s["faults"]) == 1
