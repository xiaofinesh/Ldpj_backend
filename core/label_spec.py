"""Label convention, pressure scale, and timing specification.

==========================================================================
PRESSURE SCALE (正值真空度)
==========================================================================

    RT_Pressure:  0 mbar = atmospheric (常压, no vacuum)
                ~600 mbar = full vacuum (满真空)

    保压阶段: ~600 mbar, 缓慢下降 10~50 mbar (真空度降低 = 泄漏信号)
    复压阶段: 快速从 ~550 降到 0
    空闲阶段: 0 mbar

    正值越大 = 真空度越高 = 密封越好

==========================================================================
LABEL CONVENTION
==========================================================================

    label = 0  →  LEAK (漏液)
    label = 1  →  OK   (密封正常)
    label = -1 →  N/A  (无推理)

==========================================================================
"""

LABEL_LEAK = 0
LABEL_OK = 1
LABEL_UNKNOWN = -1

PRESSURE_ATMOSPHERIC = 0.0
PRESSURE_FULL_VACUUM = 600.0
VACUUM_THRESHOLD = 500.0
REPRESS_THRESHOLD = 400.0

REVOLUTION_MS = 6944
STATION_COUNT = 25
STATION_TIME_MS = REVOLUTION_MS / STATION_COUNT

VACUUM_START_STATION = 5
VACUUM_PUMP_DURATION_MS = 400
HOLD_DURATION_MS = 4100
PREDICTION_MARGIN_MS = 500

RECOMMENDED_INTERVAL_MS = 100


def flip_old_label(old_prediction: int) -> int:
    """Old: 0=good, 1=leak → New: 0=leak, 1=good"""
    if old_prediction == 0:
        return LABEL_OK
    elif old_prediction == 1:
        return LABEL_LEAK
    return LABEL_UNKNOWN
