"""Prepare Q-labeled training data from a v2.6 raw CSV (Task 10).

Reads a CSV exported from the v2.6 backend (with ``pressure_data`` and
``angle_data`` JSON columns), looks up V_cabin per cavity from
``configs/cabins.yaml``, fits a line through the primary section's
pressure curve, and writes:

    q_measured  = V_cabin × |dp/dt|     [Pa·m³/s]
    dp_dt_pa_per_s                       [Pa/s]

Usage
-----
    python -m train.prepare_q_data \\
        --raw-csv export_S2.csv \\
        --cabins-config configs/cabins.yaml \\
        --runtime-config configs/runtime.yaml \\
        --output train_data_S2.csv

Pressure unit assumption
------------------------
The system stores pressures in mbar; we convert to Pa (×100) before
computing dp/dt. ``collection_interval_s`` from the active CycleProfile
gives the sample spacing in seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Bootstrap project path so this script works as `python -m train.prepare_q_data`
# AND as a direct invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loaders import get_v_cabin
from core.curve_segmenter import segment_by_angle
from core.cycle_profile import load_active_cycle_profile

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Minimum points in the primary section before we trust the linear fit.
_MIN_HOLD_POINTS = 5


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute q_measured from v2.6 raw CSV"
    )
    p.add_argument("--raw-csv", required=True, help="Input CSV from data_exporter")
    p.add_argument("--cabins-config", default="configs/cabins.yaml",
                   help="Path to cabins.yaml")
    p.add_argument("--runtime-config", default="configs/runtime.yaml",
                   help="Path to runtime.yaml")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--profile-id", default=None,
                   help="Override active_profile from runtime.yaml")
    return p.parse_args(argv)


def compute_q_for_row(row, profile, cabins_cfg) -> tuple[float | None, float | None]:
    """Compute (q_measured_pa_m3_s, dp_dt_pa_per_s) for one CSV row.

    Returns (None, None) when the primary section has fewer than
    ``_MIN_HOLD_POINTS`` points or the JSON columns are unparseable.
    """
    try:
        pressures = json.loads(row["pressure_data"])
        angles = json.loads(row["angle_data"])
        cabin_id = int(row["cavity_id"])
    except (KeyError, ValueError, TypeError) as exc:
        logger.debug("Row parse error: %s", exc)
        return None, None

    sections = segment_by_angle(pressures, angles, profile)
    primary = sections.get(profile.primary_section, [])
    if len(primary) < _MIN_HOLD_POINTS:
        return None, None

    # Linear fit on primary section: slope is mbar per sample
    x = np.arange(len(primary), dtype=np.float64)
    slope_mbar_per_sample = float(np.polyfit(x, primary, 1)[0])

    # Convert to Pa/s: 1 mbar = 100 Pa; samples spaced by interval_s seconds
    interval_s = profile.collection_interval_s
    dp_dt_pa_per_s = slope_mbar_per_sample * 100.0 / interval_s

    v_cabin, _u = get_v_cabin(cabins_cfg, cabin_id)
    q = v_cabin * abs(dp_dt_pa_per_s)
    return q, dp_dt_pa_per_s


def main(argv=None) -> int:
    args = parse_args(argv)

    # Load configs
    cabins_path = Path(args.cabins_config)
    if not cabins_path.exists():
        sys.exit(f"ERROR: cabins config not found: {cabins_path}")
    with open(cabins_path, "r", encoding="utf-8") as fh:
        cabins_cfg = yaml.safe_load(fh) or {}

    runtime_path = Path(args.runtime_config)
    if not runtime_path.exists():
        sys.exit(f"ERROR: runtime config not found: {runtime_path}")
    with open(runtime_path, "r", encoding="utf-8") as fh:
        runtime_cfg = yaml.safe_load(fh) or {}
    if args.profile_id:
        runtime_cfg["active_profile"] = args.profile_id
    profile = load_active_cycle_profile(runtime_cfg)

    logger.info("Profile %s, primary_section=%s, interval=%.2fs",
                profile.profile_id, profile.primary_section,
                profile.collection_interval_s)

    # Load raw CSV
    df = pd.read_csv(args.raw_csv)
    logger.info("Loaded %d rows from %s", len(df), args.raw_csv)

    # Compute q for each row
    q_list, dp_list = [], []
    skipped = 0
    for _, row in df.iterrows():
        q, dp = compute_q_for_row(row, profile, cabins_cfg)
        if q is None:
            skipped += 1
        q_list.append(q)
        dp_list.append(dp)

    df["q_measured"] = q_list
    df["dp_dt_pa_per_s"] = dp_list

    # Drop rows where q couldn't be computed
    df_valid = df.dropna(subset=["q_measured"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_valid.to_csv(out_path, index=False)

    logger.info("Wrote %d rows to %s (skipped %d short/invalid)",
                len(df_valid), out_path, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
