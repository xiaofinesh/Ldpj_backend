#!/usr/bin/env python3
"""V_cabin calibration recorder.

Records a water-fill (注水法) calibration for one cabin into
``configs/cabins.yaml`` and appends an audit row to
``data/calibration_history/v_cabin_log.csv``.

Usage:
    python scripts/calibrate_v_cabin.py \
        --cabin 5 --weights-grams 348.2,348.5,348.0 \
        --calibrator alice --notes "first batch"

Behavior:
1. Validates input (>= 3 repeats, CV <= --cv-limit, default 2%).
2. Computes mean V_cabin in m³ (1 g water ≈ 1 mL ≈ 1e-6 m³ at room temp)
   and u_v_cabin (sample standard deviation, 1-sigma).
3. If CV exceeds the limit: refuses to update yaml, but still logs the
   measurement to history (marked accepted=no) for traceability.
4. Otherwise: rewrites the cabin's entry in cabins.yaml in-place and
   appends an accepted=yes row to the log CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CABINS_YAML = PROJECT_ROOT / "configs" / "cabins.yaml"
LOG_CSV = PROJECT_ROOT / "data" / "calibration_history" / "v_cabin_log.csv"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record V_cabin calibration")
    p.add_argument("--cabin", type=int, required=True, help="Cabin ID (1..25)")
    p.add_argument(
        "--weights-grams", required=True,
        help="Comma-separated water weights in grams (>= 3 repeats)",
    )
    p.add_argument("--calibrator", default="", help="Calibrator name")
    p.add_argument("--notes", default="", help="Free-form notes")
    p.add_argument(
        "--cv-limit", type=float, default=0.02,
        help="Max coefficient of variation; CV above this rejects the write (default 2%%)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print results, don't touch yaml or history log",
    )
    return p.parse_args(argv)


def _stats(weights):
    """Return (mean_g, std_g_sample, cv) for a list of weights."""
    n = len(weights)
    mean_g = sum(weights) / n
    # sample standard deviation (n-1 denominator)
    var_g = sum((w - mean_g) ** 2 for w in weights) / (n - 1)
    std_g = var_g ** 0.5
    cv = std_g / mean_g if mean_g > 0 else 0.0
    return mean_g, std_g, cv


def _update_cabin_yaml(cabin_id, v_cabin, u_v_cabin, calibrator, notes):
    with open(CABINS_YAML, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    today = time.strftime("%Y-%m-%d")
    if not cfg.get("calibration_date"):
        cfg["calibration_date"] = today
    if calibrator and not cfg.get("calibrator"):
        cfg["calibrator"] = calibrator

    cfg.setdefault("cabins", {})
    cfg["cabins"][cabin_id] = {
        "v_cabin": float(f"{v_cabin:.4e}"),
        "u_v_cabin": float(f"{u_v_cabin:.2e}"),
        "notes": notes or f"calibrated {today}",
    }

    with open(CABINS_YAML, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


def _append_log(cabin_id, weights, mean_g, std_g, cv,
                calibrator, notes, accepted):
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_CSV.exists()
    with open(LOG_CSV, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow([
                "timestamp", "cabin_id", "weights_g", "mean_g", "std_g",
                "cv_percent", "v_cabin_m3", "u_v_cabin_m3",
                "calibrator", "notes", "accepted",
            ])
        w.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            cabin_id,
            "|".join(f"{x:.2f}" for x in weights),
            f"{mean_g:.3f}", f"{std_g:.3f}", f"{cv * 100:.2f}",
            f"{mean_g * 1e-6:.4e}", f"{std_g * 1e-6:.2e}",
            calibrator, notes, "yes" if accepted else "no",
        ])


def main(argv=None) -> int:
    args = parse_args(argv)

    if not (1 <= args.cabin <= 25):
        print(f"ERROR: cabin must be in [1, 25], got {args.cabin}", file=sys.stderr)
        return 2

    try:
        weights = [float(x.strip()) for x in args.weights_grams.split(",")]
    except ValueError as exc:
        print(f"ERROR: invalid --weights-grams: {exc}", file=sys.stderr)
        return 2

    if len(weights) < 3:
        print(f"ERROR: need >= 3 repeats, got {len(weights)}", file=sys.stderr)
        return 2

    if any(w <= 0 for w in weights):
        print("ERROR: weights must be positive", file=sys.stderr)
        return 2

    mean_g, std_g, cv = _stats(weights)

    # 1 g water at room temperature ≈ 1 mL = 1e-6 m³
    v_cabin_m3 = mean_g * 1e-6
    u_v_cabin_m3 = std_g * 1e-6

    print(f"\n=== V_cabin Calibration: Cabin {args.cabin} ===")
    print(f"Repeats:        {weights} g")
    print(f"Mean:           {mean_g:.2f} g  ->  {v_cabin_m3:.3e} m^3")
    print(f"Std deviation:  {std_g:.3f} g  ->  u = {u_v_cabin_m3:.3e} m^3")
    print(f"CV:             {cv * 100:.2f}%  (limit {args.cv_limit * 100:.1f}%)")

    if cv > args.cv_limit:
        print("\nWARNING: CV exceeds limit. Will NOT write to yaml.")
        print("  Repeat the measurement before recording.")
        if not args.dry_run:
            _append_log(
                args.cabin, weights, mean_g, std_g, cv,
                args.calibrator,
                (args.notes + " [REJECTED: CV exceeded]").strip(),
                accepted=False,
            )
        return 1

    if args.dry_run:
        print("\n[dry-run] Would update configs/cabins.yaml.")
        return 0

    _update_cabin_yaml(args.cabin, v_cabin_m3, u_v_cabin_m3,
                       args.calibrator, args.notes)
    _append_log(args.cabin, weights, mean_g, std_g, cv,
                args.calibrator, args.notes, accepted=True)

    try:
        rel_log = LOG_CSV.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_log = LOG_CSV  # tests / out-of-tree runs
    print(f"\nUpdated configs/cabins.yaml for cabin {args.cabin}")
    print(f"Logged to {rel_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
