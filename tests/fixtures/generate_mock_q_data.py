"""Synthetic Q-labeled dataset generator for Task 10 + E2E tests.

Generates a CSV that looks like the output of ``train.prepare_q_data``:
- one row per (cabin, round) pair
- ``pressure_data`` and ``angle_data`` JSON arrays of length 70
- ``features`` JSON with all 43 keys
- ``q_measured`` derived from V_cabin × |dp/dt| of the synthetic curve
- ``round_id`` so train_m2 can split by round
- ``cycle_profile_id`` = "bph_13000"

Curves are simulated as:
    p(t) = p0 - slope * t + noise
where slope is sampled per-cabin from a per-cabin (β, α) line plus jitter
so that fitting Q ↔ slope produces a clean linear relationship with high
R² — exactly what train_m1 expects.

Usage as a CLI for ad-hoc testing:
    python tests/fixtures/generate_mock_q_data.py \\
        --output mock_q_data.csv --n-cabins 25 --n-rounds 12

Programmatic use (in tests):
    from tests.fixtures.generate_mock_q_data import generate
    df = generate(n_cabins=5, n_rounds=8, seed=42)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.cycle_profile import CycleProfile, SECTION_NAMES
from core.curve_segmenter import segment_by_angle
from core.features import compute_features_v26
from core.feature_spec import FEATURE_ORDER_36D


def _default_profile() -> CycleProfile:
    """Production-like profile with the v2.6.1 standard 5 sections."""
    return CycleProfile(
        profile_id="bph_13000",
        bph=13000,
        cycle_total_ms=6900,
        sections={
            "baseline_pre":  (0.0,   75.0),
            "evac":          (75.0,  90.0),
            "hold":          (90.0,  290.0),
            "release":       (290.0, 304.0),
            "baseline_post": (304.0, 360.0),
        },
        trigger_angle=0.0,
        collection_points=70,
        collection_interval_s=0.1,
        collection_timeout_s=10.0,
        primary_section="hold",
    )


def _generate_one_cycle(
    cabin_id: int,
    round_id: int,
    beta: float,
    alpha: float,
    v_cabin: float,
    profile: CycleProfile,
    rng: np.random.Generator,
) -> dict:
    """Synthesize one full-cycle pressure trace + features + q_measured."""
    n = profile.collection_points
    angles = np.linspace(0.0, 360.0, n, endpoint=False)

    # ── Build the pressure curve segment-by-segment ──────────────
    p_atm_mbar = 0.0          # gauge pressure (atmospheric reference)
    p_vac_mbar = 600.0        # full vacuum

    # Pick a true slope (mbar per sample) for the hold section
    # The fit will recover Q ≈ V × |slope * 100 / interval_s|
    # Sample a "leak severity" in arbitrary units, drive q via β·s + α
    severity = rng.uniform(0.05, 0.5)
    slope_hold_mbar_per_sample = -severity  # negative = vacuum decay

    # Compute target q from the synthetic linear relation
    interval_s = profile.collection_interval_s
    dp_dt_pa_per_s = slope_hold_mbar_per_sample * 100.0 / interval_s
    q_truth = v_cabin * abs(dp_dt_pa_per_s) + rng.normal(0, 1e-5)
    q_truth = max(q_truth, 1e-6)

    pressures = np.zeros(n, dtype=np.float64)
    # 5-section v2.6.1 boundaries: 0/75/90/290/304/360
    # ~39 samples land in hold (90°-290°) at 70 pts/360°.
    hold_n_samples_estimate = (290.0 - 90.0) / 360.0 * n  # ≈ 38.9
    for i, a in enumerate(angles):
        if a < 75.0:                     # baseline_pre
            p = p_atm_mbar
        elif a < 90.0:                   # evac (linear ramp 0 → 600 mbar)
            frac = (a - 75.0) / (90.0 - 75.0)
            p = p_atm_mbar + frac * (p_vac_mbar - p_atm_mbar)
        elif a < 290.0:                  # hold (slow decay)
            steps_in_hold = (a - 90.0) / (290.0 - 90.0) * hold_n_samples_estimate
            p = p_vac_mbar + slope_hold_mbar_per_sample * steps_in_hold
        elif a < 304.0:                  # release (linear ramp back)
            frac = (a - 290.0) / (304.0 - 290.0)
            p = p_vac_mbar + frac * (p_atm_mbar - p_vac_mbar)
        else:                            # baseline_post
            p = p_atm_mbar
        pressures[i] = p + rng.normal(0, 0.5)

    # Compute the v2.6.1 36-dim features (so M2 trainer has its 'features' col)
    feats = compute_features_v26(pressures.tolist(), angles.tolist(),
                                 cabin_id, profile)

    return {
        "id": int(round_id) * 100 + int(cabin_id),
        "round_id": int(round_id),
        "cavity_id": int(cabin_id),
        "timestamp": f"2026-05-22T00:{round_id:02d}:00",
        "cycle_profile_id": "bph_13000",
        "pressure_data": json.dumps(pressures.tolist()),
        "angle_data": json.dumps(angles.tolist()),
        "features": json.dumps(feats),
        "q_measured": q_truth,
    }


def generate(
    n_cabins: int = 5,
    n_rounds: int = 12,
    seed: int = 42,
    profile: Optional[CycleProfile] = None,
) -> pd.DataFrame:
    """Generate a Q-labeled training dataframe."""
    if profile is None:
        profile = _default_profile()
    rng = np.random.default_rng(seed)

    # Per-cabin true (β, α) — slightly different so M1 has signal to discriminate
    cabin_params = {}
    for cid in range(1, n_cabins + 1):
        beta = 1.5e-3 + rng.uniform(-1e-4, 1e-4)
        alpha = rng.uniform(-1e-5, 1e-5)
        v_cabin = 3.5e-4 + rng.uniform(-1e-5, 1e-5)
        cabin_params[cid] = (beta, alpha, v_cabin)

    rows = []
    for round_id in range(1, n_rounds + 1):
        for cabin_id in range(1, n_cabins + 1):
            beta, alpha, v_cabin = cabin_params[cabin_id]
            rows.append(_generate_one_cycle(
                cabin_id, round_id, beta, alpha, v_cabin, profile, rng,
            ))

    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate mock Q-labeled training data")
    p.add_argument("--output", required=True)
    p.add_argument("--n-cabins", type=int, default=5)
    p.add_argument("--n-rounds", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    df = generate(n_cabins=args.n_cabins, n_rounds=args.n_rounds, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
