"""Cycle profile abstraction for v2.6.

A profile describes the time-domain layout of one production cycle,
including section boundaries (in degrees), sampling parameters, and
machine throughput tier.

Currently only one profile is populated (bph_13000), but the abstraction
allows future expansion:
1. Add more profile entries to runtime.yaml
2. Replace `load_active_cycle_profile()` with a PLC-recipe reader
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


# Standard section names (order matters for feature extraction).
#
# v2.6.1 (post-131808 calibration): the v2.6 "stable" segment (93°-115°)
# was an artifact — real production data shows the trend slope is
# continuous from the moment vacuum is reached (~90°) all the way to
# release (~290°). We merged stable into hold, leaving 5 sections.
# Boundaries below match the modes of the 4 transition points in 131808.
SECTION_NAMES: List[str] = [
    "baseline_pre", "evac", "hold", "release", "baseline_post",
]


@dataclass
class CycleProfile:
    """Describes one production cycle layout."""

    profile_id: str
    bph: int
    cycle_total_ms: int
    sections: Dict[str, Tuple[float, float]]
    trigger_angle: float
    collection_points: int
    collection_interval_s: float
    collection_timeout_s: float
    primary_section: str = "hold"
    description: str = ""

    def validate(self) -> None:
        """Sanity-check the profile. Raises ValueError on invalid config."""
        # 1. All standard section names must be present
        missing = set(SECTION_NAMES) - set(self.sections.keys())
        if missing:
            raise ValueError(
                f"Profile {self.profile_id}: missing sections {sorted(missing)}"
            )

        # 2. Section boundaries must be monotonically non-overlapping in standard order
        prev_end = -1.0
        for name in SECTION_NAMES:
            start, end = self.sections[name]
            if start < prev_end:
                raise ValueError(
                    f"Profile {self.profile_id}: section '{name}' starts at {start}° "
                    f"before previous section ended at {prev_end}°"
                )
            if end <= start:
                raise ValueError(
                    f"Profile {self.profile_id}: section '{name}' has invalid range "
                    f"[{start}, {end})"
                )
            prev_end = end

        # 3. Last section must end at <= 360.0
        last_end = self.sections[SECTION_NAMES[-1]][1]
        if last_end > 360.001:
            raise ValueError(
                f"Profile {self.profile_id}: total range exceeds 360° (ends at {last_end}°)"
            )

        # 4. primary_section must be one of the standard names
        if self.primary_section not in SECTION_NAMES:
            raise ValueError(
                f"Profile {self.profile_id}: primary_section '{self.primary_section}' "
                f"not in {SECTION_NAMES}"
            )

        # 5. Sampling sanity
        if self.collection_points <= 0 or self.collection_interval_s <= 0:
            raise ValueError(
                f"Profile {self.profile_id}: invalid sampling params "
                f"(points={self.collection_points}, interval_s={self.collection_interval_s})"
            )

        expected_duration = self.collection_points * self.collection_interval_s
        if expected_duration > self.collection_timeout_s:
            raise ValueError(
                f"Profile {self.profile_id}: timeout ({self.collection_timeout_s}s) "
                f"shorter than expected collection duration ({expected_duration}s)"
            )

        logger.info(
            "CycleProfile validated: %s (bph=%d, %d points × %.0fms, primary=%s)",
            self.profile_id, self.bph,
            self.collection_points, self.collection_interval_s * 1000,
            self.primary_section,
        )

    @classmethod
    def from_dict(cls, profile_id: str, data: Dict[str, Any]) -> "CycleProfile":
        """Build CycleProfile from a yaml-loaded dict."""
        sections = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in data.get("sections", {}).items()
        }
        collection = data.get("collection", {})
        return cls(
            profile_id=profile_id,
            bph=int(data.get("bph", 0)),
            cycle_total_ms=int(data.get("cycle_total_ms", 0)),
            sections=sections,
            trigger_angle=float(collection.get("trigger_angle", 0.0)),
            collection_points=int(collection.get("points", 70)),
            collection_interval_s=float(collection.get("interval_s", 0.1)),
            collection_timeout_s=float(collection.get("timeout_s", 10.0)),
            primary_section=data.get("primary_section", "hold"),
            description=data.get("description", ""),
        )


def load_active_cycle_profile(runtime_cfg: Dict[str, Any]) -> CycleProfile:
    """Load the currently active profile from a runtime.yaml dict.

    NOTE: v2.6 reads from yaml. Future v2.7 may replace with PLC recipe reading
    (the wire is the same: returns a validated CycleProfile object).

    Raises ValueError if active_profile is not set, not found, or invalid.
    """
    active_id = runtime_cfg.get("active_profile")
    if not active_id:
        raise ValueError("runtime.yaml: 'active_profile' is not set")

    profiles = runtime_cfg.get("cycle_profiles", {})
    if active_id not in profiles:
        raise ValueError(
            f"Active profile '{active_id}' not found in cycle_profiles. "
            f"Available: {list(profiles.keys())}"
        )

    profile = CycleProfile.from_dict(active_id, profiles[active_id])
    profile.validate()
    return profile
