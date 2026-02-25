"""Data exporter – dump test records to CSV for offline training."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from storage.database_logger import DatabaseLogger

logger = logging.getLogger(__name__)


def export_to_csv(
    db: DatabaseLogger,
    output_path: str | Path,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    cavity_id: Optional[int] = None,
    include_raw: bool = True,
) -> int:
    """Export records from the database to a CSV file.

    Parameters
    ----------
    db : DatabaseLogger
        Active database logger instance.
    output_path : str | Path
        Destination CSV file path.
    start_time, end_time : str, optional
        ISO-8601 time range filter.
    cavity_id : int, optional
        Filter by cabin.
    include_raw : bool
        Whether to include raw pressure/angle data columns.

    Returns
    -------
    int
        Number of records exported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch all matching record IDs first (summary query)
    summaries = db.query_records(
        start_time=start_time,
        end_time=end_time,
        cavity_id=cavity_id,
        limit=999_999,
    )

    if not summaries:
        logger.info("No records to export")
        return 0

    header = [
        "id", "batch_id", "cavity_id", "timestamp",
        "label", "probability", "confidence",
        "model_version", "duration_s", "point_count",
    ]
    if include_raw:
        header.extend(["pressure_data", "angle_data", "features"])

    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k, "") for k in header if k in s}
            if include_raw:
                detail = db.query_record_detail(s["id"])
                if detail:
                    row["pressure_data"] = detail.get("pressure_data", "")
                    row["angle_data"] = detail.get("angle_data", "")
                    row["features"] = detail.get("features", "")
            writer.writerow(row)
            count += 1

    logger.info("Exported %d records to %s", count, output_path)
    return count
