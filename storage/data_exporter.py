"""Data exporter – interactive CSV export with date range selection.

v2.5: Interactive terminal UI for selecting export date range.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from storage.compression import decompress_float_array
from storage.database_logger import DatabaseLogger

logger = logging.getLogger(__name__)


def get_available_dates(db: DatabaseLogger) -> List[str]:
    """Query distinct dates from the database, sorted ascending."""
    with db._lock:
        try:
            cur = db._conn.execute(
                "SELECT DISTINCT DATE(timestamp) as d FROM test_records ORDER BY d"
            )
            return [row[0] for row in cur.fetchall() if row[0]]
        except Exception:
            return []


def interactive_export(db: DatabaseLogger, base_dir: str | Path = ".") -> None:
    """Interactive terminal-based export with date selection.

    Shows available dates, lets user pick specific dates or a range,
    then exports matching records to CSV.
    """
    dates = get_available_dates(db)
    if not dates:
        print("数据库中没有记录，无法导出。")
        return

    # ── Show available dates ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("  数据导出 — 可用日期")
    print("=" * 50)

    # Count records per date
    date_counts = {}
    for d in dates:
        with db._lock:
            try:
                cur = db._conn.execute(
                    "SELECT COUNT(*) FROM test_records WHERE DATE(timestamp) = ?", (d,)
                )
                date_counts[d] = cur.fetchone()[0]
            except Exception:
                date_counts[d] = 0

    for i, d in enumerate(dates, 1):
        print(f"  [{i:3d}] {d}  ({date_counts.get(d, 0)} 条)")

    print()
    print("  选择方式:")
    print("    输入编号:      1 或 1,3,5 (逗号分隔)")
    print("    输入范围:      1-5")
    print("    输入日期:      2026-03-18")
    print("    输入日期范围:  2026-03-15:2026-03-18")
    print("    输入 a:        导出全部")
    print("    输入 q:        取消")
    print()

    try:
        choice = input("  请选择: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  取消导出。")
        return

    if not choice or choice.lower() == "q":
        print("  取消导出。")
        return

    # ── Parse user selection ──────────────────────────────────────
    selected_dates: List[str] = []
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    if choice.lower() == "a":
        # Export all
        selected_dates = dates
    elif ":" in choice and "-" in choice.split(":")[0]:
        # Date range: 2026-03-15:2026-03-18
        parts = choice.split(":")
        if len(parts) == 2:
            start_time = parts[0].strip() + "T00:00:00"
            end_time = parts[1].strip() + "T23:59:59"
            selected_dates = [d for d in dates if parts[0].strip() <= d <= parts[1].strip()]
    elif "-" in choice and not choice[0].isdigit():
        # Could be a single date like 2026-03-18
        if len(choice) == 10:
            selected_dates = [choice] if choice in dates else []
    elif "-" in choice and all(c.isdigit() or c == "-" for c in choice.replace(" ", "")):
        # Check: is it a number range (1-5) or a date?
        if len(choice) <= 5:
            # Number range
            parts = choice.split("-")
            if len(parts) == 2:
                try:
                    a, b = int(parts[0]), int(parts[1])
                    selected_dates = [dates[i - 1] for i in range(a, b + 1) if 1 <= i <= len(dates)]
                except (ValueError, IndexError):
                    pass
        else:
            # Likely a date
            selected_dates = [choice] if choice in dates else []
    elif "," in choice:
        # Multiple indices: 1,3,5
        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            selected_dates = [dates[i - 1] for i in indices if 1 <= i <= len(dates)]
        except (ValueError, IndexError):
            pass
    else:
        # Single index
        try:
            idx = int(choice)
            if 1 <= idx <= len(dates):
                selected_dates = [dates[idx - 1]]
        except ValueError:
            # Try as date string
            if choice in dates:
                selected_dates = [choice]

    if not selected_dates and not start_time:
        print("  无效的选择。")
        return

    # ── Build time filters ────────────────────────────────────────
    if not start_time:
        start_time = min(selected_dates) + "T00:00:00"
        end_time = max(selected_dates) + "T23:59:59"

    # ── Export ────────────────────────────────────────────────────
    date_label = selected_dates[0] if len(selected_dates) == 1 else f"{selected_dates[0]}_to_{selected_dates[-1]}"
    ts = time.strftime("%H%M%S")
    output_path = Path(base_dir) / f"export_{date_label}_{ts}.csv"

    total = 0
    for d in selected_dates:
        with db._lock:
            try:
                cur = db._conn.execute(
                    "SELECT COUNT(*) FROM test_records WHERE DATE(timestamp) = ?", (d,)
                )
                total += cur.fetchone()[0]
            except Exception:
                pass

    print(f"\n  导出 {len(selected_dates)} 天, 共约 {total} 条记录...")

    count = export_to_csv(db, output_path, start_time=start_time, end_time=end_time)

    if count > 0:
        print(f"  导出成功: {count} 条记录 → {output_path}")
    else:
        print("  没有匹配的记录。")


def export_to_csv(
    db: DatabaseLogger,
    output_path: str | Path,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    cavity_id: Optional[int] = None,
    include_raw: bool = True,
) -> int:
    """Export records to CSV.

    v2.6 perf-fix E: replaced the v2.5 N+1 query pattern (one
    query_records + N query_record_detail round trips) with a single
    streaming SELECT covering every column we need. For 10k records
    this drops export time from ~30s to <1s and stops hammering the
    DB write lock during export.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "id", "batch_id", "cavity_id", "timestamp", "label", "probability",
        "confidence", "model_version", "duration_s", "point_count",
        # v2.6 columns — Q regression results + data-quality flags +
        # active product / cycle profile IDs. Required for diagnostics
        # and for prepare_q_data when re-deriving training labels.
        "q_est", "q_threshold", "q_uncertainty",
        "m1_q", "m2_q", "m_disagreement",
        "product_id", "cycle_profile_id", "quality_flags",
    ]
    raw_cols = ["pressure_data", "angle_data", "features",
                "leak_valve_status", "end_angle"]
    header = base_cols + raw_cols if include_raw else list(base_cols)

    # Build one parametrized SELECT covering exactly the columns we'll write.
    select_cols = list(header)
    if include_raw:
        # Compressed BLOBs come along for the ride so we can decompress
        # in-process without a second round-trip.
        select_cols += ["pressure_data_compressed", "angle_data_compressed"]

    clauses, params = [], []
    if start_time:
        clauses.append("timestamp >= ?"); params.append(start_time)
    if end_time:
        clauses.append("timestamp <= ?"); params.append(end_time)
    if cavity_id is not None:
        clauses.append("cavity_id = ?"); params.append(cavity_id)
    where = " AND ".join(clauses) if clauses else "1=1"
    sql = (f"SELECT {','.join(select_cols)} FROM test_records "
           f"WHERE {where} ORDER BY id")

    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for record in db.iter_records_raw(sql, params):
            row = {k: record.get(k, "") for k in header}
            if include_raw:
                # Surface decompressed curves under the legacy text columns
                # so the CSV format remains v2.5-compatible for trainers.
                if not row.get("pressure_data") and record.get("pressure_data_compressed"):
                    arr = decompress_float_array(record["pressure_data_compressed"])
                    row["pressure_data"] = json.dumps(arr) if arr is not None else ""
                if not row.get("angle_data") and record.get("angle_data_compressed"):
                    arr = decompress_float_array(record["angle_data_compressed"])
                    row["angle_data"] = json.dumps(arr) if arr is not None else ""
            writer.writerow(row)
            count += 1

    logger.info("Exported %d records to %s", count, output_path)
    return count
