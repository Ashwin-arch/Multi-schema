"""Shared helper for appending measured resource-usage rows.

Each pipeline stage (data generation, schema build, benchmark) records only the
measurements it actually performs. Columns owned by other stages are preserved
so that repeated stage runs never erase previously measured values.
"""

import os

import pandas as pd

RESULTS_DIR = "results"
RESOURCE_FILE = os.path.join(RESULTS_DIR, "resource_usage.csv")

COLUMNS = [
    "dataset_size",
    "scale_label",
    "generation_time_s",
    "generation_size_mb",
    "db_build_time_s",
    "total_db_size_mb",
    "benchmark_execution_time_s",
    "peak_ram_mb",
]


def get_scale_label(rows):
    if rows >= 1_000_000:
        return f"{rows // 1_000_000}M"
    return f"{rows // 1_000}K"


def record_stage_resource(rows, **measurements):
    """Upsert measured values for one scale without clobbering other stages.

    Args:
        rows: fact-table row count identifying the scale.
        measurements: subset of COLUMNS to write. Values of None are ignored,
            so a stage only overwrites the columns it actually measured.
    """
    unknown = set(measurements) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown resource columns: {sorted(unknown)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if os.path.exists(RESOURCE_FILE):
        existing = pd.read_csv(RESOURCE_FILE)
        mask = existing["dataset_size"] == int(rows)
        if mask.any():
            # Update only the columns this stage measured
            for col, value in measurements.items():
                if col in COLUMNS and value is not None:
                    existing.loc[mask, col] = round(float(value), 2)
            combined = existing
        else:
            # New row - fill unmeasured columns with NaN
            row = {"dataset_size": int(rows), "scale_label": get_scale_label(int(rows))}
            for col in COLUMNS[2:]:
                value = measurements.get(col)
                row[col] = round(float(value), 2) if value is not None else float("nan")
            combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        row = {"dataset_size": int(rows), "scale_label": get_scale_label(int(rows))}
        for col in COLUMNS[2:]:
            value = measurements.get(col)
            row[col] = round(float(value), 2) if value is not None else float("nan")
        combined = pd.DataFrame([row])

    for col in COLUMNS:
        if col not in combined.columns:
            combined[col] = float("nan")

    combined = combined[COLUMNS]
    combined.sort_values("dataset_size", inplace=True)
    combined.to_csv(RESOURCE_FILE, index=False)
    return combined[combined["dataset_size"] == int(rows)].iloc[0].to_dict()