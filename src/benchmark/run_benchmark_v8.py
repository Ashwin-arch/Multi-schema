import argparse
import json
import os
import random
import resource
import sys
import time
import duckdb
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.generate_data_v7 import get_scale_label
from src.schemas.build_schemas_v7 import get_db_paths
from src.utils.resource_logging import record_stage_resource
from src.workload.validate_workload_v8 import adapt_sql_v8, get_applicable_schemas

WORKLOAD_FILE = "data/processed/workload_v8.json"
RESULTS_DIR = "results/v8"
RAW_RESULTS_FILE = os.path.join(RESULTS_DIR, "scale_raw_results.csv")
QUERY_SUMMARY_FILE = os.path.join(RESULTS_DIR, "scale_query_summary.csv")
SCALE_SUMMARY_FILE = os.path.join(RESULTS_DIR, "scale_summary.csv")
RESOURCE_FILE = os.path.join(RESULTS_DIR, "resource_usage.csv")


def execute_timed(con, sql):
    """Execute query with high-resolution timer and forced full materialization."""
    start = time.perf_counter()
    con.execute(sql).fetchall()
    end = time.perf_counter()
    return (end - start) * 1000.0


def bootstrap_ci(values, n_bootstrap=5000, seed=42):
    """Compute 95% bootstrap confidence interval of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_benchmark_scale(
    scale_rows=1_000_000,
    n_repeats=3,
    n_warmup=1,
    random_seed=42,
    workload_file=WORKLOAD_FILE,
    db_paths=None,
    append_results=True
):
    """
    Run multi-schema benchmark on a specific dataset scale for V8 workload.
    Only runs queries on applicable schemas (per get_applicable_schemas).
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    scale_label = get_scale_label(scale_rows)
    if db_paths is None:
        db_paths = get_db_paths(scale_label)

    print("=" * 75)
    print(f"V8 BENCHMARK EXECUTION - SCALE {scale_label} ({scale_rows:,} rows)")
    print("=" * 75)

    with open(workload_file, "r") as f:
        workload = json.load(f)

    # Open read-only connections
    connections = {}
    db_sizes_mb = {}
    for schema, path in db_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Database not found: {path}")
        connections[schema] = duckdb.connect(path, read_only=True)
        db_sizes_mb[schema] = os.path.getsize(path) / (1024 * 1024)

    # 1. Warm-up (only on applicable schemas)
    print(f"Warm-up runs ({n_warmup} per query/schema, not timed)...")
    for item in workload:
        sql = item["sql"]
        applicable = get_applicable_schemas(sql)
        for schema in applicable:
            schema_sql = adapt_sql_v8(sql, schema)
            for _ in range(n_warmup):
                connections[schema].execute(schema_sql).fetchall()

    print("Warm-up complete. Starting timed benchmark...")

    schemas = ["star", "snowflake", "fact_constellation"]
    raw_records = []
    total_expected = sum(
        len(get_applicable_schemas(item["sql"])) * n_repeats
        for item in workload
    )
    completed = 0
    t0_benchmark = time.perf_counter()

    for q_idx, item in enumerate(workload, start=1):
        qid = item["query_id"]
        workload_type = item["workload"]
        sql = item["sql"]
        applicable = get_applicable_schemas(sql)

        for rep in range(1, n_repeats + 1):
            # Randomized schema execution order per repetition (only applicable schemas)
            rep_schemas = list(applicable)
            random.shuffle(rep_schemas)

            for schema in rep_schemas:
                schema_sql = adapt_sql_v8(sql, schema)
                try:
                    latency = execute_timed(connections[schema], schema_sql)
                    raw_records.append({
                        "query_id": qid,
                        "dataset_size": scale_rows,
                        "scale_label": scale_label,
                        "workload": workload_type,
                        "schema": schema,
                        "repetition": rep,
                        "latency_ms": latency,
                        "success": 1,
                        "error": ""
                    })
                except Exception as e:
                    raw_records.append({
                        "query_id": qid,
                        "dataset_size": scale_rows,
                        "scale_label": scale_label,
                        "workload": workload_type,
                        "schema": schema,
                        "repetition": rep,
                        "latency_ms": np.nan,
                        "success": 0,
                        "error": str(e)
                    })
                completed += 1

        if q_idx % 50 == 0 or q_idx == len(workload):
            elapsed_so_far = time.perf_counter() - t0_benchmark
            rate = completed / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"\r  Progress: {completed:,}/{total_expected:,} executions ({rate:.1f} exec/s)...", end="", flush=True)

    benchmark_duration_s = time.perf_counter() - t0_benchmark
    print(f"\nCompleted {completed:,} executions in {benchmark_duration_s:.2f}s.")

    for con in connections.values():
        con.close()

    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    total_db_size_mb = sum(db_sizes_mb.values())
    print(f"Total DuckDB storage: {total_db_size_mb:.2f} MB | Peak process RSS: {peak_ram_mb:.1f} MB")
    record_resource_usage(
        dataset_size=scale_rows,
        scale_label=scale_label,
        db_size_mb=total_db_size_mb,
        benchmark_time_s=benchmark_duration_s,
        peak_ram_mb=peak_ram_mb,
    )

    new_raw_df = pd.DataFrame(raw_records)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save / append raw results
    if append_results and os.path.exists(RAW_RESULTS_FILE):
        existing_raw = pd.read_csv(RAW_RESULTS_FILE)
        existing_raw = existing_raw[existing_raw["dataset_size"] != scale_rows]
        combined_raw = pd.concat([existing_raw, new_raw_df], ignore_index=True)
        combined_raw.to_csv(RAW_RESULTS_FILE, index=False)
    else:
        new_raw_df.to_csv(RAW_RESULTS_FILE, index=False)

    # -------------------------------------------------------------
    # Compute Query-Level Summary for this scale
    # -------------------------------------------------------------
    successful_df = new_raw_df[new_raw_df["success"] == 1]
    query_stats = (
        successful_df.groupby(["query_id", "dataset_size", "scale_label", "workload", "schema"])["latency_ms"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )

    # Pivot to wide format (only includes schemas that were actually run)
    mean_wide = query_stats.pivot(
        index=["query_id", "dataset_size", "scale_label", "workload"],
        columns="schema",
        values="mean"
    ).reset_index()

    # Rename columns if they exist
    rename_map = {}
    if "star" in mean_wide.columns:
        rename_map["star"] = "star_mean_ms"
    if "snowflake" in mean_wide.columns:
        rename_map["snowflake"] = "snowflake_mean_ms"
    if "fact_constellation" in mean_wide.columns:
        rename_map["fact_constellation"] = "fc_mean_ms"
    mean_wide = mean_wide.rename(columns=rename_map)

    median_wide = query_stats.pivot(
        index=["query_id", "dataset_size", "scale_label", "workload"],
        columns="schema",
        values="median"
    ).reset_index().rename(columns={
        "star": "star_median_ms",
        "snowflake": "snowflake_median_ms",
        "fact_constellation": "fc_median_ms"
    })

    std_wide = query_stats.pivot(
        index=["query_id", "dataset_size", "scale_label", "workload"],
        columns="schema",
        values="std"
    ).reset_index().rename(columns={
        "star": "star_std_ms",
        "snowflake": "snowflake_std_ms",
        "fact_constellation": "fc_std_ms"
    })

    # Start with mean_wide and merge others
    scale_query_summary = mean_wide.merge(
        median_wide, on=["query_id", "dataset_size", "scale_label", "workload"], how="outer"
    ).merge(
        std_wide, on=["query_id", "dataset_size", "scale_label", "workload"], how="outer"
    )

    # Oracle computation per query (only for schemas that have data)
    schema_cols = [c for c in ["star_mean_ms", "snowflake_mean_ms", "fc_mean_ms"] if c in scale_query_summary.columns]
    if schema_cols:
        actual_means = scale_query_summary[schema_cols].values
        oracle_indices = np.argmin(actual_means, axis=1)
        schema_map = {i: col.replace("_mean_ms", "") for i, col in enumerate(schema_cols)}
        scale_query_summary["oracle_schema"] = [schema_map[i] for i in oracle_indices]
        scale_query_summary["oracle_mean_ms"] = np.min(actual_means, axis=1)
    else:
        scale_query_summary["oracle_schema"] = ""
        scale_query_summary["oracle_mean_ms"] = np.nan

    # Save / append query summary
    if append_results and os.path.exists(QUERY_SUMMARY_FILE):
        existing_qs = pd.read_csv(QUERY_SUMMARY_FILE)
        existing_qs = existing_qs[existing_qs["dataset_size"] != scale_rows]
        combined_qs = pd.concat([existing_qs, scale_query_summary], ignore_index=True)
        combined_qs.to_csv(QUERY_SUMMARY_FILE, index=False)
    else:
        scale_query_summary.to_csv(QUERY_SUMMARY_FILE, index=False)

    # -------------------------------------------------------------
    # Compute Scale-Level Aggregates
    # -------------------------------------------------------------
    star_mean = float(scale_query_summary["star_mean_ms"].mean()) if "star_mean_ms" in scale_query_summary.columns else np.nan
    snow_mean = float(scale_query_summary["snowflake_mean_ms"].mean()) if "snowflake_mean_ms" in scale_query_summary.columns else np.nan
    fc_mean = float(scale_query_summary["fc_mean_ms"].mean()) if "fc_mean_ms" in scale_query_summary.columns else np.nan
    oracle_mean = float(scale_query_summary["oracle_mean_ms"].mean()) if "oracle_mean_ms" in scale_query_summary.columns else np.nan

    star_median = float(scale_query_summary["star_median_ms"].median()) if "star_median_ms" in scale_query_summary.columns else np.nan
    snow_median = float(scale_query_summary["snowflake_median_ms"].median()) if "snowflake_median_ms" in scale_query_summary.columns else np.nan
    fc_median = float(scale_query_summary["fc_median_ms"].median()) if "fc_median_ms" in scale_query_summary.columns else np.nan

    # Best fixed among available schemas
    schema_means = {}
    if "star_mean_ms" in scale_query_summary.columns:
        schema_means["star"] = star_mean
    if "snowflake_mean_ms" in scale_query_summary.columns:
        schema_means["snowflake"] = snow_mean
    if "fc_mean_ms" in scale_query_summary.columns:
        schema_means["fact_constellation"] = fc_mean
    
    if schema_means:
        best_fixed_name = min(schema_means, key=schema_means.get)
        best_fixed_mean = schema_means[best_fixed_name]
    else:
        best_fixed_name = "star"
        best_fixed_mean = np.nan

    oracle_vs_star_pct = ((star_mean - oracle_mean) / star_mean) * 100.0 if not np.isnan(star_mean) and not np.isnan(oracle_mean) and star_mean != 0 else np.nan
    oracle_vs_best_pct = ((best_fixed_mean - oracle_mean) / best_fixed_mean) * 100.0 if not np.isnan(best_fixed_mean) and not np.isnan(oracle_mean) and best_fixed_mean != 0 else np.nan

    # Schema distribution in Oracle
    if "oracle_schema" in scale_query_summary.columns:
        oracle_counts = scale_query_summary["oracle_schema"].value_counts().to_dict()
        total_q = len(scale_query_summary)
        star_pct = (oracle_counts.get("star", 0) / total_q) * 100.0
        snow_pct = (oracle_counts.get("snowflake", 0) / total_q) * 100.0
        fc_pct = (oracle_counts.get("fact_constellation", 0) / total_q) * 100.0
    else:
        star_pct = snow_pct = fc_pct = 0.0

    # Bootstrap CI for Star vs Oracle advantage
    if "star_mean_ms" in scale_query_summary.columns and "oracle_mean_ms" in scale_query_summary.columns:
        ci_star_oracle = bootstrap_ci(
            scale_query_summary["star_mean_ms"].values - scale_query_summary["oracle_mean_ms"].values
        )
    else:
        ci_star_oracle = (np.nan, np.nan)

    scale_stat_row = {
        "dataset_size": scale_rows,
        "scale_label": scale_label,
        "queries": len(scale_query_summary),
        "total_executions": completed,
        "star_mean_ms": star_mean,
        "snowflake_mean_ms": snow_mean,
        "fc_mean_ms": fc_mean,
        "best_fixed_schema": best_fixed_name,
        "best_fixed_mean_ms": best_fixed_mean,
        "oracle_mean_ms": oracle_mean,
        "oracle_vs_star_percent": oracle_vs_star_pct,
        "oracle_vs_best_fixed_percent": oracle_vs_best_pct,
        "ci_star_oracle_lower_ms": ci_star_oracle[0],
        "ci_star_oracle_upper_ms": ci_star_oracle[1],
        "oracle_star_share_pct": star_pct,
        "oracle_snowflake_share_pct": snow_pct,
        "oracle_fc_share_pct": fc_pct,
        "benchmark_duration_s": benchmark_duration_s,
    }

    scale_stat_df = pd.DataFrame([scale_stat_row])
    if append_results and os.path.exists(SCALE_SUMMARY_FILE):
        existing_ss = pd.read_csv(SCALE_SUMMARY_FILE)
        existing_ss = existing_ss[existing_ss["dataset_size"] != scale_rows]
        combined_ss = pd.concat([existing_ss, scale_stat_df], ignore_index=True)
        combined_ss.sort_values("dataset_size", inplace=True)
        combined_ss.to_csv(SCALE_SUMMARY_FILE, index=False)
    else:
        scale_stat_df.to_csv(SCALE_SUMMARY_FILE, index=False)

    print("\nScale Summary:")
    print(f"  Star Mean         : {star_mean:.4f} ms" if not np.isnan(star_mean) else "  Star Mean         : N/A")
    print(f"  Snowflake Mean    : {snow_mean:.4f} ms" if not np.isnan(snow_mean) else "  Snowflake Mean    : N/A")
    print(f"  FC Mean           : {fc_mean:.4f} ms" if not np.isnan(fc_mean) else "  FC Mean           : N/A")
    print(f"  Best Fixed        : {best_fixed_name} ({best_fixed_mean:.4f} ms)" if not np.isnan(best_fixed_mean) else "  Best Fixed        : N/A")
    print(f"  Oracle Mean       : {oracle_mean:.4f} ms" if not np.isnan(oracle_mean) else "  Oracle Mean       : N/A")
    print(f"  Oracle vs Star    : {oracle_vs_star_pct:.3f}% (advantage: [{ci_star_oracle[0]:.4f}, {ci_star_oracle[1]:.4f}] ms)")
    print(f"  Oracle vs Best Fix: {oracle_vs_best_pct:.3f}%")
    print(f"  Oracle Schema Distribution: Star={star_pct:.1f}%, Snow={snow_pct:.1f}%, FC={fc_pct:.1f}%")
    print("=" * 75)

    return scale_stat_row


def record_resource_usage(
    dataset_size,
    scale_label,
    db_size_mb,
    benchmark_time_s,
    peak_ram_mb=None
):
    """Record measured resource consumption for scalability characterization."""
    record_stage_resource(
        dataset_size,
        benchmark_execution_time_s=benchmark_time_s,
        total_db_size_mb=db_size_mb,
        peak_ram_mb=peak_ram_mb,
    )


def main():
    parser = argparse.ArgumentParser(description="Run V8 scalable multi-schema benchmark.")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Sales rows scale")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per query/schema")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup executions per query/schema")
    parser.add_argument("--workload", type=str, default=WORKLOAD_FILE, help="Workload JSON path")
    args = parser.parse_args()

    run_benchmark_scale(
        scale_rows=args.rows,
        n_repeats=args.repeats,
        n_warmup=args.warmup,
        workload_file=args.workload
    )


if __name__ == "__main__":
    main()