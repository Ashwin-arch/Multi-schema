import json
import os
import random
import re
import time

import duckdb
import numpy as np
import pandas as pd


WORKLOAD_FILE = "data/processed/workload_v3.json"

V6_RESULT_FILE = (
    "data/processed/v6_relative_cost_results.csv"
)

OUTPUT_FILE = (
    "data/processed/v6_repeatability_raw.csv"
)

SUMMARY_FILE = (
    "data/processed/v6_repeatability_summary.csv"
)

QUERY_SUMMARY_FILE = (
    "data/processed/v6_repeatability_query_summary.csv"
)

N_REPEATS = 10
RANDOM_SEED = 42

DB_PATHS = {
    "star": "data/processed/star_schema.duckdb",
    "snowflake": "data/processed/snowflake_schema.duckdb",
    "fact_constellation":
        "data/processed/fact_constellation.duckdb",
}


def adapt_snowflake_sql(sql):
    """
    Adapt common SQL to the Snowflake-compatible views.
    """

    sql = re.sub(
        r"\bdim_customer\s+c\b",
        "vw_customer c",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"\bdim_product\s+p\b",
        "vw_product p",
        sql,
        flags=re.IGNORECASE,
    )

    return sql


def adapt_sql(sql, schema):

    # Common compatibility correction.
    sql = re.sub(
        r"\bfs\.sales_amount\b",
        "fs.total_amount",
        sql,
        flags=re.IGNORECASE,
    )

    if schema == "snowflake":
        sql = adapt_snowflake_sql(sql)

    return sql


def execute_timed(con, sql):

    start = time.perf_counter()

    result = con.execute(sql)

    # Force complete materialization so that the timing
    # includes execution rather than only query submission.
    result.fetchall()

    end = time.perf_counter()

    return (end - start) * 1000.0


def bootstrap_ci(values, n_bootstrap=10000, seed=42):

    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    n = len(values)

    if n == 0:
        return np.nan, np.nan

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float
    )

    # Process in batches to avoid excessive memory use.
    batch_size = 1000

    written = 0

    while written < n_bootstrap:

        batch = min(
            batch_size,
            n_bootstrap - written
        )

        indices = rng.integers(
            0,
            n,
            size=(batch, n)
        )

        bootstrap_means[
            written:written + batch
        ] = values[indices].mean(axis=1)

        written += batch

    return (
        float(np.percentile(
            bootstrap_means,
            2.5
        )),
        float(np.percentile(
            bootstrap_means,
            97.5
        )),
    )


def main():

    print("=" * 80)
    print("V6 REPEATABILITY AND STATISTICAL VALIDATION")
    print("=" * 80)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ---------------------------------------------------------
    # Load workload.
    # ---------------------------------------------------------

    with open(WORKLOAD_FILE) as f:
        workload = json.load(f)

    workload_df = pd.DataFrame(workload)

    print(
        f"\nWorkload queries : "
        f"{len(workload_df)}"
    )

    # ---------------------------------------------------------
    # Load V6 predictions.
    # ---------------------------------------------------------

    v6_df = pd.read_csv(
        V6_RESULT_FILE
    )

    required_v6_columns = [
        "query_id",
        "workload",
        "predicted_schema",
    ]

    missing = [
        c for c in required_v6_columns
        if c not in v6_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing V6 columns: {missing}"
        )

    controller_df = workload_df.merge(
        v6_df[
            required_v6_columns
        ],
        on=[
            "query_id",
            "workload",
        ],
        how="inner",
    )

    if len(controller_df) != len(workload_df):

        missing_ids = set(
            workload_df["query_id"]
        ) - set(
            controller_df["query_id"]
        )

        raise ValueError(
            "V6 prediction file does not contain "
            f"all workload queries. Missing: "
            f"{sorted(missing_ids)}"
        )

    print(
        f"V6 controller rows : "
        f"{len(controller_df)}"
    )

    print("\nV6 selected schema distribution:")
    print(
        controller_df[
            "predicted_schema"
        ]
        .value_counts()
        .reindex(
            [
                "star",
                "snowflake",
                "fact_constellation",
            ],
            fill_value=0,
        )
    )

    # ---------------------------------------------------------
    # Open persistent DuckDB connections.
    # ---------------------------------------------------------

    connections = {}

    print("\nOpening DuckDB databases...")

    for schema, path in DB_PATHS.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Database not found: {path}"
            )

        connections[schema] = (
            duckdb.connect(
                path,
                read_only=True
            )
        )

        print(
            f"  {schema:20s} -> {path}"
        )

    # ---------------------------------------------------------
    # Warm-up.
    #
    # One execution per query/schema.
    # These are NOT included in measurements.
    # ---------------------------------------------------------

    print("\nPerforming warm-up executions...")

    for _, row in controller_df.iterrows():

        sql = row["sql"]

        for schema in DB_PATHS:

            schema_sql = adapt_sql(
                sql,
                schema
            )

            connections[schema].execute(
                schema_sql
            ).fetchall()

    print("Warm-up complete.")

    # ---------------------------------------------------------
    # Timed benchmark.
    # ---------------------------------------------------------

    total_expected = (
        len(controller_df)
        * len(DB_PATHS)
        * N_REPEATS
    )

    print("\nTimed executions:")
    print(
        f"  Queries       : "
        f"{len(controller_df)}"
    )

    print(
        f"  Schemas       : "
        f"{len(DB_PATHS)}"
    )

    print(
        f"  Repetitions   : "
        f"{N_REPEATS}"
    )

    print(
        f"  Expected runs : "
        f"{total_expected}"
    )

    print()

    raw_results = []

    completed = 0

    start_total = time.perf_counter()

    # ---------------------------------------------------------
    # Query loop.
    #
    # Schema execution order is randomized for every
    # repetition to reduce systematic ordering effects.
    # ---------------------------------------------------------

    for query_index, (_, row) in enumerate(
        controller_df.iterrows(),
        start=1
    ):

        query_id = row["query_id"]
        workload_name = row["workload"]
        sql = row["sql"]
        adaptive_schema = row[
            "predicted_schema"
        ]

        schemas = list(DB_PATHS.keys())

        for repetition in range(
            1,
            N_REPEATS + 1
        ):

            random.shuffle(schemas)

            for schema in schemas:

                schema_sql = adapt_sql(
                    sql,
                    schema
                )

                try:

                    latency = execute_timed(
                        connections[schema],
                        schema_sql
                    )

                    raw_results.append({
                        "query_id": query_id,
                        "workload":
                            workload_name,
                        "repetition":
                            repetition,
                        "schema":
                            schema,
                        "adaptive_schema":
                            adaptive_schema,
                        "latency_ms":
                            latency,
                        "success":
                            1,
                    })

                except Exception as exc:

                    raw_results.append({
                        "query_id": query_id,
                        "workload":
                            workload_name,
                        "repetition":
                            repetition,
                        "schema":
                            schema,
                        "adaptive_schema":
                            adaptive_schema,
                        "latency_ms":
                            np.nan,
                        "success":
                            0,
                        "error":
                            str(exc),
                    })

                completed += 1

            if (
                query_index == 1
                or query_index % 50 == 0
                or query_index == len(
                    controller_df
                )
            ):

                elapsed = (
                    time.perf_counter()
                    - start_total
                )

                rate = (
                    completed / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"\rQueries "
                    f"{query_index:3d}/"
                    f"{len(controller_df)} | "
                    f"Executions "
                    f"{completed:5d}/"
                    f"{total_expected} | "
                    f"{rate:.1f} exec/s",
                    end="",
                    flush=True,
                )

    print("\n\nBenchmark complete.")

    raw_df = pd.DataFrame(
        raw_results
    )

    # ---------------------------------------------------------
    # Validate execution success.
    # ---------------------------------------------------------

    failed = raw_df[
        raw_df["success"] == 0
    ]

    print(
        f"\nSuccessful executions : "
        f"{len(raw_df) - len(failed)}"
    )

    print(
        f"Failed executions     : "
        f"{len(failed)}"
    )

    if len(failed) > 0:

        print("\nFirst failures:")
        print(
            failed[
                [
                    "query_id",
                    "schema",
                    "error",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        raise RuntimeError(
            "Benchmark had failed executions."
        )

    # ---------------------------------------------------------
    # Save raw timing data.
    # ---------------------------------------------------------

    raw_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nRaw results written:"
        f"\n  {OUTPUT_FILE}"
    )

    # ---------------------------------------------------------
    # Per-query/schema mean and std.
    # ---------------------------------------------------------

    query_schema_stats = (
        raw_df
        .groupby(
            [
                "query_id",
                "workload",
                "adaptive_schema",
                "schema",
            ]
        )["latency_ms"]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Convert schema means to wide format.
    # ---------------------------------------------------------

    mean_wide = (
        query_schema_stats
        .pivot(
            index=[
                "query_id",
                "workload",
                "adaptive_schema",
            ],
            columns="schema",
            values="mean",
        )
        .reset_index()
    )

    mean_wide.columns.name = None

    mean_wide = mean_wide.rename(
        columns={
            "star":
                "star_mean_ms",
            "snowflake":
                "snowflake_mean_ms",
            "fact_constellation":
                "fc_mean_ms",
        }
    )

    # ---------------------------------------------------------
    # Median wide.
    # ---------------------------------------------------------

    median_wide = (
        query_schema_stats
        .pivot(
            index=[
                "query_id",
                "workload",
                "adaptive_schema",
            ],
            columns="schema",
            values="median",
        )
        .reset_index()
    )

    median_wide.columns.name = None

    median_wide = median_wide.rename(
        columns={
            "star":
                "star_median_ms",
            "snowflake":
                "snowflake_median_ms",
            "fact_constellation":
                "fc_median_ms",
        }
    )

    # ---------------------------------------------------------
    # Standard deviation wide.
    # ---------------------------------------------------------

    std_wide = (
        query_schema_stats
        .pivot(
            index=[
                "query_id",
                "workload",
                "adaptive_schema",
            ],
            columns="schema",
            values="std",
        )
        .reset_index()
    )

    std_wide.columns.name = None

    std_wide = std_wide.rename(
        columns={
            "star":
                "star_std_ms",
            "snowflake":
                "snowflake_std_ms",
            "fact_constellation":
                "fc_std_ms",
        }
    )

    query_stats = (
        mean_wide
        .merge(
            median_wide,
            on=[
                "query_id",
                "workload",
                "adaptive_schema",
            ],
        )
        .merge(
            std_wide,
            on=[
                "query_id",
                "workload",
                "adaptive_schema",
            ],
        )
    )

    # ---------------------------------------------------------
    # Oracle based on each query's mean latency.
    # ---------------------------------------------------------

    actual_means = query_stats[
        [
            "star_mean_ms",
            "snowflake_mean_ms",
            "fc_mean_ms",
        ]
    ].values

    oracle_indices = np.argmin(
        actual_means,
        axis=1
    )

    schema_names = [
        "star",
        "snowflake",
        "fact_constellation",
    ]

    query_stats[
        "oracle_schema"
    ] = [
        schema_names[i]
        for i in oracle_indices
    ]

    query_stats[
        "oracle_mean_ms"
    ] = np.min(
        actual_means,
        axis=1
    )

    # ---------------------------------------------------------
    # Adaptive mean latency.
    # ---------------------------------------------------------

    adaptive_values = []

    for _, row in query_stats.iterrows():

        adaptive_values.append(
            row[
                f"{row['adaptive_schema']}_mean_ms"
                if row["adaptive_schema"] !=
                "fact_constellation"
                else "fc_mean_ms"
            ]
        )

    query_stats[
        "adaptive_mean_ms"
    ] = adaptive_values

    query_stats[
        "regret_ms"
    ] = (
        query_stats["adaptive_mean_ms"]
        - query_stats["oracle_mean_ms"]
    )

    query_stats[
        "oracle_match"
    ] = (
        query_stats[
            "adaptive_schema"
        ]
        == query_stats[
            "oracle_schema"
        ]
    ).astype(int)

    # Best fixed means are computed globally later.
    query_stats.to_csv(
        QUERY_SUMMARY_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Global fixed-schema means.
    # ---------------------------------------------------------

    star_mean = query_stats[
        "star_mean_ms"
    ].mean()

    snowflake_mean = query_stats[
        "snowflake_mean_ms"
    ].mean()

    fc_mean = query_stats[
        "fc_mean_ms"
    ].mean()

    adaptive_mean = query_stats[
        "adaptive_mean_ms"
    ].mean()

    oracle_mean = query_stats[
        "oracle_mean_ms"
    ].mean()

    best_fixed_mean = min(
        star_mean,
        snowflake_mean,
        fc_mean
    )

    # ---------------------------------------------------------
    # Bootstrap paired confidence interval.
    #
    # Positive difference means Adaptive is faster.
    # ---------------------------------------------------------

    adaptive_minus_star = (
        query_stats["star_mean_ms"]
        - query_stats["adaptive_mean_ms"]
    ).values

    adaptive_minus_best_fixed = (
        np.minimum(
            np.minimum(
                query_stats[
                    "star_mean_ms"
                ].values,
                query_stats[
                    "snowflake_mean_ms"
                ].values,
            ),
            query_stats[
                "fc_mean_ms"
            ].values,
        )
        - query_stats[
            "adaptive_mean_ms"
        ].values
    )

    ci_star = bootstrap_ci(
        adaptive_minus_star
    )

    ci_best = bootstrap_ci(
        adaptive_minus_best_fixed
    )

    # ---------------------------------------------------------
    # Overall percentage improvements.
    # ---------------------------------------------------------

    adaptive_vs_star = (
        (star_mean - adaptive_mean)
        / star_mean
        * 100
    )

    adaptive_vs_best = (
        (best_fixed_mean - adaptive_mean)
        / best_fixed_mean
        * 100
    )

    oracle_vs_star = (
        (star_mean - oracle_mean)
        / star_mean
        * 100
    )

    adaptive_gap = (
        (adaptive_mean - oracle_mean)
        / oracle_mean
        * 100
    )

    mean_regret = query_stats[
        "regret_ms"
    ].mean()

    median_regret = query_stats[
        "regret_ms"
    ].median()

    match_rate = (
        query_stats[
            "oracle_match"
        ].mean()
        * 100
    )

    # ---------------------------------------------------------
    # Repeatability across executions.
    # ---------------------------------------------------------

    overall_std_by_schema = (
        raw_df
        .groupby("schema")[
            "latency_ms"
        ]
        .std()
    )

    # ---------------------------------------------------------
    # Per-workload summary.
    # ---------------------------------------------------------

    workload_rows = []

    for workload_name, group in (
        query_stats
        .groupby("workload")
    ):

        star = group[
            "star_mean_ms"
        ].mean()

        snowflake = group[
            "snowflake_mean_ms"
        ].mean()

        fc = group[
            "fc_mean_ms"
        ].mean()

        adaptive = group[
            "adaptive_mean_ms"
        ].mean()

        oracle = group[
            "oracle_mean_ms"
        ].mean()

        best_fixed = min(
            star,
            snowflake,
            fc
        )

        workload_rows.append({
            "workload":
                workload_name,

            "queries":
                len(group),

            "star_mean_ms":
                star,

            "snowflake_mean_ms":
                snowflake,

            "fc_mean_ms":
                fc,

            "best_fixed_ms":
                best_fixed,

            "adaptive_mean_ms":
                adaptive,

            "oracle_mean_ms":
                oracle,

            "adaptive_vs_best_fixed_percent":
                (
                    (best_fixed - adaptive)
                    / best_fixed
                    * 100
                ),

            "oracle_vs_star_percent":
                (
                    (star - oracle)
                    / star
                    * 100
                ),

            "adaptive_gap_to_oracle_percent":
                (
                    (adaptive - oracle)
                    / oracle
                    * 100
                ),

            "mean_regret_ms":
                group[
                    "regret_ms"
                ].mean(),

            "oracle_match_rate_percent":
                group[
                    "oracle_match"
                ].mean()
                * 100,
        })

    workload_summary = pd.DataFrame(
        workload_rows
    )

    workload_summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Print final report.
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("V6 REPEATABILITY RESULTS")
    print("=" * 80)

    print(
        f"\nTimed executions : "
        f"{len(raw_df)}"
    )

    print(
        f"Queries          : "
        f"{len(query_stats)}"
    )

    print(
        f"Repetitions/query : "
        f"{N_REPEATS}"
    )

    print("\nAverage latency across query means:")

    print(
        f"  Fixed Star              : "
        f"{star_mean:.6f} ms"
    )

    print(
        f"  Fixed Snowflake         : "
        f"{snowflake_mean:.6f} ms"
    )

    print(
        f"  Fixed Fact Constellation: "
        f"{fc_mean:.6f} ms"
    )

    print(
        f"  Best Fixed              : "
        f"{best_fixed_mean:.6f} ms"
    )

    print(
        f"  V6 Adaptive             : "
        f"{adaptive_mean:.6f} ms"
    )

    print(
        f"  Oracle                  : "
        f"{oracle_mean:.6f} ms"
    )

    print("\nPerformance:")

    print(
        f"  Adaptive vs Star        : "
        f"{adaptive_vs_star:.3f}%"
    )

    print(
        f"  Adaptive vs Best Fixed  : "
        f"{adaptive_vs_best:.3f}%"
    )

    print(
        f"  Oracle vs Star          : "
        f"{oracle_vs_star:.3f}%"
    )

    print(
        f"  Adaptive gap to Oracle  : "
        f"{adaptive_gap:.3f}%"
    )

    print("\n95% Bootstrap CI for latency advantage:")

    print(
        "  Adaptive - Star "
        "(positive = adaptive faster):"
    )

    print(
        f"    [{ci_star[0]:.6f}, "
        f"{ci_star[1]:.6f}] ms"
    )

    print(
        "  Adaptive - Best Fixed "
        "(positive = adaptive faster):"
    )

    print(
        f"    [{ci_best[0]:.6f}, "
        f"{ci_best[1]:.6f}] ms"
    )

    print("\nRegret:")

    print(
        f"  Mean regret             : "
        f"{mean_regret:.6f} ms"
    )

    print(
        f"  Median regret           : "
        f"{median_regret:.6f} ms"
    )

    mean_regret_percent = (
        query_stats["regret_ms"]
        .div(query_stats["oracle_mean_ms"])
        .mean()
        * 100
    )

    print(
        f"  Mean regret percent     : "
        f"{mean_regret_percent:.3f}%"
    )

    print(
        f"  Oracle match rate       : "
        f"{match_rate:.2f}%"
    )

    print("\nOverall execution variability:")

    for schema in [
        "star",
        "snowflake",
        "fact_constellation",
    ]:

        print(
            f"  {schema:20s}: "
            f"std={overall_std_by_schema[schema]:.6f} ms"
        )

    print("\nV6 schema selection distribution:")

    print(
        query_stats[
            "adaptive_schema"
        ]
        .value_counts()
        .reindex(
            schema_names,
            fill_value=0
        )
    )

    print("\nOracle distribution based on query mean:")

    print(
        query_stats[
            "oracle_schema"
        ]
        .value_counts()
        .reindex(
            schema_names,
            fill_value=0
        )
    )

    print("\nPer-workload results:")
    print(
        workload_summary.to_string(
            index=False
        )
    )

    print("\nFiles written:")

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {QUERY_SUMMARY_FILE}"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print("\nRepeatability evaluation complete.")


if __name__ == "__main__":
    main()
