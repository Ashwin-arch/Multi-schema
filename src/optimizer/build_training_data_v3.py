import json
import re
import os

import pandas as pd


WORKLOAD_FILE = "data/processed/workload_v3.json"
BENCHMARK_FILE = "data/processed/benchmark_results_v3.csv"
OUTPUT_FILE = "data/processed/training_data_v3.csv"


def count_pattern(query, pattern):

    return len(
        re.findall(
            pattern,
            query,
            flags=re.IGNORECASE
        )
    )


def extract_query_features(query):

    joins = count_pattern(
        query,
        r"\bJOIN\b"
    )

    aggregations = count_pattern(
        query,
        r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\("
    )

    predicates = count_pattern(
        query,
        r"\b(WHERE|AND|OR)\b"
    )

    group_by_count = count_pattern(
        query,
        r"\bGROUP\s+BY\b"
    )

    tables = re.findall(
        r"\b(?:FROM|JOIN)\s+"
        r"([a-zA-Z_][a-zA-Z0-9_]*)",
        query,
        flags=re.IGNORECASE
    )

    return {
        "join_count": joins,
        "aggregation_count": aggregations,
        "predicate_count": predicates,
        "group_by_count": group_by_count,
        "tables_used": len(set(tables)),
        "query_length": len(query)
    }


def build_training_data():

    with open(WORKLOAD_FILE, "r") as file:
        workloads = json.load(file)

    feature_records = []

    for query_info in workloads:

        features = extract_query_features(
            query_info["sql"]
        )

        feature_records.append({
            "query_id": query_info["query_id"],
            "workload": query_info["workload"],
            "query_type": query_info["query_type"],
            **features
        })

    features_df = pd.DataFrame(
        feature_records
    )

    benchmark = pd.read_csv(
        BENCHMARK_FILE
    )

    benchmark = benchmark[
        benchmark["status"] == "success"
    ].copy()

    # Median latency across the 5 repetitions.
    timing = (
        benchmark
        .groupby(
            ["query_id", "schema"]
        )["time_ms"]
        .median()
        .reset_index()
    )

    timing = timing.pivot(
        index="query_id",
        columns="schema",
        values="time_ms"
    ).reset_index()

    timing = timing.rename(
        columns={
            "star": "star_time_ms",
            "snowflake": "snowflake_time_ms",
            "fact_constellation":
                "constellation_time_ms"
        }
    )

    df = features_df.merge(
        timing,
        on="query_id",
        how="inner"
    )

    time_columns = [
        "star_time_ms",
        "snowflake_time_ms",
        "constellation_time_ms"
    ]

    df["best_schema"] = (
        df[time_columns]
        .idxmin(axis=1)
    )

    df["best_schema"] = (
        df["best_schema"]
        .replace({
            "star_time_ms": "star",
            "snowflake_time_ms": "snowflake",
            "constellation_time_ms":
                "fact_constellation"
        })
    )

    df["best_time_ms"] = (
        df[time_columns]
        .min(axis=1)
    )

    # Oracle improvement over each fixed schema.
    for schema_name, column in [
        ("star", "star_time_ms"),
        ("snowflake", "snowflake_time_ms"),
        ("fact_constellation",
         "constellation_time_ms")
    ]:

        df[
            f"{schema_name}_improvement_percent"
        ] = (
            (
                df[column] -
                df["best_time_ms"]
            )
            /
            df[column]
        ) * 100

    # Regret if a fixed Star schema is used.
    df["star_regret_ms"] = (
        df["star_time_ms"] -
        df["best_time_ms"]
    )

    # Oracle potential improvement compared
    # with fixed Star.
    df["oracle_improvement_percent"] = (
        (
            df["star_time_ms"] -
            df["best_time_ms"]
        )
        /
        df["star_time_ms"]
    ) * 100

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("V3 training dataset created.")
    print()

    print(f"Queries: {len(df)}")
    print()

    print("Feature columns:")

    print(
        df[
            [
                "query_id",
                "join_count",
                "aggregation_count",
                "predicate_count",
                "group_by_count",
                "tables_used",
                "query_length"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print()
    print("Best schema distribution:")

    print(
        df["best_schema"]
        .value_counts()
    )

    print()
    print("Oracle average latency:")

    print(
        round(
            df["best_time_ms"].mean(),
            4
        ),
        "ms"
    )

    print()
    print("Fixed Star average latency:")

    print(
        round(
            df["star_time_ms"].mean(),
            4
        ),
        "ms"
    )

    print()
    print("Oracle improvement over Star:")

    print(
        round(
            df[
                "oracle_improvement_percent"
            ].mean(),
            2
        ),
        "%"
    )

    print()
    print("Workload-wise oracle improvement:")

    workload_summary = (
        df.groupby("workload")
        .agg(
            queries=("query_id", "count"),
            star_ms=("star_time_ms", "mean"),
            oracle_ms=("best_time_ms", "mean"),
            improvement_percent=(
                "oracle_improvement_percent",
                "mean"
            )
        )
        .round(4)
    )

    print(workload_summary)

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_training_data()
