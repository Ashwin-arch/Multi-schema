import json
import re
import pandas as pd
import os


WORKLOAD_FILE = "data/processed/workload_v2.json"
BENCHMARK_FILE = "data/processed/benchmark_results_v2.csv"
OUTPUT_FILE = "data/processed/training_data_v2.csv"


def count_pattern(query, pattern):
    return len(
        re.findall(
            pattern,
            query,
            flags=re.IGNORECASE
        )
    )


def extract_features():

    with open(WORKLOAD_FILE, "r") as file:
        workloads = json.load(file)

    records = []

    for query_info in workloads:

        query = query_info["sql"]

        joins = count_pattern(query, r"\bJOIN\b")
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
            r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            query,
            flags=re.IGNORECASE
        )

        query_length = len(query)

        records.append({
            "query_id": query_info["query_id"],
            "workload": query_info["workload"],
            "query_type": query_info["query_type"],
            "join_count": joins,
            "aggregation_count": aggregations,
            "predicate_count": predicates,
            "group_by_count": group_by_count,
            "tables_used": len(set(tables)),
            "query_length": query_length
        })

    features = pd.DataFrame(records)

    # -----------------------------
    # Load benchmark measurements
    # -----------------------------

    benchmark = pd.read_csv(BENCHMARK_FILE)

    benchmark = benchmark[
        benchmark["status"] == "success"
    ].copy()

    # Pivot schema timings
    timing = benchmark.pivot(
        index="query_id",
        columns="schema",
        values="median_time_ms"
    ).reset_index()

    timing = timing.rename(columns={
        "star": "star_time_ms",
        "snowflake": "snowflake_time_ms",
        "fact_constellation": "constellation_time_ms"
    })

    # Add timings
    df = features.merge(
        timing,
        on="query_id",
        how="inner"
    )

    # -----------------------------
    # Determine observed best schema
    # -----------------------------

    time_columns = [
        "star_time_ms",
        "snowflake_time_ms",
        "constellation_time_ms"
    ]

    df["best_schema"] = df[
        time_columns
    ].idxmin(axis=1)

    df["best_schema"] = df["best_schema"].replace({
        "star_time_ms": "star",
        "snowflake_time_ms": "snowflake",
        "constellation_time_ms": "fact_constellation"
    })

    df["best_time_ms"] = df[
        time_columns
    ].min(axis=1)

    # -----------------------------
    # Static Star baseline
    # -----------------------------

    df["star_improvement_percent"] = (
        (
            df["star_time_ms"] -
            df["best_time_ms"]
        )
        /
        df["star_time_ms"]
    ) * 100

    # -----------------------------
    # Static Snowflake baseline
    # -----------------------------

    df["snowflake_improvement_percent"] = (
        (
            df["snowflake_time_ms"] -
            df["best_time_ms"]
        )
        /
        df["snowflake_time_ms"]
    ) * 100

    # -----------------------------
    # Static Constellation baseline
    # -----------------------------

    df["constellation_improvement_percent"] = (
        (
            df["constellation_time_ms"] -
            df["best_time_ms"]
        )
        /
        df["constellation_time_ms"]
    ) * 100

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("V2 training dataset created.")
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
        ].head(10).to_string(index=False)
    )

    print()
    print("Best schema distribution:")
    print(
        df["best_schema"]
        .value_counts()
    )

    print()
    print("Average improvement over Star:")
    print(
        round(
            df["star_improvement_percent"].mean(),
            2
        ),
        "%"
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_features()
