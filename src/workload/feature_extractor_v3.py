import json
import re
import pandas as pd


WORKLOAD_FILE = "data/processed/workload_v3.json"
BENCHMARK_FILE = "data/processed/benchmark_results_v3.csv"
OUTPUT_FILE = "data/processed/training_data_v3_rich.csv"


def contains(query, pattern):
    return int(
        re.search(
            pattern,
            query,
            flags=re.IGNORECASE
        ) is not None
    )


def count(query, pattern):
    return len(
        re.findall(
            pattern,
            query,
            flags=re.IGNORECASE
        )
    )


def extract_features(query):

    query_lower = query.lower()

    joins = count(
        query,
        r"\bJOIN\b"
    )

    aggregations = count(
        query,
        r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\("
    )

    predicates = count(
        query,
        r"\b(WHERE|AND|OR)\b"
    )

    group_by_count = count(
        query,
        r"\bGROUP\s+BY\b"
    )

    tables = re.findall(
        r"\b(?:FROM|JOIN)\s+"
        r"([a-zA-Z_][a-zA-Z0-9_]*)",
        query,
        flags=re.IGNORECASE
    )

    # Detect referenced dimensions.
    has_customer = int(
        "dim_customer" in query_lower
        or "vw_customer" in query_lower
    )

    has_product = int(
        "dim_product" in query_lower
        or "vw_product" in query_lower
    )

    has_date = int(
        "dim_date" in query_lower
    )

    has_region = contains(
        query,
        r"\bregion\b"
    )

    has_segment = contains(
        query,
        r"\bsegment\b"
    )

    has_category = contains(
        query,
        r"\bcategory\b"
    )

    # Detect filtering dimensions.
    has_date_filter = (
        int(
            re.search(
                r"\bWHERE\b.*\b(year|month|quarter)\b",
                query,
                flags=re.IGNORECASE |
                re.DOTALL
            )
            is not None
        )
    )

    has_customer_filter = int(
        re.search(
            r"\bWHERE\b.*\b(customer_id|region|segment)\b",
            query,
            flags=re.IGNORECASE |
            re.DOTALL
        )
        is not None
    )

    has_product_filter = int(
        re.search(
            r"\bWHERE\b.*\b(product_id|category)\b",
            query,
            flags=re.IGNORECASE |
            re.DOTALL
        )
        is not None
    )

    has_order_by = contains(
        query,
        r"\bORDER\s+BY\b"
    )

    has_limit = contains(
        query,
        r"\bLIMIT\b"
    )

    # Aggregation type.
    aggregation_type = "none"

    if re.search(
        r"\bSUM\s*\(",
        query,
        re.IGNORECASE
    ):
        aggregation_type = "sum"

    elif re.search(
        r"\bAVG\s*\(",
        query,
        re.IGNORECASE
    ):
        aggregation_type = "avg"

    elif re.search(
        r"\bCOUNT\s*\(",
        query,
        re.IGNORECASE
    ):
        aggregation_type = "count"

    elif re.search(
        r"\bMIN\s*\(",
        query,
        re.IGNORECASE
    ):
        aggregation_type = "min"

    elif re.search(
        r"\bMAX\s*\(",
        query,
        re.IGNORECASE
    ):
        aggregation_type = "max"

    # Number of grouping dimensions.
    group_match = re.search(
        r"\bGROUP\s+BY\s+(.+?)(?:\bORDER\s+BY\b|\bLIMIT\b|$)",
        query,
        flags=re.IGNORECASE |
        re.DOTALL
    )

    if group_match:

        group_text = group_match.group(1)

        group_dimension_count = (
            group_text.count(",") + 1
        )

    else:

        group_dimension_count = 0

    # Query complexity indicators.
    join_depth = joins

    return {
        "join_count": joins,
        "aggregation_count": aggregations,
        "predicate_count": predicates,
        "group_by_count": group_by_count,
        "tables_used": len(set(tables)),
        "query_length": len(query),

        "has_customer": has_customer,
        "has_product": has_product,
        "has_date": has_date,

        "has_region": has_region,
        "has_segment": has_segment,
        "has_category": has_category,

        "has_date_filter": has_date_filter,
        "has_customer_filter": has_customer_filter,
        "has_product_filter": has_product_filter,

        "has_order_by": has_order_by,
        "has_limit": has_limit,

        "aggregation_type": aggregation_type,
        "group_dimension_count":
            group_dimension_count,

        "filter_count": predicates,
        "join_depth": join_depth
    }


def build_dataset():

    with open(WORKLOAD_FILE, "r") as file:
        workloads = json.load(file)

    records = []

    for query_info in workloads:

        features = extract_features(
            query_info["sql"]
        )

        records.append({
            "query_id":
                query_info["query_id"],

            "workload":
                query_info["workload"],

            "query_type":
                query_info["query_type"],

            **features
        })

    features = pd.DataFrame(records)

    benchmark = pd.read_csv(
        BENCHMARK_FILE
    )

    benchmark = benchmark[
        benchmark["status"] == "success"
    ]

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
            "snowflake":
                "snowflake_time_ms",
            "fact_constellation":
                "constellation_time_ms"
        }
    )

    df = features.merge(
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
            "snowflake_time_ms":
                "snowflake",
            "constellation_time_ms":
                "fact_constellation"
        })
    )

    df["best_time_ms"] = (
        df[time_columns]
        .min(axis=1)
    )

    df["oracle_improvement_percent"] = (
        (
            df["star_time_ms"] -
            df["best_time_ms"]
        )
        /
        df["star_time_ms"]
    ) * 100

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Rich V3 feature dataset created.")
    print()
    print(f"Queries: {len(df)}")
    print()

    print("Feature count:")
    print(
        len(
            [
                column
                for column in df.columns
                if column not in [
                    "query_id",
                    "workload",
                    "query_type",
                    "star_time_ms",
                    "snowflake_time_ms",
                    "constellation_time_ms",
                    "best_schema",
                    "best_time_ms",
                    "oracle_improvement_percent"
                ]
            ]
        )
    )

    print()
    print("Best schema distribution:")
    print(
        df["best_schema"].value_counts()
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    build_dataset()
