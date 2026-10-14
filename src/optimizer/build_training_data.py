import pandas as pd
import os


FEATURE_FILE = "data/processed/workload_features.csv"
BENCHMARK_FILE = "data/processed/benchmark_results.csv"
OUTPUT_FILE = "data/processed/training_data.csv"


def build_training_data():

    features = pd.read_csv(FEATURE_FILE)
    benchmark = pd.read_csv(BENCHMARK_FILE)

    # Keep only successful benchmark runs
    benchmark = benchmark[
        benchmark["status"] == "success"
    ].copy()

    # Convert execution time to numeric
    benchmark["execution_time_ms"] = pd.to_numeric(
        benchmark["execution_time_ms"]
    )

    # Pivot schema execution times
    timing = benchmark.pivot(
        index="query_id",
        columns="schema",
        values="execution_time_ms"
    ).reset_index()

    # Rename columns
    timing = timing.rename(columns={
        "star": "star_time_ms",
        "snowflake": "snowflake_time_ms",
        "fact_constellation": "constellation_time_ms"
    })

    # Merge workload features + benchmark timings
    df = features.merge(
        timing,
        on="query_id",
        how="inner"
    )

    # Find observed best schema
    time_columns = {
        "star": "star_time_ms",
        "snowflake": "snowflake_time_ms",
        "fact_constellation": "constellation_time_ms"
    }

    df["best_schema"] = df[
        list(time_columns.values())
    ].idxmin(axis=1)

    df["best_schema"] = df["best_schema"].replace({
        "star_time_ms": "star",
        "snowflake_time_ms": "snowflake",
        "constellation_time_ms": "fact_constellation"
    })

    # Best observed execution time
    df["best_time_ms"] = df[
        list(time_columns.values())
    ].min(axis=1)

    # Oracle improvement over static Star
    df["star_improvement_percent"] = (
        (
            df["star_time_ms"] - df["best_time_ms"]
        )
        / df["star_time_ms"]
    ) * 100

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Training dataset created.")
    print()
    print(df.to_string(index=False))
    print()
    print("Best schema distribution:")
    print(df["best_schema"].value_counts())
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_training_data()
