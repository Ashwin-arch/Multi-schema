import duckdb
import json
import os
import time
import pandas as pd


WORKLOAD_FILE = "data/processed/workload_v2.json"
RESULT_FILE = "data/processed/benchmark_results_v2.csv"

REPEATS = 5


DATABASES = {
    "star": {
        "path": "data/processed/star_schema.duckdb",
        "customer": "dim_customer",
        "product": "dim_product"
    },
    "snowflake": {
        "path": "data/processed/snowflake_schema.duckdb",
        "customer": "vw_customer",
        "product": "vw_product"
    },
    "fact_constellation": {
        "path": "data/processed/fact_constellation.duckdb",
        "customer": "dim_customer",
        "product": "dim_product"
    }
}


def load_workloads():

    with open(WORKLOAD_FILE, "r") as file:
        return json.load(file)


def adapt_query(query, customer_table, product_table):

    query = query.replace(
        "dim_customer",
        customer_table
    )

    query = query.replace(
        "dim_product",
        product_table
    )

    return query


def execute_query(database, query):

    con = duckdb.connect(
        database,
        read_only=True
    )

    timings = []
    rows = None

    for _ in range(REPEATS):

        start = time.perf_counter()

        result = con.execute(query).fetchdf()

        end = time.perf_counter()

        timings.append(
            (end - start) * 1000
        )

        rows = len(result)

    con.close()

    return timings, rows


def run_benchmark():

    workloads = load_workloads()

    results = []

    total_queries = (
        len(workloads)
        * len(DATABASES)
    )

    completed = 0

    print()
    print("=" * 70)
    print("ADAPTIVE DATA WAREHOUSE V2 BENCHMARK")
    print("=" * 70)
    print(f"Queries per schema : {len(workloads)}")
    print(f"Schemas            : {len(DATABASES)}")
    print(f"Repeats per query  : {REPEATS}")
    print(f"Total executions   : {total_queries * REPEATS}")
    print("=" * 70)

    for schema_name, config in DATABASES.items():

        print()
        print(f"Schema: {schema_name}")
        print("-" * 70)

        for query_info in workloads:

            query_id = query_info["query_id"]
            workload = query_info["workload"]

            query = adapt_query(
                query_info["sql"],
                config["customer"],
                config["product"]
            )

            completed += 1

            try:

                timings, rows = execute_query(
                    config["path"],
                    query
                )

                median_time = float(
                    pd.Series(timings).median()
                )

                mean_time = float(
                    pd.Series(timings).mean()
                )

                min_time = float(
                    min(timings)
                )

                results.append({
                    "schema": schema_name,
                    "query_id": query_id,
                    "workload": workload,
                    "median_time_ms": median_time,
                    "mean_time_ms": mean_time,
                    "min_time_ms": min_time,
                    "rows_returned": rows,
                    "status": "success"
                })

                print(
                    f"{query_id} | "
                    f"{workload:22} | "
                    f"median={median_time:8.3f} ms | "
                    f"rows={rows}"
                )

            except Exception as error:

                results.append({
                    "schema": schema_name,
                    "query_id": query_id,
                    "workload": workload,
                    "median_time_ms": None,
                    "mean_time_ms": None,
                    "min_time_ms": None,
                    "rows_returned": None,
                    "status": f"error: {error}"
                })

                print(
                    f"{query_id} | ERROR | {error}"
                )

    df = pd.DataFrame(results)

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        RESULT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)

    successful = df[
        df["status"] == "success"
    ]

    print(
        f"Successful: {len(successful)} / {len(df)}"
    )

    print()
    print("Average median execution time:")
    print()

    summary = (
        successful
        .groupby("schema")["median_time_ms"]
        .agg(
            ["mean", "median", "min", "max"]
        )
    )

    print(summary)

    print()
    print(f"Results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    run_benchmark()
