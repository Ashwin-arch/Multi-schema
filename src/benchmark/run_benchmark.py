import duckdb
import json
import os
import time
import pandas as pd


WORKLOAD_FILE = "data/processed/workload.json"
RESULT_FILE = "data/processed/benchmark_results.csv"


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
    con = duckdb.connect(database, read_only=True)

    start = time.perf_counter()

    result = con.execute(query).fetchdf()

    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000

    con.close()

    return elapsed_ms, len(result)


def run_benchmark():

    workloads = load_workloads()

    results = []

    for schema_name, config in DATABASES.items():

        print()
        print("=" * 60)
        print(f"Testing schema: {schema_name}")
        print("=" * 60)

        for workload_type, queries in workloads.items():

            for query_info in queries:

                query_id = query_info["query_id"]

                query = adapt_query(
                    query_info["sql"],
                    config["customer"],
                    config["product"]
                )

                try:

                    execution_time, rows = execute_query(
                        config["path"],
                        query
                    )

                    results.append({
                        "schema": schema_name,
                        "workload": workload_type,
                        "query_id": query_id,
                        "execution_time_ms": execution_time,
                        "rows_returned": rows,
                        "status": "success"
                    })

                    print(
                        f"{query_id:5} | "
                        f"{workload_type:20} | "
                        f"{execution_time:10.3f} ms | "
                        f"{rows:6} rows"
                    )

                except Exception as error:

                    results.append({
                        "schema": schema_name,
                        "workload": workload_type,
                        "query_id": query_id,
                        "execution_time_ms": None,
                        "rows_returned": None,
                        "status": f"error: {error}"
                    })

                    print(
                        f"{query_id:5} | "
                        f"{workload_type:20} | ERROR: {error}"
                    )

    df = pd.DataFrame(results)

    df.to_csv(
        RESULT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("Benchmark completed.")
    print(f"Results saved to: {RESULT_FILE}")
    print("=" * 60)

    successful = df[df["status"] == "success"]

    print()
    print("Successful queries:", len(successful), "/", len(df))

    print()
    print("Average execution time by schema:")

    summary = (
        successful
        .groupby("schema")["execution_time_ms"]
        .mean()
        .sort_values()
    )

    print(summary)


if __name__ == "__main__":
    run_benchmark()
