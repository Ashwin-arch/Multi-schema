import json
import os
import time
import duckdb
import pandas as pd


WORKLOAD_FILE = "data/processed/workload_v3.json"
OUTPUT_FILE = "data/processed/benchmark_results_v3.csv"

SCHEMAS = {
    "star": "data/processed/star_schema.duckdb",
    "snowflake": "data/processed/snowflake_schema.duckdb",
    "fact_constellation": "data/processed/fact_constellation.duckdb"
}

REPEATS = 5
WARMUP = 1


def adapt_query(sql, schema):

    # All schemas use total_amount in fact_sales.
    sql = sql.replace(
        "fs.sales_amount",
        "fs.total_amount"
    )

    # Snowflake normalizes customer/product dimensions.
    # Compatibility views expose the same logical attributes
    # used by the workload.
    if schema == "snowflake":

        sql = sql.replace(
            "dim_customer c",
            "vw_customer c"
        )

        sql = sql.replace(
            "dim_product p",
            "vw_product p"
        )

    return sql


def execute_query(con, sql):
    start = time.perf_counter()

    con.execute(sql).fetchall()

    end = time.perf_counter()

    return (end - start) * 1000


def benchmark():

    with open(WORKLOAD_FILE, "r") as file:
        workloads = json.load(file)

    results = []

    expected = (
        len(workloads)
        * len(SCHEMAS)
        * REPEATS
    )

    print(f"Loaded queries: {len(workloads)}")
    print(f"Schemas: {len(SCHEMAS)}")
    print(f"Repeats: {REPEATS}")
    print(f"Expected executions: {expected}")
    print()

    for schema, db_path in SCHEMAS.items():

        print(f"Benchmarking {schema}...")

        con = duckdb.connect(
            db_path,
            read_only=True
        )

        successful = 0
        failed = 0

        for index, query_info in enumerate(
            workloads,
            start=1
        ):

            query_id = query_info["query_id"]
            sql = adapt_query(
                query_info["sql"],
                schema
            )

            # Warm-up execution.
            try:
                for _ in range(WARMUP):
                    con.execute(sql).fetchall()

            except Exception as error:

                results.append({
                    "query_id": query_id,
                    "workload": query_info["workload"],
                    "query_type": query_info["query_type"],
                    "schema": schema,
                    "repeat": 0,
                    "time_ms": None,
                    "status": "failed",
                    "error": str(error)
                })

                failed += 1

                continue

            for repeat in range(
                1,
                REPEATS + 1
            ):

                try:

                    elapsed = execute_query(
                        con,
                        sql
                    )

                    results.append({
                        "query_id": query_id,
                        "workload": query_info["workload"],
                        "query_type": query_info["query_type"],
                        "schema": schema,
                        "repeat": repeat,
                        "time_ms": elapsed,
                        "status": "success",
                        "error": ""
                    })

                    successful += 1

                except Exception as error:

                    results.append({
                        "query_id": query_id,
                        "workload": query_info["workload"],
                        "query_type": query_info["query_type"],
                        "schema": schema,
                        "repeat": repeat,
                        "time_ms": None,
                        "status": "failed",
                        "error": str(error)
                    })

                    failed += 1

            if index % 100 == 0:

                print(
                    f"  {index}/{len(workloads)} queries completed "
                    f"({successful} successful executions, "
                    f"{failed} failed)"
                )

        con.close()

        print(
            f"  {schema}: "
            f"{successful} successful, "
            f"{failed} failed"
        )

        print()

    df = pd.DataFrame(results)

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Benchmark complete.")
    print(
        f"Total execution records: {len(df)}"
    )

    print(
        f"Successful: "
        f"{(df['status'] == 'success').sum()}"
    )

    print(
        f"Failed: "
        f"{(df['status'] == 'failed').sum()}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    successful_df = df[
        df["status"] == "success"
    ].copy()

    if not successful_df.empty:

        print()
        print(
            "Overall timing summary (ms):"
        )

        summary = (
            successful_df
            .groupby("schema")["time_ms"]
            .agg([
                "mean",
                "median",
                "min",
                "max"
            ])
            .round(4)
        )

        print(summary)


if __name__ == "__main__":
    benchmark()
