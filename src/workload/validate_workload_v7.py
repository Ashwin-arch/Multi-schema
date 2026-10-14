import argparse
import json
import os
import re
import sys
import duckdb

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.generate_data_v7 import get_scale_label
from src.schemas.build_schemas_v7 import get_db_paths

WORKLOAD_FILE = "data/processed/workload_v3.json"


def adapt_sql(sql, schema):
    """
    Robust SQL adapter for multi-schema execution.
    - Ensures total_amount measure is used instead of sales_amount.
    - Routes Snowflake queries to the normalized compatibility views.
    """
    # Replace sales_amount with total_amount
    adapted = re.sub(r"\bfs\.sales_amount\b", "fs.total_amount", sql, flags=re.IGNORECASE)

    # Snowflake-specific view mappings
    if schema == "snowflake":
        adapted = re.sub(r"\bdim_customer\s+c\b", "vw_customer c", adapted, flags=re.IGNORECASE)
        adapted = re.sub(r"\bdim_product\s+p\b", "vw_product p", adapted, flags=re.IGNORECASE)

    return adapted


def validate_workload(workload_file=WORKLOAD_FILE, scale_rows=100_000, db_paths=None):
    """
    Validate all workload queries against Star, Snowflake, and Fact Constellation schemas.
    Returns validation report dictionary and boolean overall_success.
    """
    if db_paths is None:
        scale_label = get_scale_label(scale_rows)
        db_paths = get_db_paths(scale_label)

    print("=" * 65)
    print("WORKLOAD PRE-BENCHMARK VALIDATION")
    print("=" * 65)
    print(f"Workload file : {workload_file}")
    print(f"Dataset scale : {scale_rows:,} rows")

    if not os.path.exists(workload_file):
        raise FileNotFoundError(f"Workload file not found: {workload_file}")

    with open(workload_file, "r") as f:
        workload = json.load(f)

    print(f"Total queries : {len(workload)}")

    # Connect to databases
    connections = {}
    for schema, path in db_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Database not found: {path}. Run build_schemas_v7.py first.")
        connections[schema] = duckdb.connect(path, read_only=True)

    results = {
        "total_queries": len(workload),
        "schemas": {},
        "failures": []
    }

    for schema in ["star", "snowflake", "fact_constellation"]:
        con = connections[schema]
        success_count = 0
        fail_count = 0

        for item in workload:
            qid = item["query_id"]
            sql = item["sql"]
            schema_sql = adapt_sql(sql, schema)

            try:
                con.execute(schema_sql).fetchall()
                success_count += 1
            except Exception as e:
                fail_count += 1
                results["failures"].append({
                    "query_id": qid,
                    "schema": schema,
                    "error": str(e),
                    "sql": schema_sql
                })

        results["schemas"][schema] = {
            "success": success_count,
            "failed": fail_count
        }
        print(f"  {schema:20s}: {success_count} successful, {fail_count} failed")

    for con in connections.values():
        con.close()

    total_expected = len(workload) * 3
    total_successful = sum(s["success"] for s in results["schemas"].values())
    total_failed = sum(s["failed"] for s in results["schemas"].values())

    print("-" * 65)
    print(f"Total executions: {total_expected} ({total_successful} successful, {total_failed} failed)")
    is_valid = (total_failed == 0)
    print(f"Validation status: {'PASSED (100% compatible)' if is_valid else 'FAILED'}")
    print("=" * 65)

    return results, is_valid


def main():
    parser = argparse.ArgumentParser(description="Validate workload queries across schemas.")
    parser.add_argument("--rows", type=int, default=100_000, help="Scale in sales rows")
    parser.add_argument("--workload", type=str, default=WORKLOAD_FILE, help="Path to workload JSON")
    args = parser.parse_args()

    _, is_valid = validate_workload(workload_file=args.workload, scale_rows=args.rows)
    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
