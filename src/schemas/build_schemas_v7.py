import argparse
import os
import sys
import time
import duckdb

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.generate_data_v7 import get_scale_label, RAW_DIR
from src.utils.resource_logging import record_stage_resource


def get_db_paths(scale_label):
    """Return dictionary of database file paths for a given scale."""
    suffix = f"_{scale_label.lower()}"
    return {
        "star": f"data/processed/star_schema{suffix}.duckdb",
        "snowflake": f"data/processed/snowflake_schema{suffix}.duckdb",
        "fact_constellation": f"data/processed/fact_constellation{suffix}.duckdb",
    }


def build_star_schema(sales_csv_path, db_path):
    """Construct Star Schema in DuckDB."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)

    con.execute(f"""
        CREATE TABLE dim_customer AS
        SELECT
            customer_id,
            customer_name,
            region,
            segment
        FROM read_csv_auto('{RAW_DIR}/customers.csv')
    """)

    con.execute(f"""
        CREATE TABLE dim_product AS
        SELECT
            product_id,
            product_name,
            category,
            unit_price
        FROM read_csv_auto('{RAW_DIR}/products.csv')
    """)

    con.execute(f"""
        CREATE TABLE dim_date AS
        SELECT
            date_id,
            full_date,
            year,
            quarter,
            month,
            month_name,
            day
        FROM read_csv_auto('{RAW_DIR}/dates.csv')
    """)

    con.execute(f"""
        CREATE TABLE fact_sales AS
        SELECT
            sale_id,
            date_id,
            customer_id,
            product_id,
            quantity,
            discount,
            total_amount
        FROM read_csv_auto('{sales_csv_path}')
    """)

    # Add fact_returns for cross-schema query compatibility
    con.execute(f"""
        CREATE TABLE fact_returns AS
        SELECT
            sale_id AS return_id,
            date_id,
            customer_id,
            product_id,
            CASE
                WHEN sale_id % 10 = 0 THEN 1
                WHEN sale_id % 37 = 0 THEN 1
                ELSE 0
            END AS returned_quantity,
            CASE
                WHEN sale_id % 10 = 0 THEN ROUND(total_amount * 0.80, 2)
                WHEN sale_id % 37 = 0 THEN ROUND(total_amount * 0.50, 2)
                ELSE 0
            END AS refund_amount
        FROM read_csv_auto('{sales_csv_path}')
        WHERE sale_id % 10 = 0 OR sale_id % 37 = 0
    """)

    row_counts = {}
    for table in ["dim_customer", "dim_product", "dim_date", "fact_sales", "fact_returns"]:
        row_counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    con.close()
    return row_counts


def build_snowflake_schema(sales_csv_path, db_path):
    """Construct Snowflake Schema and compatibility views in DuckDB."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)

    con.execute(f"""
        CREATE TABLE dim_region AS
        SELECT
            ROW_NUMBER() OVER () AS region_id,
            region
        FROM (
            SELECT DISTINCT region
            FROM read_csv_auto('{RAW_DIR}/customers.csv')
        )
    """)

    con.execute(f"""
        CREATE TABLE dim_segment AS
        SELECT
            ROW_NUMBER() OVER () AS segment_id,
            segment
        FROM (
            SELECT DISTINCT segment
            FROM read_csv_auto('{RAW_DIR}/customers.csv')
        )
    """)

    con.execute(f"""
        CREATE TABLE dim_customer AS
        SELECT
            c.customer_id,
            c.customer_name,
            r.region_id,
            s.segment_id
        FROM read_csv_auto('{RAW_DIR}/customers.csv') c
        JOIN dim_region r ON c.region = r.region
        JOIN dim_segment s ON c.segment = s.segment
    """)

    con.execute(f"""
        CREATE TABLE dim_category AS
        SELECT
            ROW_NUMBER() OVER () AS category_id,
            category
        FROM (
            SELECT DISTINCT category
            FROM read_csv_auto('{RAW_DIR}/products.csv')
        )
    """)

    con.execute(f"""
        CREATE TABLE dim_product AS
        SELECT
            p.product_id,
            p.product_name,
            c.category_id,
            p.unit_price
        FROM read_csv_auto('{RAW_DIR}/products.csv') p
        JOIN dim_category c ON p.category = c.category
    """)

    con.execute(f"""
        CREATE TABLE dim_date AS
        SELECT
            date_id,
            full_date,
            year,
            quarter,
            month,
            month_name,
            day
        FROM read_csv_auto('{RAW_DIR}/dates.csv')
    """)

    con.execute(f"""
        CREATE TABLE fact_sales AS
        SELECT
            sale_id,
            date_id,
            customer_id,
            product_id,
            quantity,
            discount,
            total_amount
        FROM read_csv_auto('{sales_csv_path}')
    """)

    # Add fact_returns for cross-schema query compatibility
    con.execute(f"""
        CREATE TABLE fact_returns AS
        SELECT
            sale_id AS return_id,
            date_id,
            customer_id,
            product_id,
            CASE
                WHEN sale_id % 10 = 0 THEN 1
                WHEN sale_id % 37 = 0 THEN 1
                ELSE 0
            END AS returned_quantity,
            CASE
                WHEN sale_id % 10 = 0 THEN ROUND(total_amount * 0.80, 2)
                WHEN sale_id % 37 = 0 THEN ROUND(total_amount * 0.50, 2)
                ELSE 0
            END AS refund_amount
        FROM read_csv_auto('{sales_csv_path}')
        WHERE sale_id % 10 = 0 OR sale_id % 37 = 0
    """)

    # Required Compatibility Views
    con.execute("""
        CREATE VIEW vw_customer AS
        SELECT
            c.customer_id,
            c.customer_name,
            r.region,
            s.segment
        FROM dim_customer c
        JOIN dim_region r ON c.region_id = r.region_id
        JOIN dim_segment s ON c.segment_id = s.segment_id
    """)

    con.execute("""
        CREATE VIEW vw_product AS
        SELECT
            p.product_id,
            p.product_name,
            c.category,
            p.unit_price
        FROM dim_product p
        JOIN dim_category c ON p.category_id = c.category_id
    """)

    row_counts = {}
    for table in ["dim_region", "dim_segment", "dim_customer", "dim_category", "dim_product", "dim_date", "fact_sales", "fact_returns"]:
        row_counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    con.close()
    return row_counts


def build_fact_constellation(sales_csv_path, db_path):
    """Construct Fact Constellation Schema (fact_sales + fact_returns) in DuckDB."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)

    con.execute(f"""
        CREATE TABLE dim_customer AS
        SELECT * FROM read_csv_auto('{RAW_DIR}/customers.csv')
    """)

    con.execute(f"""
        CREATE TABLE dim_product AS
        SELECT * FROM read_csv_auto('{RAW_DIR}/products.csv')
    """)

    con.execute(f"""
        CREATE TABLE dim_date AS
        SELECT * FROM read_csv_auto('{RAW_DIR}/dates.csv')
    """)

    con.execute(f"""
        CREATE TABLE fact_sales AS
        SELECT
            sale_id,
            date_id,
            customer_id,
            product_id,
            quantity,
            discount,
            total_amount
        FROM read_csv_auto('{sales_csv_path}')
    """)

    con.execute(f"""
        CREATE TABLE fact_returns AS
        SELECT
            sale_id AS return_id,
            date_id,
            customer_id,
            product_id,
            CASE
                WHEN sale_id % 10 = 0 THEN 1
                WHEN sale_id % 37 = 0 THEN 1
                ELSE 0
            END AS returned_quantity,
            CASE
                WHEN sale_id % 10 = 0 THEN ROUND(total_amount * 0.80, 2)
                WHEN sale_id % 37 = 0 THEN ROUND(total_amount * 0.50, 2)
                ELSE 0
            END AS refund_amount
        FROM read_csv_auto('{sales_csv_path}')
        WHERE sale_id % 10 = 0 OR sale_id % 37 = 0
    """)

    row_counts = {}
    for table in ["dim_customer", "dim_product", "dim_date", "fact_sales", "fact_returns"]:
        row_counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    con.close()
    return row_counts


def build_all_schemas(sales_csv_path, rows=100_000, set_as_default=False):
    """Construct all 3 schemas for a given scale and report metrics."""
    scale_label = get_scale_label(rows)
    db_paths = get_db_paths(scale_label)

    print("=" * 65)
    print(f"BUILDING SCHEMAS FOR SCALE {scale_label} ({rows:,} rows)")
    print("=" * 65)

    start_total = time.perf_counter()
    schema_stats = {}

    # 1. Star Schema
    t0 = time.perf_counter()
    star_counts = build_star_schema(sales_csv_path, db_paths["star"])
    t_star = time.perf_counter() - t0
    size_star_mb = os.path.getsize(db_paths["star"]) / (1024 * 1024)
    schema_stats["star"] = {"time_s": t_star, "size_mb": size_star_mb, "counts": star_counts}
    print(f"  [Star]               Built in {t_star:.2f}s ({size_star_mb:.2f} MB, {star_counts['fact_sales']:,} sales)")

    # 2. Snowflake Schema
    t0 = time.perf_counter()
    snow_counts = build_snowflake_schema(sales_csv_path, db_paths["snowflake"])
    t_snow = time.perf_counter() - t0
    size_snow_mb = os.path.getsize(db_paths["snowflake"]) / (1024 * 1024)
    schema_stats["snowflake"] = {"time_s": t_snow, "size_mb": size_snow_mb, "counts": snow_counts}
    print(f"  [Snowflake]          Built in {t_snow:.2f}s ({size_snow_mb:.2f} MB, {snow_counts['fact_sales']:,} sales, {snow_counts['fact_returns']:,} returns)")

    # 3. Fact Constellation Schema
    t0 = time.perf_counter()
    fc_counts = build_fact_constellation(sales_csv_path, db_paths["fact_constellation"])
    t_fc = time.perf_counter() - t0
    size_fc_mb = os.path.getsize(db_paths["fact_constellation"]) / (1024 * 1024)
    schema_stats["fact_constellation"] = {"time_s": t_fc, "size_mb": size_fc_mb, "counts": fc_counts}
    print(f"  [Fact Constellation] Built in {t_fc:.2f}s ({size_fc_mb:.2f} MB, {fc_counts['fact_sales']:,} sales, {fc_counts['fact_returns']:,} returns)")

    total_time = time.perf_counter() - start_total
    total_size_mb = size_star_mb + size_snow_mb + size_fc_mb

    print("-" * 65)
    print(f"Total build time: {total_time:.2f}s | Total DuckDB storage: {total_size_mb:.2f} MB")
    print("=" * 65)

    record_stage_resource(
        rows,
        db_build_time_s=total_time,
        total_db_size_mb=total_size_mb,
    )

    if set_as_default:
        import shutil
        shutil.copyfile(db_paths["star"], "data/processed/star_schema.duckdb")
        shutil.copyfile(db_paths["snowflake"], "data/processed/snowflake_schema.duckdb")
        shutil.copyfile(db_paths["fact_constellation"], "data/processed/fact_constellation.duckdb")
        print("Updated default active databases in data/processed/.")

    return {
        "scale": scale_label,
        "rows": rows,
        "total_build_time_s": total_time,
        "total_size_mb": total_size_mb,
        "schemas": schema_stats,
        "paths": db_paths
    }


def main():
    parser = argparse.ArgumentParser(description="Build Star, Snowflake, and Fact Constellation schemas in DuckDB.")
    parser.add_argument("--rows", type=int, default=100_000, help="Scale in sales rows (e.g. 100000, 500000, 1000000)")
    parser.add_argument("--sales_csv", type=str, default=None, help="Path to input sales CSV")
    parser.add_argument("--set_default", action="store_true", help="Copy built dbs to default paths")
    args = parser.parse_args()

    scale_label = get_scale_label(args.rows)
    sales_csv = args.sales_csv or f"{RAW_DIR}/sales_{scale_label.lower()}.csv"
    if not os.path.exists(sales_csv) and args.rows == 100_000 and os.path.exists(f"{RAW_DIR}/sales.csv"):
        sales_csv = f"{RAW_DIR}/sales.csv"

    if not os.path.exists(sales_csv):
        print(f"Error: sales CSV not found at {sales_csv}. Please run data generation first.")
        sys.exit(1)

    build_all_schemas(sales_csv, rows=args.rows, set_as_default=args.set_default)


if __name__ == "__main__":
    main()
