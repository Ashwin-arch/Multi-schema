import duckdb
import os


DB_PATH = "data/processed/snowflake_schema.duckdb"

os.makedirs("data/processed", exist_ok=True)


def create_snowflake_schema():
    con = duckdb.connect(DB_PATH)

    # Remove old tables
    tables = [
        "fact_sales",
        "dim_customer",
        "dim_region",
        "dim_segment",
        "dim_product",
        "dim_category",
        "dim_date"
    ]

    for table in tables:
        con.execute(f"DROP TABLE IF EXISTS {table}")

    # -----------------------------
    # Region Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_region AS
        SELECT
            ROW_NUMBER() OVER () AS region_id,
            region
        FROM (
            SELECT DISTINCT region
            FROM read_csv_auto('data/raw/customers.csv')
        )
    """)

    # -----------------------------
    # Segment Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_segment AS
        SELECT
            ROW_NUMBER() OVER () AS segment_id,
            segment
        FROM (
            SELECT DISTINCT segment
            FROM read_csv_auto('data/raw/customers.csv')
        )
    """)

    # -----------------------------
    # Customer Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_customer AS
        SELECT
            c.customer_id,
            c.customer_name,
            r.region_id,
            s.segment_id
        FROM read_csv_auto('data/raw/customers.csv') c
        JOIN dim_region r
            ON c.region = r.region
        JOIN dim_segment s
            ON c.segment = s.segment
    """)

    # -----------------------------
    # Category Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_category AS
        SELECT
            ROW_NUMBER() OVER () AS category_id,
            category
        FROM (
            SELECT DISTINCT category
            FROM read_csv_auto('data/raw/products.csv')
        )
    """)

    # -----------------------------
    # Product Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_product AS
        SELECT
            p.product_id,
            p.product_name,
            c.category_id,
            p.unit_price
        FROM read_csv_auto('data/raw/products.csv') p
        JOIN dim_category c
            ON p.category = c.category
    """)

    # -----------------------------
    # Date Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_date AS
        SELECT
            date_id,
            full_date,
            year,
            quarter,
            month,
            month_name,
            day
        FROM read_csv_auto('data/raw/dates.csv')
    """)

    # -----------------------------
    # Sales Fact
    # -----------------------------
    con.execute("""
        CREATE TABLE fact_sales AS
        SELECT
            sale_id,
            date_id,
            customer_id,
            product_id,
            quantity,
            discount,
            total_amount
        FROM read_csv_auto('data/raw/sales.csv')
    """)

    print("Snowflake schema created successfully.")
    print()

    for table in [
        "dim_region",
        "dim_segment",
        "dim_customer",
        "dim_category",
        "dim_product",
        "dim_date",
        "fact_sales"
    ]:
        count = con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table:15} : {count:,} rows")

    con.close()


if __name__ == "__main__":
    create_snowflake_schema()
