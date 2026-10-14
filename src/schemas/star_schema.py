import duckdb
import os


DB_PATH = "data/processed/star_schema.duckdb"

os.makedirs("data/processed", exist_ok=True)


def create_star_schema():
    con = duckdb.connect(DB_PATH)

    # Drop existing tables so the script is reproducible
    con.execute("DROP TABLE IF EXISTS fact_sales")
    con.execute("DROP TABLE IF EXISTS dim_customer")
    con.execute("DROP TABLE IF EXISTS dim_product")
    con.execute("DROP TABLE IF EXISTS dim_date")

    # -----------------------------
    # Customer Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_customer AS
        SELECT
            customer_id,
            customer_name,
            region,
            segment
        FROM read_csv_auto('data/raw/customers.csv')
    """)

    # -----------------------------
    # Product Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_product AS
        SELECT
            product_id,
            product_name,
            category,
            unit_price
        FROM read_csv_auto('data/raw/products.csv')
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

    print("Star schema created successfully.")
    print()

    for table in [
        "dim_customer",
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
    create_star_schema()
