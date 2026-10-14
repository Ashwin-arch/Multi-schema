import duckdb
import os


DB_PATH = "data/processed/fact_constellation.duckdb"

os.makedirs("data/processed", exist_ok=True)


def create_fact_constellation():
    con = duckdb.connect(DB_PATH)

    # Remove old tables
    tables = [
        "fact_sales",
        "fact_returns",
        "dim_customer",
        "dim_product",
        "dim_date"
    ]

    for table in tables:
        con.execute(f"DROP TABLE IF EXISTS {table}")

    # -----------------------------
    # Shared Customer Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_customer AS
        SELECT *
        FROM read_csv_auto('data/raw/customers.csv')
    """)

    # -----------------------------
    # Shared Product Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_product AS
        SELECT *
        FROM read_csv_auto('data/raw/products.csv')
    """)

    # -----------------------------
    # Shared Date Dimension
    # -----------------------------
    con.execute("""
        CREATE TABLE dim_date AS
        SELECT *
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

    # -----------------------------
    # Returns Fact
    #
    # We generate deterministic
    # returns from sales records.
    # -----------------------------
    con.execute("""
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
                WHEN sale_id % 10 = 0
                    THEN ROUND(total_amount * 0.80, 2)
                WHEN sale_id % 37 = 0
                    THEN ROUND(total_amount * 0.50, 2)
                ELSE 0
            END AS refund_amount
        FROM read_csv_auto('data/raw/sales.csv')
        WHERE sale_id % 10 = 0
           OR sale_id % 37 = 0
    """)

    print("Fact constellation schema created successfully.")
    print()

    for table in [
        "dim_customer",
        "dim_product",
        "dim_date",
        "fact_sales",
        "fact_returns"
    ]:
        count = con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table:15} : {count:,} rows")

    con.close()


if __name__ == "__main__":
    create_fact_constellation()
