import duckdb


DB_PATH = "data/processed/snowflake_schema.duckdb"


def create_compatibility_views():
    con = duckdb.connect(DB_PATH)

    con.execute("DROP VIEW IF EXISTS vw_customer")
    con.execute("DROP VIEW IF EXISTS vw_product")

    # Logical customer view over normalized Snowflake dimensions
    con.execute("""
        CREATE VIEW vw_customer AS
        SELECT
            c.customer_id,
            c.customer_name,
            c.region_id,
            c.segment_id,
            r.region,
            s.segment
        FROM dim_customer c
        JOIN dim_region r
            ON c.region_id = r.region_id
        JOIN dim_segment s
            ON c.segment_id = s.segment_id
    """)

    # Logical product view over normalized Snowflake dimensions
    con.execute("""
        CREATE VIEW vw_product AS
        SELECT
            p.product_id,
            p.product_name,
            c.category,
            p.unit_price
        FROM dim_product p
        JOIN dim_category c
            ON p.category_id = c.category_id
    """)

    con.close()

    print("Snowflake compatibility views created successfully.")


if __name__ == "__main__":
    create_compatibility_views()
