import json
import os
import random


SEED = 42
random.seed(SEED)

OUTPUT = "data/processed/workload_v2.json"

os.makedirs("data/processed", exist_ok=True)


queries = []

query_id = 1


def add_query(workload, query_type, sql):
    global query_id

    queries.append({
        "query_id": f"Q{query_id:03d}",
        "workload": workload,
        "query_type": query_type,
        "sql": sql.strip()
    })

    query_id += 1


# --------------------------------------------------
# 1. Aggregation workloads
# --------------------------------------------------

aggregation_templates = [
    """
    SELECT
        d.year,
        SUM(f.total_amount) AS total_sales
    FROM fact_sales f
    JOIN dim_date d
        ON f.date_id = d.date_id
    GROUP BY d.year
    """,

    """
    SELECT
        d.month,
        SUM(f.total_amount) AS total_sales
    FROM fact_sales f
    JOIN dim_date d
        ON f.date_id = d.date_id
    GROUP BY d.month
    """,

    """
    SELECT
        p.category,
        SUM(f.total_amount) AS revenue
    FROM fact_sales f
    JOIN dim_product p
        ON f.product_id = p.product_id
    GROUP BY p.category
    """,

    """
    SELECT
        c.region,
        SUM(f.total_amount) AS revenue
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_id = c.customer_id
    GROUP BY c.region
    """,

    """
    SELECT
        c.segment,
        AVG(f.total_amount) AS average_sale
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_id = c.customer_id
    GROUP BY c.segment
    """,

    """
    SELECT
        p.category,
        SUM(f.quantity) AS units
    FROM fact_sales f
    JOIN dim_product p
        ON f.product_id = p.product_id
    GROUP BY p.category
    """
]

for template in aggregation_templates:
    for _ in range(4):
        add_query(
            "aggregation",
            "aggregation",
            template
        )


# --------------------------------------------------
# 2. Join-heavy workloads
# --------------------------------------------------

join_templates = [
    """
    SELECT
        c.region,
        p.category,
        SUM(f.total_amount) AS revenue
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_id = c.customer_id
    JOIN dim_product p
        ON f.product_id = p.product_id
    GROUP BY c.region, p.category
    """,

    """
    SELECT
        d.year,
        c.segment,
        SUM(f.total_amount) AS revenue
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_id = c.customer_id
    JOIN dim_date d
        ON f.date_id = d.date_id
    GROUP BY d.year, c.segment
    """,

    """
    SELECT
        d.month,
        p.category,
        SUM(f.quantity) AS units
    FROM fact_sales f
    JOIN dim_product p
        ON f.product_id = p.product_id
    JOIN dim_date d
        ON f.date_id = d.date_id
    GROUP BY d.month, p.category
    """,

    """
    SELECT
        c.region,
        p.category,
        d.quarter,
        SUM(f.total_amount) AS revenue
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_id = c.customer_id
    JOIN dim_product p
        ON f.product_id = p.product_id
    JOIN dim_date d
        ON f.date_id = d.date_id
    GROUP BY c.region, p.category, d.quarter
    """
]

for template in join_templates:
    for _ in range(4):
        add_query(
            "join_heavy",
            "join",
            template
        )


# --------------------------------------------------
# 3. Selective workloads
# --------------------------------------------------

customer_ids = [5, 10, 25, 50, 100, 250, 500, 750, 900]
product_ids = [5, 10, 15, 20, 25, 30, 40, 50]
date_ranges = [
    (1, 7),
    (1, 30),
    (1, 60),
    (100, 130),
    (150, 180),
    (200, 230),
    (300, 365)
]

for customer_id in customer_ids:
    add_query(
        "selective",
        "selective",
        f"""
        SELECT *
        FROM fact_sales
        WHERE customer_id = {customer_id}
        """
    )

for product_id in product_ids:
    add_query(
        "selective",
        "selective",
        f"""
        SELECT *
        FROM fact_sales
        WHERE product_id = {product_id}
        """
    )

for start_date, end_date in date_ranges:
    add_query(
        "selective",
        "selective",
        f"""
        SELECT *
        FROM fact_sales
        WHERE date_id BETWEEN {start_date} AND {end_date}
        """
    )


# --------------------------------------------------
# 4. Filter + aggregation workloads
# --------------------------------------------------

quantity_thresholds = [2, 4, 6, 8]
discount_thresholds = [0.05, 0.10, 0.15, 0.20]

for threshold in quantity_thresholds:

    add_query(
        "filtered_aggregation",
        "mixed",
        f"""
        SELECT
            SUM(total_amount) AS revenue,
            AVG(total_amount) AS average_sale
        FROM fact_sales
        WHERE quantity >= {threshold}
        """
    )

for threshold in discount_thresholds:

    add_query(
        "filtered_aggregation",
        "mixed",
        f"""
        SELECT
            SUM(total_amount) AS revenue,
            COUNT(*) AS sales_count
        FROM fact_sales
        WHERE discount >= {threshold}
        """
    )


# --------------------------------------------------
# 5. Multi-dimensional mixed workloads
# --------------------------------------------------

mixed_templates = [
    """
    SELECT
        c.region,
        p.category,
        SUM(f.total_amount) AS revenue
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_id = c.customer_id
    JOIN dim_product p
        ON f.product_id = p.product_id
    WHERE f.quantity >= 5
    GROUP BY c.region, p.category
    """,

    """
    SELECT
        d.quarter,
        c.segment,
        SUM(f.total_amount) AS revenue
    FROM fact_sales f
    JOIN dim_date d
        ON f.date_id = d.date_id
    JOIN dim_customer c
        ON f.customer_id = c.customer_id
    WHERE f.discount > 0.10
    GROUP BY d.quarter, c.segment
    """,

    """
    SELECT
        d.month,
        p.category,
        AVG(f.total_amount) AS average_sale
    FROM fact_sales f
    JOIN dim_date d
        ON f.date_id = d.date_id
    JOIN dim_product p
        ON f.product_id = p.product_id
    WHERE f.quantity >= 3
    GROUP BY d.month, p.category
    """,

    """
    SELECT
        c.region,
        d.year,
        p.category,
        SUM(f.total_amount) AS revenue
    FROM fact_sales f
    JOIN dim_customer c
        ON f.customer_id = c.customer_id
    JOIN dim_product p
        ON f.product_id = p.product_id
    JOIN dim_date d
        ON f.date_id = d.date_id
    WHERE f.discount >= 0.10
      AND f.quantity >= 4
    GROUP BY c.region, d.year, p.category
    """
]

for template in mixed_templates:
    for _ in range(5):
        add_query(
            "mixed",
            "mixed",
            template
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(OUTPUT, "w") as file:
    json.dump(queries, file, indent=4)


print("Workload V2 generated successfully.")
print()
print(f"Total queries : {len(queries)}")
print()
print("Workload distribution:")

distribution = {}

for query in queries:
    workload = query["workload"]
    distribution[workload] = distribution.get(workload, 0) + 1

for workload, count in distribution.items():
    print(f"{workload:25} : {count}")

print()
print(f"Saved to: {OUTPUT}")
