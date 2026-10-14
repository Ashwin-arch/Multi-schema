import json
import os


OUTPUT = "data/processed/workload.json"

os.makedirs("data/processed", exist_ok=True)


workloads = {

    "aggregation_heavy": [
        {
            "query_id": "Q001",
            "type": "aggregation",
            "sql": """
                SELECT
                    d.year,
                    d.month,
                    SUM(f.total_amount) AS total_sales
                FROM fact_sales f
                JOIN dim_date d
                    ON f.date_id = d.date_id
                GROUP BY d.year, d.month
            """
        },
        {
            "query_id": "Q002",
            "type": "aggregation",
            "sql": """
                SELECT
                    p.category,
                    SUM(f.total_amount) AS revenue
                FROM fact_sales f
                JOIN dim_product p
                    ON f.product_id = p.product_id
                GROUP BY p.category
            """
        },
        {
            "query_id": "Q003",
            "type": "aggregation",
            "sql": """
                SELECT
                    c.region,
                    SUM(f.total_amount) AS revenue
                FROM fact_sales f
                JOIN dim_customer c
                    ON f.customer_id = c.customer_id
                GROUP BY c.region
            """
        }
    ],

    "join_heavy": [
        {
            "query_id": "Q004",
            "type": "join",
            "sql": """
                SELECT
                    c.region,
                    p.category,
                    d.year,
                    SUM(f.total_amount) AS revenue
                FROM fact_sales f
                JOIN dim_customer c
                    ON f.customer_id = c.customer_id
                JOIN dim_product p
                    ON f.product_id = p.product_id
                JOIN dim_date d
                    ON f.date_id = d.date_id
                GROUP BY
                    c.region,
                    p.category,
                    d.year
            """
        },
        {
            "query_id": "Q005",
            "type": "join",
            "sql": """
                SELECT
                    c.segment,
                    p.category,
                    SUM(f.quantity) AS units
                FROM fact_sales f
                JOIN dim_customer c
                    ON f.customer_id = c.customer_id
                JOIN dim_product p
                    ON f.product_id = p.product_id
                GROUP BY
                    c.segment,
                    p.category
            """
        }
    ],

    "selective": [
        {
            "query_id": "Q006",
            "type": "selective",
            "sql": """
                SELECT *
                FROM fact_sales
                WHERE customer_id = 10
            """
        },
        {
            "query_id": "Q007",
            "type": "selective",
            "sql": """
                SELECT *
                FROM fact_sales
                WHERE product_id = 25
            """
        },
        {
            "query_id": "Q008",
            "type": "selective",
            "sql": """
                SELECT *
                FROM fact_sales
                WHERE date_id BETWEEN 1 AND 30
            """
        }
    ],

    "mixed": [
        {
            "query_id": "Q009",
            "type": "mixed",
            "sql": """
                SELECT
                    c.region,
                    p.category,
                    AVG(f.total_amount) AS avg_sale
                FROM fact_sales f
                JOIN dim_customer c
                    ON f.customer_id = c.customer_id
                JOIN dim_product p
                    ON f.product_id = p.product_id
                WHERE f.quantity >= 5
                GROUP BY
                    c.region,
                    p.category
            """
        },
        {
            "query_id": "Q010",
            "type": "mixed",
            "sql": """
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
                GROUP BY
                    d.quarter,
                    c.segment
            """
        }
    ]
}


with open(OUTPUT, "w") as file:
    json.dump(workloads, file, indent=4)


print("Workload generation completed.")
print(f"Saved to: {OUTPUT}")

for workload_name, queries in workloads.items():
    print(f"{workload_name:20} : {len(queries)} queries")
