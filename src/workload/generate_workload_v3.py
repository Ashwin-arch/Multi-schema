import json
import random
import os

OUTPUT_FILE = "data/processed/workload_v3.json"

random.seed(2026)

NUM_QUERIES = 600


REGIONS = [
    "North", "South", "East", "West"
]

SEGMENTS = [
    "Consumer", "Corporate", "Home Office"
]

CATEGORIES = [
    "Electronics", "Furniture", "Office Supplies"
]

CUSTOMER_IDS = list(range(1, 1001))
PRODUCT_IDS = list(range(1, 61))

AGGREGATIONS = [
    "SUM(fs.sales_amount)",
    "AVG(fs.sales_amount)",
    "COUNT(fs.sale_id)",
    "MIN(fs.sales_amount)",
    "MAX(fs.sales_amount)"
]

GROUP_COLUMNS = [
    "c.region",
    "c.segment",
    "p.category",
    "p.product_name",
    "d.year",
    "d.month",
    "d.quarter"
]


def random_date_filter():
    year = random.choice([2024, 2025])

    month1 = random.randint(1, 10)
    month2 = random.randint(month1 + 1, 12)

    return f"d.month BETWEEN {month1} AND {month2} AND d.year = {year}"


def generate_aggregation_query():
    aggregation = random.choice(AGGREGATIONS)
    group_col = random.choice(GROUP_COLUMNS)

    return f"""
SELECT
    {group_col},
    {aggregation} AS metric
FROM fact_sales fs
JOIN dim_customer c
    ON fs.customer_id = c.customer_id
JOIN dim_product p
    ON fs.product_id = p.product_id
JOIN dim_date d
    ON fs.date_id = d.date_id
GROUP BY {group_col}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()


def generate_filtered_aggregation_query():
    aggregation = random.choice(AGGREGATIONS)
    group_col = random.choice(GROUP_COLUMNS)

    filters = [
        random_date_filter(),
        f"c.region = '{random.choice(REGIONS)}'",
        f"c.segment = '{random.choice(SEGMENTS)}'",
        f"p.category = '{random.choice(CATEGORIES)}'",
        f"p.product_id BETWEEN "
        f"{random.randint(1, 30)} AND "
        f"{random.randint(31, 60)}"
    ]

    selected_filters = random.sample(
        filters,
        random.randint(1, 3)
    )

    where_clause = " AND ".join(selected_filters)

    return f"""
SELECT
    {group_col},
    {aggregation} AS metric
FROM fact_sales fs
JOIN dim_customer c
    ON fs.customer_id = c.customer_id
JOIN dim_product p
    ON fs.product_id = p.product_id
JOIN dim_date d
    ON fs.date_id = d.date_id
WHERE {where_clause}
GROUP BY {group_col}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()


def generate_join_query():

    aggregation = random.choice(AGGREGATIONS)

    patterns = [
        (
            ["c.region"],
            [
                "JOIN dim_customer c ON fs.customer_id = c.customer_id",
                "JOIN dim_date d ON fs.date_id = d.date_id"
            ],
            ["c", "d"]
        ),
        (
            ["c.segment"],
            [
                "JOIN dim_customer c ON fs.customer_id = c.customer_id",
                "JOIN dim_product p ON fs.product_id = p.product_id"
            ],
            ["c", "p"]
        ),
        (
            ["p.category"],
            [
                "JOIN dim_product p ON fs.product_id = p.product_id",
                "JOIN dim_date d ON fs.date_id = d.date_id"
            ],
            ["p", "d"]
        ),
        (
            ["c.region", "p.category"],
            [
                "JOIN dim_customer c ON fs.customer_id = c.customer_id",
                "JOIN dim_product p ON fs.product_id = p.product_id",
                "JOIN dim_date d ON fs.date_id = d.date_id"
            ],
            ["c", "p", "d"]
        ),
        (
            ["d.year", "d.month"],
            [
                "JOIN dim_date d ON fs.date_id = d.date_id",
                "JOIN dim_customer c ON fs.customer_id = c.customer_id"
            ],
            ["c", "d"]
        ),
        (
            ["d.year", "c.segment", "p.category"],
            [
                "JOIN dim_date d ON fs.date_id = d.date_id",
                "JOIN dim_customer c ON fs.customer_id = c.customer_id",
                "JOIN dim_product p ON fs.product_id = p.product_id"
            ],
            ["c", "p", "d"]
        )
    ]

    group_columns, joins, available_tables = random.choice(
        patterns
    )

    filters = []

    if "c" in available_tables:
        filters.extend([
            f"c.region = '{random.choice(REGIONS)}'",
            f"c.segment = '{random.choice(SEGMENTS)}'"
        ])

    if "p" in available_tables:
        filters.append(
            f"p.category = '{random.choice(CATEGORIES)}'"
        )

    if "d" in available_tables:
        filters.append(
            random_date_filter()
        )

    where_clause = ""

    if filters and random.choice([True, True, False]):

        selected_filters = random.sample(
            filters,
            min(
                random.randint(1, 2),
                len(filters)
            )
        )

        where_clause = (
            "\nWHERE " +
            " AND ".join(selected_filters)
        )

    group_clause = ", ".join(group_columns)

    join_clause = "\n".join(joins)

    return f"""
SELECT
    {group_clause},
    {aggregation} AS metric
FROM fact_sales fs
{join_clause}{where_clause}
GROUP BY {group_clause}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20, 50])}
""".strip()


def generate_selective_query():
    query_type = random.choice([
        "customer",
        "product",
        "region",
        "segment",
        "date"
    ])

    if query_type == "customer":

        customer_id = random.choice(CUSTOMER_IDS)

        return f"""
SELECT
    fs.sale_id,
    fs.customer_id,
    fs.product_id,
    fs.sales_amount
FROM fact_sales fs
WHERE fs.customer_id = {customer_id}
ORDER BY fs.sales_amount DESC
LIMIT {random.choice([5, 10, 20, 50])}
""".strip()

    if query_type == "product":

        product_id = random.choice(PRODUCT_IDS)

        return f"""
SELECT
    fs.sale_id,
    fs.customer_id,
    fs.product_id,
    fs.sales_amount
FROM fact_sales fs
WHERE fs.product_id = {product_id}
ORDER BY fs.sales_amount DESC
LIMIT {random.choice([5, 10, 20, 50])}
""".strip()

    if query_type == "region":

        region = random.choice(REGIONS)

        return f"""
SELECT
    fs.sale_id,
    fs.customer_id,
    fs.product_id,
    fs.sales_amount
FROM fact_sales fs
JOIN dim_customer c
    ON fs.customer_id = c.customer_id
WHERE c.region = '{region}'
ORDER BY fs.sales_amount DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

    if query_type == "segment":

        segment = random.choice(SEGMENTS)

        return f"""
SELECT
    fs.sale_id,
    fs.customer_id,
    fs.product_id,
    fs.sales_amount
FROM fact_sales fs
JOIN dim_customer c
    ON fs.customer_id = c.customer_id
WHERE c.segment = '{segment}'
ORDER BY fs.sales_amount DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

    month = random.randint(1, 12)

    return f"""
SELECT
    fs.sale_id,
    fs.customer_id,
    fs.product_id,
    fs.sales_amount
FROM fact_sales fs
JOIN dim_date d
    ON fs.date_id = d.date_id
WHERE d.month = {month}
ORDER BY fs.sales_amount DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()


def generate_mixed_query():

    aggregation = random.choice(AGGREGATIONS)
    group_col_1 = random.choice(GROUP_COLUMNS)
    group_col_2 = random.choice(GROUP_COLUMNS)

    while group_col_2 == group_col_1:
        group_col_2 = random.choice(GROUP_COLUMNS)

    filters = [
        random_date_filter(),
        f"c.region = '{random.choice(REGIONS)}'",
        f"c.segment = '{random.choice(SEGMENTS)}'",
        f"p.category = '{random.choice(CATEGORIES)}'"
    ]

    selected_filters = random.sample(
        filters,
        random.randint(1, 2)
    )

    where_clause = " AND ".join(selected_filters)

    return f"""
SELECT
    {group_col_1},
    {group_col_2},
    {aggregation} AS metric
FROM fact_sales fs
JOIN dim_customer c
    ON fs.customer_id = c.customer_id
JOIN dim_product p
    ON fs.product_id = p.product_id
JOIN dim_date d
    ON fs.date_id = d.date_id
WHERE {where_clause}
GROUP BY {group_col_1}, {group_col_2}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()


def generate_workload():

    queries = []

    generators = [
        ("aggregation", "aggregation", generate_aggregation_query),
        ("filtered_aggregation", "filtered_aggregation",
         generate_filtered_aggregation_query),
        ("join_heavy", "join_heavy", generate_join_query),
        ("selective", "selective", generate_selective_query),
        ("mixed", "mixed", generate_mixed_query)
    ]

    seen_queries = set()

    attempts = 0

    while len(queries) < NUM_QUERIES:

        attempts += 1

        workload, query_type, generator = random.choice(
            generators
        )

        sql = generator()

        normalized_sql = " ".join(
            sql.lower().split()
        )

        if normalized_sql in seen_queries:
            continue

        seen_queries.add(normalized_sql)

        query_id = f"Q{len(queries) + 1:03d}"

        queries.append({
            "query_id": query_id,
            "workload": workload,
            "query_type": query_type,
            "sql": sql
        })

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            queries,
            file,
            indent=2
        )

    print("V3 workload generated successfully.")
    print()
    print(f"Queries: {len(queries)}")
    print(f"Unique SQL queries: {len(seen_queries)}")
    print(f"Generation attempts: {attempts}")
    print()

    print("Workload distribution:")

    distribution = {}

    for query in queries:

        workload = query["workload"]

        distribution[workload] = (
            distribution.get(workload, 0) + 1
        )

    for workload, count in sorted(
        distribution.items()
    ):

        print(
            f"{workload:22s}: {count}"
        )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_workload()
