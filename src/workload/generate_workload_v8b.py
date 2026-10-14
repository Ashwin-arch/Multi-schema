import json
import random
import os

OUTPUT_FILE = "data/processed/workload_v8b.json"

# Deterministic seed for reproducibility
random.seed(2026)

# Dimension cardinalities
NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 60
NUM_DATES = 365

REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
CATEGORIES = ["Electronics", "Furniture", "Office"]
YEARS = [2024, 2025]

REGION_IDS = {r: i+1 for i, r in enumerate(REGIONS)}
SEGMENT_IDS = {s: i+1 for i, s in enumerate(SEGMENTS)}
CATEGORY_IDS = {c: i+1 for i, c in enumerate(CATEGORIES)}

CUSTOMER_IDS = list(range(1, NUM_CUSTOMERS + 1))
PRODUCT_IDS = list(range(1, NUM_PRODUCTS + 1))

AGGREGATIONS = [
    "SUM(fs.total_amount)",
    "AVG(fs.total_amount)",
    "COUNT(fs.sale_id)",
    "MIN(fs.total_amount)",
    "MAX(fs.total_amount)",
]

RETURN_AGGREGATIONS = [
    "SUM(fr.returned_quantity)",
    "AVG(fr.refund_amount)",
    "COUNT(fr.return_id)",
    "SUM(fr.refund_amount)",
]

# ============================================================
# ALL QUERIES USE CANONICAL (DEORMALIZED) COLUMN NAMES
# The adapter handles schema-specific translations:
# - Star: direct denormalized access
# - Snowflake: compatibility views (vw_customer, vw_product)
# - Fact Constellation: direct denormalized access + fact_returns
# ============================================================

# --- Class A: STAR-ORIENTED (wide scans, minimal joins) ---

def star_wide_aggregation():
    agg = random.choice(AGGREGATIONS)
    group_col = random.choice(["d.year", "d.quarter", "d.month"])
    return f"""
SELECT {group_col}, {agg} AS metric
FROM fact_sales fs
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY {group_col}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()

def star_customer_aggregation():
    agg = random.choice(AGGREGATIONS)
    group_col = random.choice(["c.region", "c.segment", "c.region, c.segment"])
    return f"""
SELECT {group_col}, {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY {group_col}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()

def star_product_aggregation():
    agg = random.choice(AGGREGATIONS)
    group_col = random.choice(["p.category", "p.product_name"])
    return f"""
SELECT {group_col}, {agg} AS metric
FROM fact_sales fs
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY {group_col}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()

def star_large_aggregation():
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT {agg} AS metric
FROM fact_sales fs
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {random.choice(YEARS)}
""".strip()

# --- Class B: SNOWFLAKE-ORIENTED (normalized dimension paths) ---
# These use canonical column names (c.region, c.segment, p.category)
# Adapter routes to vw_customer/vw_product which expose these

def snowflake_region_drilldown():
    region = random.choice(REGIONS)
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT c.segment, {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE c.region = '{region}'
  AND d.year = {random.choice(YEARS)}
GROUP BY c.segment
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()

def snowflake_segment_drilldown():
    region = random.choice(REGIONS)
    segment = random.choice(SEGMENTS)
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT c.customer_name, {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE c.region = '{region}'
  AND c.segment = '{segment}'
  AND d.year = {random.choice(YEARS)}
GROUP BY c.customer_name
ORDER BY metric DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

def snowflake_category_drilldown():
    category = random.choice(CATEGORIES)
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT p.product_name, {agg} AS metric
FROM fact_sales fs
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE p.category = '{category}'
  AND d.year = {random.choice(YEARS)}
GROUP BY p.product_name
ORDER BY metric DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

def snowflake_normalized_filter():
    region = random.choice(REGIONS)
    segment = random.choice(SEGMENTS)
    category = random.choice(CATEGORIES)
    year = random.choice(YEARS)
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE c.region = '{region}'
  AND c.segment = '{segment}'
  AND p.category = '{category}'
  AND d.year = {year}
""".strip()

# --- Class C: FACT CONSTELLATION-ORIENTED (multi-fact: sales + returns) ---
# These reference fact_returns. Adapter ensures all schemas can access it.
# For Star/Snowflake: fact_returns is accessed via cross-db or same-db if available
# For FC: native fact_returns table

def fc_sales_vs_returns_by_category():
    agg = random.choice(AGGREGATIONS)
    return_agg = random.choice(RETURN_AGGREGATIONS)
    return f"""
SELECT p.category, {agg} AS sales_metric, {return_agg} AS returns_metric
FROM fact_sales fs
JOIN fact_returns fr ON fs.sale_id = fr.return_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY p.category
ORDER BY sales_metric DESC
""".strip()

def fc_return_rate_by_region():
    return f"""
SELECT c.region,
       COUNT(DISTINCT fs.sale_id) AS total_sales,
       COUNT(fr.return_id) AS total_returns,
       COUNT(fr.return_id) * 1.0 / COUNT(DISTINCT fs.sale_id) AS return_rate
FROM fact_sales fs
LEFT JOIN fact_returns fr ON fs.sale_id = fr.return_id
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {random.choice(YEARS)}
GROUP BY c.region
ORDER BY return_rate DESC
""".strip()

def fc_customer_return_behavior():
    return f"""
SELECT c.customer_id, c.region, c.segment,
       COUNT(DISTINCT fs.sale_id) AS sales_count,
       SUM(fs.total_amount) AS total_sales,
       COUNT(fr.return_id) AS returns_count,
       SUM(fr.refund_amount) AS total_refunds,
       COUNT(fr.return_id) * 1.0 / COUNT(DISTINCT fs.sale_id) AS return_rate
FROM fact_sales fs
LEFT JOIN fact_returns fr ON fs.sale_id = fr.return_id
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {random.choice(YEARS)}
GROUP BY c.customer_id, c.region, c.segment
HAVING COUNT(DISTINCT fs.sale_id) > 5
ORDER BY return_rate DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

def fc_monthly_sales_returns_trend():
    year = random.choice(YEARS)
    return f"""
SELECT d.month,
       SUM(fs.total_amount) AS monthly_sales,
       SUM(fr.refund_amount) AS monthly_returns,
       SUM(fr.refund_amount) * 1.0 / SUM(fs.total_amount) AS return_ratio
FROM fact_sales fs
LEFT JOIN fact_returns fr ON fs.sale_id = fr.return_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {year}
GROUP BY d.month
ORDER BY d.month
""".strip()

def fc_product_return_analysis():
    return f"""
SELECT p.product_name, p.category,
       COUNT(DISTINCT fs.sale_id) AS units_sold,
       SUM(fs.total_amount) AS revenue,
       COUNT(fr.return_id) AS units_returned,
       SUM(fr.refund_amount) AS refund_amount,
       COUNT(fr.return_id) * 1.0 / COUNT(DISTINCT fs.sale_id) AS return_rate
FROM fact_sales fs
LEFT JOIN fact_returns fr ON fs.sale_id = fr.return_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {random.choice(YEARS)}
GROUP BY p.product_name, p.category
HAVING COUNT(DISTINCT fs.sale_id) > 10
ORDER BY return_rate DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

def fc_deep_multi_fact():
    year = random.choice(YEARS)
    return f"""
SELECT c.region, p.category,
       COUNT(DISTINCT fs.sale_id) AS sales,
       SUM(fs.total_amount) AS revenue,
       COUNT(fr.return_id) AS returns,
       SUM(fr.refund_amount) AS refunds,
       COUNT(fr.return_id) * 1.0 / COUNT(DISTINCT fs.sale_id) AS return_rate,
       SUM(fr.refund_amount) * 1.0 / SUM(fs.total_amount) AS refund_ratio
FROM fact_sales fs
LEFT JOIN fact_returns fr ON fs.sale_id = fr.return_id
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {year}
GROUP BY c.region, p.category
ORDER BY return_rate DESC
""".strip()

# --- Class D: SELECTIVE (point lookups) ---

def selective_customer():
    cid = random.choice(CUSTOMER_IDS)
    return f"""
SELECT fs.sale_id, fs.date_id, fs.product_id, fs.total_amount
FROM fact_sales fs
WHERE fs.customer_id = {cid}
ORDER BY fs.total_amount DESC
LIMIT {random.choice([5, 10, 20, 50])}
""".strip()

def selective_product():
    pid = random.choice(PRODUCT_IDS)
    return f"""
SELECT fs.sale_id, fs.date_id, fs.customer_id, fs.total_amount
FROM fact_sales fs
WHERE fs.product_id = {pid}
ORDER BY fs.total_amount DESC
LIMIT {random.choice([5, 10, 20, 50])}
""".strip()

def selective_region():
    region = random.choice(REGIONS)
    return f"""
SELECT fs.sale_id, fs.customer_id, fs.product_id, fs.total_amount
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
WHERE c.region = '{region}'
ORDER BY fs.total_amount DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

# --- Class E: MULTI-DIMENSIONAL GROUPING ---

def multi_dim_grouping():
    agg = random.choice(AGGREGATIONS)
    group_cols = random.sample(["c.region", "c.segment", "p.category", "d.year", "d.quarter"], 3)
    return f"""
SELECT {', '.join(group_cols)}, {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {random.choice(YEARS)}
GROUP BY {', '.join(group_cols)}
ORDER BY metric DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

# --- Class F: MIXED COMPLEX ---

def mixed_complex():
    agg = random.choice(AGGREGATIONS)
    filters = []
    if random.random() < 0.7:
        filters.append(f"d.year = {random.choice(YEARS)}")
    if random.random() < 0.5:
        filters.append(f"c.region = '{random.choice(REGIONS)}'")
    if random.random() < 0.3:
        filters.append(f"p.category = '{random.choice(CATEGORIES)}'")
    where = " AND ".join(filters) if filters else "1=1"
    group_cols = random.sample(["c.region", "c.segment", "p.category", "d.year", "d.quarter"], random.randint(2, 3))
    return f"""
SELECT {', '.join(group_cols)}, {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE {where}
GROUP BY {', '.join(group_cols)}
ORDER BY metric DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()


# ============================================================
# WORKLOAD GENERATION - TARGET ~300 QUERIES
# ============================================================

TARGET_COUNTS = {
    "star_oriented": 40,      # A
    "snowflake_oriented": 70, # B
    "fact_constellation_oriented": 70,  # C (multi-fact)
    "selective": 25,          # D
    "multi_dimensional": 20,  # E
    "mixed_complex": 35,      # F
}

GENERATORS = {
    "star_wide": ("star_oriented", star_wide_aggregation, 12),
    "star_customer": ("star_oriented", star_customer_aggregation, 12),
    "star_product": ("star_oriented", star_product_aggregation, 8),
    "star_large": ("star_oriented", star_large_aggregation, 8),
    "snowflake_region": ("snowflake_oriented", snowflake_region_drilldown, 20),
    "snowflake_segment": ("snowflake_oriented", snowflake_segment_drilldown, 20),
    "snowflake_category": ("snowflake_oriented", snowflake_category_drilldown, 15),
    "snowflake_normalized": ("snowflake_oriented", snowflake_normalized_filter, 15),
    "fc_sales_returns_cat": ("fact_constellation_oriented", fc_sales_vs_returns_by_category, 15),
    "fc_return_rate_region": ("fact_constellation_oriented", fc_return_rate_by_region, 15),
    "fc_customer_returns": ("fact_constellation_oriented", fc_customer_return_behavior, 12),
    "fc_monthly_trend": ("fact_constellation_oriented", fc_monthly_sales_returns_trend, 8),
    "fc_product_returns": ("fact_constellation_oriented", fc_product_return_analysis, 12),
    "fc_deep": ("fact_constellation_oriented", fc_deep_multi_fact, 8),
    "selective_cust": ("selective", selective_customer, 10),
    "selective_prod": ("selective", selective_product, 10),
    "selective_region": ("selective", selective_region, 5),
    "multi_dim": ("multi_dimensional", multi_dim_grouping, 20),
    "mixed": ("mixed_complex", mixed_complex, 35),
}


def generate_workload():
    queries = []
    seen = set()

    for name, (w_type, gen, target) in GENERATORS.items():
        count = 0
        attempts = 0
        while count < target and attempts < target * 10:
            attempts += 1
            sql = gen()
            norm = " ".join(sql.lower().split())
            if norm in seen:
                continue
            seen.add(norm)
            qid = f"Q{len(queries) + 1:03d}"
            queries.append({
                "query_id": qid,
                "workload": w_type,
                "query_type": name,
                "sql": sql
            })
            count += 1
        print(f"  {w_type:30s} {name:30s}: {count}/{target} (attempts: {attempts})")

    random.shuffle(queries)
    for i, q in enumerate(queries):
        q["query_id"] = f"Q{i+1:03d}"

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(queries, f, indent=2)

    print(f"\nGenerated {len(queries)} unique queries -> {OUTPUT_FILE}")
    dist = {}
    for q in queries:
        dist[q["workload"]] = dist.get(q["workload"], 0) + 1
    for k, v in sorted(dist.items()):
        print(f"  {k:30s}: {v}")

    return queries


if __name__ == "__main__":
    generate_workload()