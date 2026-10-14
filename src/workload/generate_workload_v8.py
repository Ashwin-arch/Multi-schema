import json
import random
import os

OUTPUT_FILE = "data/processed/workload_v8.json"

# Deterministic seed for reproducibility
random.seed(2026)

# Dimension cardinalities (matching V7 data)
NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 60
NUM_DATES = 365

REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
CATEGORIES = ["Electronics", "Furniture", "Office"]
YEARS = [2024, 2025]

# Region IDs in Snowflake (1-4 based on ROW_NUMBER over distinct regions)
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
# CLASS A: STAR-ORIENTED — Wide scans, full fact table, minimal dimensions
# These favor Star (and FC) because they scan the full fact and join few dims
# ============================================================

def generate_star_wide_aggregation():
    """Full fact scan with simple grouping — Star excels at wide sequential scans."""
    agg = random.choice(AGGREGATIONS)
    group_col = random.choice(["d.year", "d.quarter", "d.month"])
    return f"""
SELECT
    {group_col},
    {agg} AS metric
FROM fact_sales fs
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY {group_col}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()

def generate_star_customer_aggregation():
    """Aggregation by customer attributes — 2-3 joins, still Star-friendly."""
    agg = random.choice(AGGREGATIONS)
    group_col = random.choice(["c.region", "c.segment", "c.region, c.segment"])
    return f"""
SELECT
    {group_col},
    {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY {group_col}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()

def generate_star_product_aggregation():
    """Aggregation by product category."""
    agg = random.choice(AGGREGATIONS)
    group_col = random.choice(["p.category", "p.product_name"])
    return f"""
SELECT
    {group_col},
    {agg} AS metric
FROM fact_sales fs
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY {group_col}
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()

# ============================================================
# CLASS B: SNOWFLAKE-ORIENTED — Normalized dimension traversal
# These favor Snowflake when filtering on normalized keys (region_id, segment_id, category_id)
# or when doing deep hierarchy drill-down
# ============================================================

def generate_snowflake_region_drilldown():
    """Drill from region → segment → customer — uses normalized path efficiently."""
    region = random.choice(REGIONS)
    agg = random.choice(AGGREGATIONS)
    # Query uses region_id filter directly, can use dim_region (4 rows) efficiently
    return f"""
SELECT
    c.segment,
    {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_segment s ON c.segment_id = s.segment_id
JOIN dim_region r ON c.region_id = r.region_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE r.region_id = {REGION_IDS[region]}
  AND d.year = {random.choice(YEARS)}
GROUP BY c.segment
ORDER BY metric DESC
LIMIT {random.choice([5, 10, 20])}
""".strip()

def generate_snowflake_segment_drilldown():
    """Drill into segment within region."""
    region = random.choice(REGIONS)
    segment = random.choice(SEGMENTS)
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT
    c.customer_name,
    {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_segment s ON c.segment_id = s.segment_id
JOIN dim_region r ON c.region_id = r.region_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE r.region_id = {REGION_IDS[region]}
  AND s.segment_id = {SEGMENT_IDS[segment]}
  AND d.year = {random.choice(YEARS)}
GROUP BY c.customer_name
ORDER BY metric DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

def generate_snowflake_category_drilldown():
    """Product → Category hierarchy drill-down."""
    category = random.choice(CATEGORIES)
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT
    p.product_name,
    {agg} AS metric
FROM fact_sales fs
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_category c ON p.category_id = c.category_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE c.category_id = {CATEGORY_IDS[category]}
  AND d.year = {random.choice(YEARS)}
GROUP BY p.product_name
ORDER BY metric DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

def generate_snowflake_normalized_filter():
    """High-selectivity filter on normalized dimension keys."""
    region = random.choice(REGIONS)
    segment = random.choice(SEGMENTS)
    category = random.choice(CATEGORIES)
    year = random.choice(YEARS)
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT
    {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_segment s ON c.segment_id = s.segment_id
JOIN dim_region r ON c.region_id = r.region_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_category cat ON p.category_id = cat.category_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE r.region_id = {REGION_IDS[region]}
  AND s.segment_id = {SEGMENT_IDS[segment]}
  AND cat.category_id = {CATEGORY_IDS[category]}
  AND d.year = {year}
""".strip()

# ============================================================
# CLASS C: FACT CONSTELLATION — Multi-fact analytical queries
# These require joining fact_sales + fact_returns — ONLY FC has both facts
# ============================================================

def generate_fc_sales_vs_returns_by_category():
    """Classic multi-fact: sales vs returns by product category."""
    agg = random.choice(AGGREGATIONS)
    return_agg = random.choice(RETURN_AGGREGATIONS)
    return f"""
SELECT
    p.category,
    {agg} AS sales_metric,
    {return_agg} AS returns_metric
FROM fact_sales fs
JOIN fact_returns fr ON fs.sale_id = fr.return_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
GROUP BY p.category
ORDER BY sales_metric DESC
""".strip()

def generate_fc_return_rate_by_region():
    """Return rate analysis by region — requires both facts."""
    return f"""
SELECT
    c.region,
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

def generate_fc_customer_return_behavior():
    """Customer-level sales and returns analysis."""
    return f"""
SELECT
    c.customer_id,
    c.region,
    c.segment,
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

def generate_fc_monthly_sales_returns_trend():
    """Time-series comparison of sales vs returns."""
    year = random.choice(YEARS)
    return f"""
SELECT
    d.month,
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

def generate_fc_product_return_analysis():
    """Product-level return analysis."""
    return f"""
SELECT
    p.product_name,
    p.category,
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

# ============================================================
# CLASS D: SELECTIVE — Point lookups, PK filters
# Should be fast on all schemas
# ============================================================

def generate_selective_customer():
    cid = random.choice(CUSTOMER_IDS)
    return f"""
SELECT
    fs.sale_id,
    fs.date_id,
    fs.product_id,
    fs.total_amount
FROM fact_sales fs
WHERE fs.customer_id = {cid}
ORDER BY fs.total_amount DESC
LIMIT {random.choice([5, 10, 20, 50])}
""".strip()

def generate_selective_product():
    pid = random.choice(PRODUCT_IDS)
    return f"""
SELECT
    fs.sale_id,
    fs.date_id,
    fs.customer_id,
    fs.total_amount
FROM fact_sales fs
WHERE fs.product_id = {pid}
ORDER BY fs.total_amount DESC
LIMIT {random.choice([5, 10, 20, 50])}
""".strip()

def generate_selective_region():
    region = random.choice(REGIONS)
    return f"""
SELECT
    fs.sale_id,
    fs.customer_id,
    fs.product_id,
    fs.total_amount
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
WHERE c.region = '{region}'
ORDER BY fs.total_amount DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

# ============================================================
# CLASS E: LARGE AGGREGATION — Full scans, minimal grouping
# Star should excel here
# ============================================================

def generate_large_aggregation():
    agg = random.choice(AGGREGATIONS)
    return f"""
SELECT
    {agg} AS metric
FROM fact_sales fs
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {random.choice(YEARS)}
""".strip()

# ============================================================
# CLASS F: MULTI-DIMENSIONAL GROUPING — Complex GROUP BY
# Tests join ordering and aggregation complexity
# ============================================================

def generate_multi_dim_grouping():
    agg = random.choice(AGGREGATIONS)
    group_cols = random.sample(["c.region", "c.segment", "p.category", "d.year", "d.quarter"], 3)
    return f"""
SELECT
    {', '.join(group_cols)},
    {agg} AS metric
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
JOIN dim_product p ON fs.product_id = p.product_id
JOIN dim_date d ON fs.date_id = d.date_id
WHERE d.year = {random.choice(YEARS)}
GROUP BY {', '.join(group_cols)}
ORDER BY metric DESC
LIMIT {random.choice([10, 20, 50])}
""".strip()

# ============================================================
# CLASS G: MULTI-FACT ANALYTICAL — Sales + Returns deep analysis
# ONLY Fact Constellation can do this efficiently
# ============================================================

def generate_multi_fact_deep_analysis():
    """Complex multi-fact query with multiple grouping dimensions."""
    year = random.choice(YEARS)
    return f"""
SELECT
    c.region,
    p.category,
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

# ============================================================
# CLASS H: MIXED COMPLEX — Realistic analytical patterns
# ============================================================

def generate_mixed_complex():
    agg = random.choice(AGGREGATIONS)
    filters = []
    if random.random() < 0.7:
        filters.append(f"d.year = {random.choice(YEARS)}")
    if random.random() < 0.5:
        filters.append(f"c.region = '{random.choice(REGIONS)}'")
    if random.random() < 0.3:
        filters.append(f"p.category = '{random.choice(CATEGORIES)}'")
    where = " AND ".join(filters) if filters else "1=1"
    
    group_cols = random.sample(["c.region", "c.segment", "p.category", "d.year", "d.quarter"], 
                               random.randint(2, 3))
    return f"""
SELECT
    {', '.join(group_cols)},
    {agg} AS metric
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
# WORKLOAD GENERATION
# ============================================================

NUM_QUERIES = 300  # 100-300 as specified

def generate_workload():
    queries = []
    
    # Target distribution across classes
    generators = [
        ("star_wide", "star_oriented", generate_star_wide_aggregation, 15),
        ("star_customer", "star_oriented", generate_star_customer_aggregation, 15),
        ("star_product", "star_oriented", generate_star_product_aggregation, 10),
        ("snowflake_region", "snowflake_oriented", generate_snowflake_region_drilldown, 20),
        ("snowflake_segment", "snowflake_oriented", generate_snowflake_segment_drilldown, 20),
        ("snowflake_category", "snowflake_oriented", generate_snowflake_category_drilldown, 15),
        ("snowflake_normalized", "snowflake_oriented", generate_snowflake_normalized_filter, 15),
        ("fc_sales_returns_cat", "fact_constellation_oriented", generate_fc_sales_vs_returns_by_category, 20),
        ("fc_return_rate_region", "fact_constellation_oriented", generate_fc_return_rate_by_region, 15),
        ("fc_customer_returns", "fact_constellation_oriented", generate_fc_customer_return_behavior, 15),
        ("fc_monthly_trend", "fact_constellation_oriented", generate_fc_monthly_sales_returns_trend, 10),
        ("fc_product_returns", "fact_constellation_oriented", generate_fc_product_return_analysis, 15),
        ("fc_deep", "fact_constellation_oriented", generate_multi_fact_deep_analysis, 10),
        ("selective_cust", "selective", generate_selective_customer, 10),
        ("selective_prod", "selective", generate_selective_product, 10),
        ("selective_region", "selective", generate_selective_region, 5),
        ("large_agg", "large_aggregation", generate_large_aggregation, 5),
        ("multi_dim", "multi_dimensional", generate_multi_dim_grouping, 20),
        ("mixed", "mixed_complex", generate_mixed_complex, 35),
    ]
    
    seen = set()
    total_target = sum(w for _, _, _, w in generators)
    
    for name, w_type, gen, target in generators:
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
        print(f"  {w_type:25s} {name:30s}: {count}/{target} (attempts: {attempts})")
    
    # Shuffle for random execution order
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
        print(f"  {k:25s}: {v}")
    
    return queries


if __name__ == "__main__":
    generate_workload()