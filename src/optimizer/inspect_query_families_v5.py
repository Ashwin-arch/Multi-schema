import json
import re
from collections import Counter

INPUT_FILE = "data/processed/workload_v3.json"


def structural_signature(row):
    sql = row["sql"].lower()

    # --------------------------------------------------------
    # Remove literal strings
    # --------------------------------------------------------
    sql = re.sub(r"'[^']*'", "'VALUE'", sql)

    # --------------------------------------------------------
    # Replace numeric literals
    # --------------------------------------------------------
    sql = re.sub(r"\b\d+\b", "N", sql)

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------
    sql = " ".join(sql.split())

    # --------------------------------------------------------
    # Extract aggregation
    # --------------------------------------------------------
    agg = "none"

    for fn in ["sum", "avg", "count", "min", "max"]:
        if re.search(rf"\b{fn}\s*\(", sql):
            agg = fn
            break

    # --------------------------------------------------------
    # Extract joins
    # --------------------------------------------------------
    joins = []

    for table in ["dim_customer", "dim_product", "dim_date"]:
        if re.search(
            rf"\bjoin\s+{table}\b",
            sql
        ):
            joins.append(table)

    joins = tuple(joins)

    # --------------------------------------------------------
    # Extract grouping dimensions
    # --------------------------------------------------------
    group_cols = []

    match = re.search(
        r"\bgroup\s+by\s+(.+?)\s+order\s+by",
        sql
    )

    if match:
        group_part = match.group(1)

        for col in [
            "c.region",
            "c.segment",
            "p.category",
            "p.product_id",
            "d.year",
            "d.month",
        ]:
            if col in group_part:
                group_cols.append(col)

    group_cols = tuple(group_cols)

    # --------------------------------------------------------
    # Extract filter columns
    # --------------------------------------------------------
    filter_cols = []

    if "where" in sql:

        for col in [
            "c.region",
            "c.segment",
            "p.category",
            "p.product_id",
            "c.customer_id",
            "fs.customer_id",
            "fs.product_id",
            "d.month",
            "d.year",
        ]:
            if re.search(
                rf"\b{re.escape(col)}\b",
                sql
            ):
                filter_cols.append(col)

    filter_cols = tuple(sorted(set(filter_cols)))

    # --------------------------------------------------------
    # Selective query subtype
    # --------------------------------------------------------
    selective_type = "none"

    if row["workload"] == "selective":

        if "fs.customer_id =" in sql:
            selective_type = "customer"

        elif "fs.product_id =" in sql:
            selective_type = "product"

        elif "c.region =" in sql:
            selective_type = "region"

        elif "c.segment =" in sql:
            selective_type = "segment"

        elif "d.month =" in sql:
            selective_type = "date"

    # --------------------------------------------------------
    # Number of WHERE predicates
    # --------------------------------------------------------
    predicate_count = 0

    where_match = re.search(
        r"\bwhere\b(.+?)(?:\bgroup\s+by\b|\border\s+by\b|$)",
        sql
    )

    if where_match:
        predicate_count = (
            len(
                re.findall(
                    r"\band\b",
                    where_match.group(1)
                )
            )
            + 1
        )

    # --------------------------------------------------------
    # LIMIT structure
    # --------------------------------------------------------
    limit = "none"

    limit_match = re.search(
        r"\blimit\s+N\b",
        sql
    )

    if limit_match:
        limit = "N"

    # --------------------------------------------------------
    # Structural signature
    # --------------------------------------------------------
    return (
        row["workload"],
        agg,
        joins,
        group_cols,
        filter_cols,
        predicate_count,
        selective_type,
        limit,
    )


# ============================================================
# LOAD
# ============================================================

with open(INPUT_FILE) as f:
    queries = json.load(f)


signatures = []

for row in queries:

    sig = structural_signature(row)

    signatures.append({
        "query_id": row["query_id"],
        "workload": row["workload"],
        "signature": sig,
    })


# ============================================================
# GROUP
# ============================================================

counts = Counter(
    x["signature"]
    for x in signatures
)


print("=" * 75)
print("V5 STRUCTURAL QUERY FAMILY ANALYSIS")
print("=" * 75)

print(f"\nTotal queries       : {len(queries)}")
print(f"Unique families    : {len(counts)}")

print(
    f"Average family size: "
    f"{len(queries) / len(counts):.2f}"
)

print(
    f"Largest family     : "
    f"{max(counts.values())}"
)

print(
    f"Singleton families : "
    f"{sum(1 for x in counts.values() if x == 1)}"
)


# ============================================================
# SIZE DISTRIBUTION
# ============================================================

print("\nFamily size distribution")
print("-" * 75)

size_distribution = Counter(
    counts.values()
)

for size in sorted(size_distribution):
    print(
        f"size {size:3d}: "
        f"{size_distribution[size]:3d} families"
    )


# ============================================================
# TOP FAMILIES
# ============================================================

print("\nLargest structural families")
print("-" * 75)

for i, (sig, count) in enumerate(
    counts.most_common(20),
    start=1
):

    workload, agg, joins, groups, filters, predicates, selective, limit = sig

    print(
        f"{i:2d}. "
        f"count={count:3d} | "
        f"workload={workload:22s} | "
        f"agg={agg:5s} | "
        f"joins={len(joins)} | "
        f"groups={groups} | "
        f"filters={filters} | "
        f"predicates={predicates} | "
        f"selective={selective}"
    )


print("\nAnalysis complete.")
