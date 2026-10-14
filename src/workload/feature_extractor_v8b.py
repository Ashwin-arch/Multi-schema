import argparse
import json
import os
import re
import sys
import duckdb
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.generate_data_v7 import get_scale_label
from src.schemas.build_schemas_v7 import get_db_paths
from src.workload.validate_workload_v8 import adapt_sql_v8

WORKLOAD_FILE = "data/processed/workload_v8b.json"


def count_pattern(text, pattern):
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def extract_number(pattern, text):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_semantic_features(query_sql):
    """Extract syntactic and semantic properties of query SQL without execution."""
    sql = query_sql.lower()

    joins = count_pattern(query_sql, r"\bJOIN\b")
    aggregations = count_pattern(query_sql, r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(")
    predicates = count_pattern(query_sql, r"\b(WHERE|AND|OR)\b")
    group_by_count = count_pattern(query_sql, r"\bGROUP\s+BY\b")

    tables = re.findall(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", query_sql, flags=re.IGNORECASE)

    has_customer = int("dim_customer" in sql or "vw_customer" in sql)
    has_product = int("dim_product" in sql or "vw_product" in sql)
    has_date = int("dim_date" in sql)
    has_region = int(re.search(r"\bregion\b", query_sql, flags=re.IGNORECASE) is not None)
    has_segment = int(re.search(r"\bsegment\b", query_sql, flags=re.IGNORECASE) is not None)
    has_category = int(re.search(r"\bcategory\b", query_sql, flags=re.IGNORECASE) is not None)

    has_date_filter = int(bool(re.search(r"\bWHERE\b.*\b(year|month|quarter)\b", query_sql, flags=re.IGNORECASE | re.DOTALL)))
    has_customer_filter = int(bool(re.search(r"\bWHERE\b.*\b(region|segment)\b", query_sql, flags=re.IGNORECASE | re.DOTALL)))
    has_product_filter = int(bool(re.search(r"\bWHERE\b.*\b(category|product)\b", query_sql, flags=re.IGNORECASE | re.DOTALL)))

    group_dimension_count = (
        int("region" in sql)
        + int("segment" in sql)
        + int("category" in sql)
        + int("product" in sql)
        + int("year" in sql)
        + int("month" in sql)
        + int("quarter" in sql)
    )

    filter_count = count_pattern(query_sql, r"\b(BETWEEN|=)\b")
    join_depth = max(0, joins)

    return {
        "join_count": joins,
        "aggregation_count": aggregations,
        "predicate_count": predicates,
        "group_by_count": group_by_count,
        "tables_used": len(set(tables)),
        "has_customer": has_customer,
        "has_product": has_product,
        "has_date": has_date,
        "has_region": has_region,
        "has_segment": has_segment,
        "has_category": has_category,
        "has_date_filter": has_date_filter,
        "has_customer_filter": has_customer_filter,
        "has_product_filter": has_product_filter,
        "group_dimension_count": group_dimension_count,
        "filter_count": filter_count,
        "join_depth": join_depth,
    }


def extract_plan_features(plan_text):
    """Extract structural and cardinality features from DuckDB EXPLAIN plan."""
    plan_upper = plan_text.upper()

    scan_count = count_pattern(plan_text, r"(SEQ_SCAN|TABLE_SCAN|COLUMN_DATA_SCAN|READ_PARQUET)")
    hash_join_count = count_pattern(plan_text, r"HASH_JOIN")
    nested_loop_count = count_pattern(plan_text, r"NESTED_LOOP_JOIN")
    merge_join_count = count_pattern(plan_text, r"PIECEWISE_MERGE_JOIN")
    join_count = hash_join_count + nested_loop_count + merge_join_count

    filter_operator_count = count_pattern(plan_text, r"\bFILTER\b")
    aggregate_count = count_pattern(plan_text, r"(UNGROUPED_AGGREGATE|HASH_GROUP_BY|PERFECT_HASH_GROUP_BY)")
    group_by_count = count_pattern(plan_text, r"(HASH_GROUP_BY|PERFECT_HASH_GROUP_BY)")
    order_by_count = count_pattern(plan_text, r"\bORDER_BY\b")
    limit_count = count_pattern(plan_text, r"\bSTREAMING_LIMIT\b|\bTOP_N\b|\bLIMIT\b")
    projection_count = count_pattern(plan_text, r"\bPROJECTION\b")

    left_join_count = count_pattern(plan_text, r"LEFT")
    inner_join_count = count_pattern(plan_text, r"INNER")

    # Cardinality estimates: e.g. ~1234 rows or EC: 1234
    cardinality_matches = re.findall(r"~([0-9,]+)\s+rows", plan_text, flags=re.IGNORECASE)
    cardinality_values = []
    for match in cardinality_matches:
        try:
            cardinality_values.append(float(match.replace(",", "")))
        except ValueError:
            pass

    if cardinality_values:
        card_max = max(cardinality_values)
        card_min = min(cardinality_values)
        card_mean = float(np.mean(cardinality_values))
        card_nodes = len(cardinality_values)
    else:
        card_max = 0.0
        card_min = 0.0
        card_mean = 0.0
        card_nodes = 0

    operator_count = len(re.findall(r"─\s*([A-Z_]+)", plan_text))
    if operator_count == 0:
        operator_count = (
            scan_count + join_count + filter_operator_count
            + aggregate_count + order_by_count + limit_count + projection_count
        )

    plan_depth = len(re.findall(r"┌|├|└", plan_text))
    if plan_depth == 0:
        plan_depth = max(1, join_count + 1)

    return {
        "plan_scan_count": scan_count,
        "plan_join_count": join_count,
        "plan_hash_join_count": hash_join_count,
        "plan_nested_loop_join_count": nested_loop_count,
        "plan_merge_join_count": merge_join_count,
        "plan_filter_count": filter_operator_count,
        "plan_aggregate_count": aggregate_count,
        "plan_group_by_count": group_by_count,
        "plan_order_by_count": order_by_count,
        "plan_limit_count": limit_count,
        "plan_projection_count": projection_count,
        "plan_left_join_count": left_join_count,
        "plan_inner_join_count": inner_join_count,
        "estimated_cardinality_max": card_max,
        "estimated_cardinality_min": card_min,
        "estimated_cardinality_mean": card_mean,
        "cardinality_node_count": card_nodes,
        "plan_operator_count": operator_count,
        "plan_depth": plan_depth,
        "plan_text_length": len(plan_text),
    }


def extract_all_features_for_scale(scale_rows=100_000, workload_file=WORKLOAD_FILE, db_paths=None, output_csv=None):
    """
    Extract pre-execution semantic and cross-schema query plan features.
    Guaranteed: NO target leakage, NO execution latency used.
    """
    scale_label = get_scale_label(scale_rows)
    if db_paths is None:
        db_paths = get_db_paths(scale_label)

    if output_csv is None:
        output_csv = f"data/processed/features_{scale_label.lower()}_v8b.csv"

    print("=" * 75)
    print(f"PRE-EXECUTION FEATURE EXTRACTION V8b - SCALE {scale_label} ({scale_rows:,} rows)")
    print("=" * 75)

    with open(workload_file, "r") as f:
        workload = json.load(f)

    connections = {}
    for schema, path in db_paths.items():
        connections[schema] = duckdb.connect(path, read_only=True)

    records = []
    comparable_bases = [
        "plan_scan_count", "plan_join_count", "plan_hash_join_count",
        "plan_filter_count", "plan_aggregate_count", "plan_group_by_count",
        "plan_order_by_count", "plan_operator_count", "plan_depth",
        "estimated_cardinality_max", "estimated_cardinality_mean",
        "plan_text_length"
    ]

    for item in workload:
        qid = item["query_id"]
        w_type = item["workload"]
        q_type = item.get("query_type", "unknown")
        sql = item["sql"]

        # Semantic features
        sem = extract_semantic_features(sql)

        # Plan features across schemas
        schema_plans = {}
        for schema, con in connections.items():
            schema_sql = adapt_sql_v8(sql, schema)
            try:
                res = con.execute("EXPLAIN " + schema_sql).fetchall()
                plan_str = "\n".join(str(x[1] if len(x) > 1 else x[0]) for x in res)
            except Exception:
                plan_str = ""
            plan_feats = extract_plan_features(plan_str)
            schema_plans[schema] = plan_feats

        row = {
            "query_id": qid,
            "dataset_size": scale_rows,
            "scale_label": scale_label,
            "workload": w_type,
            "query_type": q_type,
            **sem
        }

        # Add schema-specific plan features
        for schema, feats in schema_plans.items():
            for k, v in feats.items():
                row[f"{schema}_{k}"] = v

        # Add cross-schema differential features
        for base in comparable_bases:
            star_val = schema_plans["star"].get(base, 0)
            snow_val = schema_plans["snowflake"].get(base, 0)
            fc_val = schema_plans["fact_constellation"].get(base, 0)

            row[f"diff_snow_minus_star_{base}"] = snow_val - star_val
            row[f"diff_fc_minus_star_{base}"] = fc_val - star_val
            row[f"diff_snow_minus_fc_{base}"] = snow_val - fc_val

        records.append(row)

    for con in connections.values():
        con.close()

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Extracted features for {len(df)} queries -> {output_csv}")
    print(f"Total feature columns: {len(df.columns) - 5}")
    print("=" * 75)
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract pre-execution query-plan and semantic features V8b.")
    parser.add_argument("--rows", type=int, default=100_000, help="Sales rows scale")
    parser.add_argument("--workload", type=str, default=WORKLOAD_FILE, help="Workload file")
    parser.add_argument("--output", type=str, default=None, help="Output feature CSV")
    args = parser.parse_args()

    extract_all_features_for_scale(scale_rows=args.rows, workload_file=args.workload, output_csv=args.output)


if __name__ == "__main__":
    main()