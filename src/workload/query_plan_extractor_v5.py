import json
import re
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

WORKLOAD_FILE = "data/processed/workload_v3.json"

DATABASES = {
    "star": "data/processed/star_schema.duckdb",
    "snowflake": "data/processed/snowflake_schema.duckdb",
    "fact_constellation": "data/processed/fact_constellation.duckdb",
}

OUTPUT = "data/processed/query_plan_features_v5.csv"


# ============================================================
# LOAD WORKLOAD
# ============================================================

with open(WORKLOAD_FILE, "r") as f:
    workload = json.load(f)


# Handle possible JSON structures
if isinstance(workload, dict):

    if "queries" in workload:
        queries = workload["queries"]

    elif "workload" in workload:
        queries = workload["workload"]

    else:
        # Try first list-valued field
        queries = None

        for value in workload.values():
            if isinstance(value, list):
                queries = value
                break

        if queries is None:
            raise ValueError(
                "Could not find query list in workload_v3.json"
            )

else:
    queries = workload


print("=" * 75)
print("QUERY-PLAN FEATURE EXTRACTION - V5")
print("=" * 75)

print(f"\nLoaded queries: {len(queries)}")


# ============================================================
# SQL EXTRACTION
# ============================================================

def get_query(item):

    if isinstance(item, str):
        return item

    for key in ["sql", "query", "query_sql", "statement"]:
        if key in item:
            return item[key]

    raise ValueError(
        f"Could not find SQL query in item: {item}"
    )


def get_query_id(item, index):

    if isinstance(item, dict):

        for key in ["query_id", "id"]:
            if key in item:
                return str(item[key])

    return f"Q{index + 1:03d}"


def get_workload(item):

    if isinstance(item, dict):

        for key in ["workload", "query_type", "type"]:
            if key in item:
                return str(item[key])

    return "unknown"


# ============================================================
# SNOWFLAKE QUERY ADAPTER
# ============================================================

def adapt_snowflake_sql(sql):

    sql = re.sub(
        r"\bdim_customer\s+c\b",
        "vw_customer c",
        sql,
        flags=re.IGNORECASE
    )

    sql = re.sub(
        r"\bdim_product\s+p\b",
        "vw_product p",
        sql,
        flags=re.IGNORECASE
    )

    sql = re.sub(
        r"\bfs\.sales_amount\b",
        "fs.total_amount",
        sql,
        flags=re.IGNORECASE
    )

    return sql


# ============================================================
# PLAN TEXT PARSING
# ============================================================

def count_pattern(plan, pattern):

    return len(
        re.findall(
            pattern,
            plan,
            flags=re.IGNORECASE
        )
    )


def extract_number(pattern, text):

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None


def extract_plan_features(plan):

    plan_upper = plan.upper()

    # --------------------------------------------------------
    # Operators
    # --------------------------------------------------------

    scan_count = count_pattern(
        plan,
        r"(SEQ_SCAN|TABLE_SCAN|COLUMN_DATA_SCAN|READ_PARQUET)"
    )

    hash_join_count = count_pattern(
        plan,
        r"HASH_JOIN"
    )

    nested_loop_count = count_pattern(
        plan,
        r"NESTED_LOOP_JOIN"
    )

    piecewise_merge_join_count = count_pattern(
        plan,
        r"PIECEWISE_MERGE_JOIN"
    )

    join_count = (
        hash_join_count
        + nested_loop_count
        + piecewise_merge_join_count
    )

    filter_operator_count = count_pattern(
        plan,
        r"\bFILTER\b"
    )

    aggregate_count = count_pattern(
        plan,
        r"(UNGROUPED_AGGREGATE|HASH_GROUP_BY|PERFECT_HASH_GROUP_BY)"
    )

    group_by_count = count_pattern(
        plan,
        r"(HASH_GROUP_BY|PERFECT_HASH_GROUP_BY)"
    )

    order_by_count = count_pattern(
        plan,
        r"\bORDER_BY\b"
    )

    limit_count = count_pattern(
        plan,
        r"\bSTREAMING_LIMIT\b|\bTOP_N\b|\bLIMIT\b"
    )

    projection_count = count_pattern(
        plan,
        r"\bPROJECTION\b"
    )

    # --------------------------------------------------------
    # Join types
    # --------------------------------------------------------

    left_join_count = count_pattern(
        plan,
        r"LEFT"
    )

    inner_join_count = count_pattern(
        plan,
        r"INNER"
    )

    # --------------------------------------------------------
    # Estimated cardinality
    # DuckDB EXPLAIN commonly exposes:
    # ~123 rows
    # --------------------------------------------------------

    cardinalities = re.findall(
        r"~\s*([\d,]+)\s+rows",
        plan,
        flags=re.IGNORECASE
    )

    cardinalities = [
        float(x.replace(",", ""))
        for x in cardinalities
    ]

    if cardinalities:

        estimated_cardinality_max = max(
            cardinalities
        )

        estimated_cardinality_min = min(
            cardinalities
        )

        estimated_cardinality_mean = sum(
            cardinalities
        ) / len(cardinalities)

        cardinality_nodes = len(
            cardinalities
        )

    else:

        estimated_cardinality_max = 0
        estimated_cardinality_min = 0
        estimated_cardinality_mean = 0
        cardinality_nodes = 0

    # --------------------------------------------------------
    # Plan complexity
    # --------------------------------------------------------

    plan_lines = [
        line
        for line in plan.splitlines()
        if line.strip()
    ]

    plan_depth = 0

    for line in plan_lines:

        stripped = line.lstrip()

        if stripped.startswith("┌") or stripped.startswith("│"):
            continue

        indentation = (
            len(line)
            - len(line.lstrip())
        )

        plan_depth = max(
            plan_depth,
            indentation
        )

    operator_count = (
        scan_count
        + join_count
        + filter_operator_count
        + aggregate_count
        + order_by_count
        + limit_count
        + projection_count
    )

    # --------------------------------------------------------
    # Plan text statistics
    # --------------------------------------------------------

    return {
        "plan_scan_count": scan_count,
        "plan_join_count": join_count,
        "plan_hash_join_count": hash_join_count,
        "plan_nested_loop_join_count": nested_loop_count,
        "plan_merge_join_count": piecewise_merge_join_count,

        "plan_filter_count": filter_operator_count,

        "plan_aggregate_count": aggregate_count,
        "plan_group_by_count": group_by_count,

        "plan_order_by_count": order_by_count,
        "plan_limit_count": limit_count,
        "plan_projection_count": projection_count,

        "plan_left_join_count": left_join_count,
        "plan_inner_join_count": inner_join_count,

        "estimated_cardinality_max":
            estimated_cardinality_max,

        "estimated_cardinality_min":
            estimated_cardinality_min,

        "estimated_cardinality_mean":
            estimated_cardinality_mean,

        "cardinality_node_count":
            cardinality_nodes,

        "plan_operator_count":
            operator_count,

        "plan_depth":
            plan_depth,

        "plan_text_length":
            len(plan),
    }


# ============================================================
# EXTRACT PLANS
# ============================================================

connections = {}

for schema, db_path in DATABASES.items():

    print(f"\nConnecting to {schema}...")

    connections[schema] = duckdb.connect(
        db_path,
        read_only=True
    )


rows = []

failed = 0


for index, item in enumerate(queries):

    query_id = get_query_id(
        item,
        index
    )

    workload_type = get_workload(item)

    sql = get_query(item)

    print(
        f"\rProcessing "
        f"{index + 1}/{len(queries)}",
        end=""
    )

    row = {
        "query_id": query_id,
        "workload": workload_type,
    }

    successful = True

    # --------------------------------------------------------
    # Extract plan from each schema
    # --------------------------------------------------------

    for schema, con in connections.items():

        # Common fact-table compatibility fix
        schema_sql = re.sub(
            r"\bfs\.sales_amount\b",
            "fs.total_amount",
            sql,
            flags=re.IGNORECASE
        )

        # Snowflake additionally requires normalized dimension views
        if schema == "snowflake":
            schema_sql = adapt_snowflake_sql(schema_sql)

        try:

            result = con.execute(
                "EXPLAIN " + schema_sql
            ).fetchall()

            plan = "\n".join(
                str(x[1] if len(x) > 1 else x[0])
                for x in result
            )

            features = extract_plan_features(
                plan
            )

            # Prefix every feature by schema
            for feature, value in features.items():

                row[
                    f"{schema}_{feature}"
                ] = value

        except Exception as e:

            successful = False

            row[
                f"{schema}_plan_error"
            ] = str(e)

    if not successful:
        failed += 1

    rows.append(row)


print("\n")


# ============================================================
# CLOSE CONNECTIONS
# ============================================================

for con in connections.values():
    con.close()


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

Path(OUTPUT).parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 75)
print("QUERY-PLAN EXTRACTION COMPLETE")
print("=" * 75)

print(f"\nQueries processed : {len(df)}")
print(f"Queries with errors: {failed}")

print(
    f"Feature columns   : "
    f"{len(df.columns)}"
)

print("\nOutput:")
print(OUTPUT)


# ============================================================
# FEATURE SUMMARY
# ============================================================

numeric_cols = df.select_dtypes(
    include="number"
).columns

print("\nNumeric feature uniqueness:")

for col in numeric_cols:

    print(
        f"{col:55} "
        f"{df[col].nunique()}"
    )


print("\nMissing values:")

missing = df.isnull().sum()

print(
    missing[missing > 0]
    .to_string()
    if (missing > 0).any()
    else "None"
)


print("\nSample:")
print(
    df.head(3).to_string()
)

print("\n" + "=" * 75)
