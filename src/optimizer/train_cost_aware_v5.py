import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RICH_FILE = ROOT / "data/processed/training_data_v3_rich.csv"
PLAN_FILE = ROOT / "data/processed/query_plan_features_v5.csv"

OUTPUT_FILE = ROOT / "data/processed/adaptive_model_v5_results.csv"
IMPORTANCE_FILE = ROOT / "data/processed/v5_feature_importance.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("COST-AWARE QUERY-PLAN MODEL - V5")
print("=" * 75)

rich = pd.read_csv(RICH_FILE)
plan = pd.read_csv(PLAN_FILE)

print(f"\nRich training rows : {len(rich)}")
print(f"Plan feature rows  : {len(plan)}")


# ============================================================
# MERGE
# ============================================================

df = rich.merge(
    plan,
    on=["query_id", "workload"],
    how="inner"
)

print(f"Merged rows        : {len(df)}")

if len(df) != len(rich):
    raise RuntimeError(
        f"Merge mismatch: expected {len(rich)} rows, got {len(df)}"
    )


# ============================================================
# TARGETS
# ============================================================

targets = [
    "star_time_ms",
    "snowflake_time_ms",
    "constellation_time_ms",
]


missing_targets = [c for c in targets if c not in df.columns]

if missing_targets:
    raise RuntimeError(
        f"Missing target columns: {missing_targets}"
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

# Query-plan columns are explicitly allowed.
plan_prefixes = (
    "star_plan_",
    "snowflake_plan_",
    "fact_constellation_plan_",
    "star_estimated_",
    "snowflake_estimated_",
    "fact_constellation_estimated_",
    "star_cardinality_",
    "snowflake_cardinality_",
    "fact_constellation_cardinality_",
)

plan_columns = [
    c for c in df.columns
    if c.startswith(plan_prefixes)
]


# Existing semantic SQL features from V3 rich data.
# query_length is deliberately excluded because V3 showed
# that it dominated the model and may encode generator/template identity.

semantic_features = [
    "join_count",
    "aggregation_count",
    "predicate_count",
    "group_by_count",
    "tables_used",

    "has_customer",
    "has_product",
    "has_date",
    "has_region",
    "has_segment",
    "has_category",

    "has_date_filter",
    "has_customer_filter",
    "has_product_filter",

    "group_dimension_count",
    "filter_count",
    "join_depth",
]


semantic_features = [
    c for c in semantic_features
    if c in df.columns
]


# Categorical features
categorical_features = [
    c for c in ["workload", "query_type", "aggregation_type"]
    if c in df.columns
]


# Combine
feature_columns = (
    semantic_features
    + categorical_features
    + plan_columns
)

print("\nFeature groups")
print("-" * 75)
print(f"Semantic features   : {len(semantic_features)}")
print(f"Categorical         : {len(categorical_features)}")
print(f"Plan features       : {len(plan_columns)}")
print(f"Total raw features  : {len(feature_columns)}")


# ============================================================
# BUILD MODEL MATRIX
# ============================================================

X = df[feature_columns].copy()
y = df[targets].copy()


# One-hot encode categorical fields
X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False
)


# Safety conversion
X = X.replace([np.inf, -np.inf], np.nan)

if X.isna().any().any():
    print("\nMissing values detected after encoding.")
    print(X.isna().sum()[X.isna().sum() > 0])
    raise RuntimeError("Feature matrix contains missing values.")


print(f"\nFinal model features: {X.shape[1]}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Keep the same basic 80/20 evaluation structure as V4
# while stratifying by workload so the test set contains
# all major workload categories.

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42,
    stratify=df["workload"]
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

test_meta = df.iloc[test_idx].copy()

print("\nDataset split")
print("-" * 75)
print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")


# ============================================================
# TRAIN ONE REGRESSOR PER SCHEMA
# ============================================================

schema_models = {}

predictions = {}

for target in targets:

    print(f"\nTraining model for: {target}")

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train[target])

    pred = model.predict(X_test)

    schema_models[target] = model
    predictions[target] = pred

    mae = mean_absolute_error(
        y_test[target],
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test[target],
            pred
        )
    )

    print(f"MAE  : {mae:.6f} ms")
    print(f"RMSE : {rmse:.6f} ms")


# ============================================================
# PREDICTED COST TABLE
# ============================================================

pred_df = pd.DataFrame({
    "query_id": test_meta["query_id"].values,
    "workload": test_meta["workload"].values,

    "actual_star_ms":
        y_test["star_time_ms"].values,

    "actual_snowflake_ms":
        y_test["snowflake_time_ms"].values,

    "actual_constellation_ms":
        y_test["constellation_time_ms"].values,

    "pred_star_ms":
        predictions["star_time_ms"],

    "pred_snowflake_ms":
        predictions["snowflake_time_ms"],

    "pred_constellation_ms":
        predictions["constellation_time_ms"],
})


# ============================================================
# ACTUAL ORACLE
# ============================================================

actual_matrix = pred_df[
    [
        "actual_star_ms",
        "actual_snowflake_ms",
        "actual_constellation_ms"
    ]
].values

pred_matrix = pred_df[
    [
        "pred_star_ms",
        "pred_snowflake_ms",
        "pred_constellation_ms"
    ]
].values


schema_names = np.array([
    "star",
    "snowflake",
    "fact_constellation"
])


actual_best_idx = np.argmin(
    actual_matrix,
    axis=1
)

predicted_best_idx = np.argmin(
    pred_matrix,
    axis=1
)


pred_df["oracle_schema"] = schema_names[
    actual_best_idx
]

pred_df["adaptive_schema"] = schema_names[
    predicted_best_idx
]


# ============================================================
# ACTUAL LATENCIES OF ADAPTIVE CHOICES
# ============================================================

adaptive_actual = actual_matrix[
    np.arange(len(actual_matrix)),
    predicted_best_idx
]

oracle_actual = np.min(
    actual_matrix,
    axis=1
)

fixed_star = actual_matrix[:, 0]
fixed_snowflake = actual_matrix[:, 1]
fixed_constellation = actual_matrix[:, 2]


pred_df["adaptive_actual_ms"] = adaptive_actual
pred_df["oracle_ms"] = oracle_actual

pred_df["regret_ms"] = (
    adaptive_actual - oracle_actual
)

pred_df["regret_percent"] = np.where(
    oracle_actual > 0,
    100.0 * (adaptive_actual - oracle_actual)
    / oracle_actual,
    0
)


# ============================================================
# OVERALL METRICS
# ============================================================

fixed_star_mean = np.mean(fixed_star)
fixed_snowflake_mean = np.mean(fixed_snowflake)
fixed_constellation_mean = np.mean(fixed_constellation)

adaptive_mean = np.mean(adaptive_actual)
oracle_mean = np.mean(oracle_actual)

best_fixed_mean = min(
    fixed_star_mean,
    fixed_snowflake_mean,
    fixed_constellation_mean
)

adaptive_vs_star = (
    100.0 *
    (fixed_star_mean - adaptive_mean)
    / fixed_star_mean
)

adaptive_vs_best_fixed = (
    100.0 *
    (best_fixed_mean - adaptive_mean)
    / best_fixed_mean
)

oracle_vs_star = (
    100.0 *
    (fixed_star_mean - oracle_mean)
    / fixed_star_mean
)

adaptive_gap_oracle = (
    100.0 *
    (adaptive_mean - oracle_mean)
    / oracle_mean
)

mean_regret = np.mean(
    pred_df["regret_ms"]
)

median_regret = np.median(
    pred_df["regret_ms"]
)

mean_regret_percent = np.mean(
    pred_df["regret_percent"]
)

oracle_match_rate = np.mean(
    pred_df["adaptive_schema"]
    == pred_df["oracle_schema"]
) * 100


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("V5 FINAL RESULTS")
print("=" * 75)

print(f"\nFixed Star              : {fixed_star_mean:.6f} ms")
print(f"Fixed Snowflake         : {fixed_snowflake_mean:.6f} ms")
print(f"Fixed Fact Constellation: {fixed_constellation_mean:.6f} ms")

print(f"\nAdaptive ML             : {adaptive_mean:.6f} ms")
print(f"Oracle                  : {oracle_mean:.6f} ms")

print(f"\nAdaptive vs Star        : {adaptive_vs_star:.3f}%")
print(f"Adaptive vs Best Fixed  : {adaptive_vs_best_fixed:.3f}%")
print(f"Oracle vs Star          : {oracle_vs_star:.3f}%")
print(f"Adaptive gap to Oracle  : {adaptive_gap_oracle:.3f}%")

print(f"\nMean regret             : {mean_regret:.6f} ms")
print(f"Median regret           : {median_regret:.6f} ms")
print(f"Mean regret percent     : {mean_regret_percent:.3f}%")
print(f"Oracle match rate       : {oracle_match_rate:.2f}%")


# ============================================================
# SCHEMA SELECTION DISTRIBUTION
# ============================================================

print("\nAdaptive schema distribution")
print("-" * 75)

print(
    pred_df["adaptive_schema"]
    .value_counts()
    .to_string()
)

print("\nOracle schema distribution")
print("-" * 75)

print(
    pred_df["oracle_schema"]
    .value_counts()
    .to_string()
)


# ============================================================
# WORKLOAD ANALYSIS
# ============================================================

print("\nWorkload-level V5 results")
print("-" * 75)

workload_rows = []

for workload, group in pred_df.groupby("workload"):

    actual = group[
        [
            "actual_star_ms",
            "actual_snowflake_ms",
            "actual_constellation_ms"
        ]
    ].values

    adaptive = group["adaptive_actual_ms"].values
    oracle = group["oracle_ms"].values

    workload_rows.append({
        "workload": workload,
        "queries": len(group),

        "adaptive_mean_ms":
            np.mean(adaptive),

        "star_mean_ms":
            np.mean(actual[:, 0]),

        "snowflake_mean_ms":
            np.mean(actual[:, 1]),

        "constellation_mean_ms":
            np.mean(actual[:, 2]),

        "oracle_mean_ms":
            np.mean(oracle),

        "mean_regret_ms":
            np.mean(group["regret_ms"]),

        "oracle_match_percent":
            np.mean(
                group["adaptive_schema"]
                == group["oracle_schema"]
            ) * 100
    })


workload_df = pd.DataFrame(
    workload_rows
)

print(
    workload_df.to_string(index=False)
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_frames = []

for target, model in schema_models.items():

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
        "target": target
    })

    importance_frames.append(
        importance
    )


importance_df = pd.concat(
    importance_frames,
    ignore_index=True
)

importance_df = importance_df.sort_values(
    ["target", "importance"],
    ascending=[True, False]
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

pred_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# Save workload summary beside predictions
workload_output = (
    ROOT /
    "data/processed/adaptive_model_v5_workload_results.csv"
)

workload_df.to_csv(
    workload_output,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 75)
print("V5 TRAINING COMPLETE")
print("=" * 75)

print(f"\nPredictions:")
print(OUTPUT_FILE)

print(f"\nFeature importance:")
print(IMPORTANCE_FILE)

print(f"\nWorkload results:")
print(workload_output)

print("\nNo benchmark queries were executed.")
print("V5 uses existing benchmark latency labels only for offline evaluation.")
