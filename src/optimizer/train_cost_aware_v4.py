import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT = "data/processed/training_data_v3_rich.csv"
OUTPUT = "data/processed/adaptive_model_v4_results.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 75)
print("COST-AWARE ADAPTIVE MULTI-SCHEMA MODEL - V4")
print("=" * 75)

print(f"\nTotal queries: {len(df)}")


# ============================================================
# 2. INPUT FEATURES
# ============================================================
# Deliberately exclude:
# - query_length: can encode generator/template identity
# - all measured timing columns: would cause target leakage
# - best_schema / best_time_ms / oracle improvement: targets

numeric_features = [
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

categorical_features = [
    "aggregation_type"
]

feature_columns = numeric_features + categorical_features

X = df[feature_columns]

# Three independent latency targets
targets = {
    "star": "star_time_ms",
    "snowflake": "snowflake_time_ms",
    "fact_constellation": "constellation_time_ms",
}


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["best_schema"]
)

X_train = df.loc[train_idx, feature_columns]
X_test = df.loc[test_idx, feature_columns]

print(f"Training queries: {len(X_train)}")
print(f"Testing queries : {len(X_test)}")


# ============================================================
# 4. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 5. TRAIN ONE REGRESSOR PER SCHEMA
# ============================================================

models = {}
predictions = {}

for schema, target_column in targets.items():

    print("\n" + "-" * 75)
    print(f"Training latency model: {schema}")
    print("-" * 75)

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    model.fit(
        X_train,
        df.loc[train_idx, target_column]
    )

    pred = model.predict(X_test)

    models[schema] = model
    predictions[schema] = pred

    actual = df.loc[test_idx, target_column].values

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    print(f"MAE  : {mae:.6f} ms")
    print(f"RMSE : {rmse:.6f} ms")


# ============================================================
# 6. CREATE PREDICTION DATAFRAME
# ============================================================

results = df.loc[test_idx].copy()

results["predicted_star_ms"] = predictions["star"]
results["predicted_snowflake_ms"] = predictions["snowflake"]
results["predicted_constellation_ms"] = predictions[
    "fact_constellation"
]

results["predicted_fact_constellation_ms"] = predictions[
    "fact_constellation"
]


# ============================================================
# 7. SELECT SCHEMA USING PREDICTED COST
# ============================================================

prediction_matrix = np.column_stack([
    results["predicted_star_ms"],
    results["predicted_snowflake_ms"],
    results["predicted_constellation_ms"]
])

schema_names = np.array([
    "star",
    "snowflake",
    "fact_constellation"
])

best_indices = np.argmin(
    prediction_matrix,
    axis=1
)

results["predicted_schema"] = schema_names[best_indices]


# ============================================================
# 8. ACTUAL LATENCY OF THE SELECTED SCHEMA
# ============================================================

def actual_selected_latency(row):

    schema = row["predicted_schema"]

    if schema == "star":
        return row["star_time_ms"]

    if schema == "snowflake":
        return row["snowflake_time_ms"]

    return row["constellation_time_ms"]


results["adaptive_time_ms"] = results.apply(
    actual_selected_latency,
    axis=1
)


# ============================================================
# 9. FIXED BASELINES
# ============================================================

results["fixed_star_ms"] = results["star_time_ms"]

results["fixed_snowflake_ms"] = results["snowflake_time_ms"]

results["fixed_constellation_ms"] = (
    results["constellation_time_ms"]
)

results["oracle_time_ms"] = results["best_time_ms"]


# ============================================================
# 10. REGRET
# ============================================================

results["adaptive_regret_ms"] = (
    results["adaptive_time_ms"]
    - results["oracle_time_ms"]
)

results["adaptive_regret_percent"] = (
    results["adaptive_regret_ms"]
    / results["oracle_time_ms"]
) * 100


# ============================================================
# 11. MATCH WITH ORACLE
# ============================================================

results["oracle_match"] = (
    results["predicted_schema"]
    == results["best_schema"]
)


# ============================================================
# 12. BASELINE MEANS
# ============================================================

star_mean = results["fixed_star_ms"].mean()
snowflake_mean = results["fixed_snowflake_ms"].mean()
constellation_mean = results["fixed_constellation_ms"].mean()

adaptive_mean = results["adaptive_time_ms"].mean()
oracle_mean = results["oracle_time_ms"].mean()

best_fixed_mean = min(
    star_mean,
    snowflake_mean,
    constellation_mean
)


# ============================================================
# 13. IMPROVEMENTS
# ============================================================

adaptive_vs_star = (
    (star_mean - adaptive_mean)
    / star_mean
) * 100

adaptive_vs_best_fixed = (
    (best_fixed_mean - adaptive_mean)
    / best_fixed_mean
) * 100

oracle_vs_star = (
    (star_mean - oracle_mean)
    / star_mean
) * 100

oracle_gap = (
    (adaptive_mean - oracle_mean)
    / oracle_mean
) * 100


# ============================================================
# 14. RESULTS
# ============================================================

print("\n" + "=" * 75)
print("V4 COST-AWARE RESULTS")
print("=" * 75)

print("\nActual test-set latency:")
print(f"Fixed Star              : {star_mean:.6f} ms")
print(f"Fixed Snowflake         : {snowflake_mean:.6f} ms")
print(f"Fixed Fact Constellation: {constellation_mean:.6f} ms")
print(f"Adaptive ML             : {adaptive_mean:.6f} ms")
print(f"Oracle                  : {oracle_mean:.6f} ms")

print("\nImprovement:")
print(f"Adaptive vs Star        : {adaptive_vs_star:.3f}%")
print(f"Adaptive vs Best Fixed  : {adaptive_vs_best_fixed:.3f}%")
print(f"Oracle vs Star          : {oracle_vs_star:.3f}%")
print(f"Adaptive gap to Oracle  : {oracle_gap:.3f}%")

print("\nRegret:")
print(
    f"Mean regret             : "
    f"{results['adaptive_regret_ms'].mean():.6f} ms"
)

print(
    f"Median regret           : "
    f"{results['adaptive_regret_ms'].median():.6f} ms"
)

print(
    f"Mean regret             : "
    f"{results['adaptive_regret_percent'].mean():.3f}%"
)

print(
    f"Oracle match rate       : "
    f"{results['oracle_match'].mean() * 100:.2f}%"
)


# ============================================================
# 15. SCHEMA SELECTION
# ============================================================

print("\n" + "=" * 75)
print("ADAPTIVE SCHEMA SELECTION")
print("=" * 75)

print(
    results["predicted_schema"]
    .value_counts()
)


# ============================================================
# 16. ACTUAL ORACLE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("ORACLE SCHEMA DISTRIBUTION")
print("=" * 75)

print(
    results["best_schema"]
    .value_counts()
)


# ============================================================
# 17. ADAPTIVE PERFORMANCE BY WORKLOAD
# ============================================================

workload_summary = (
    results
    .groupby("workload")
    .agg(
        queries=("query_id", "count"),
        adaptive_ms=("adaptive_time_ms", "mean"),
        star_ms=("star_time_ms", "mean"),
        snowflake_ms=("snowflake_time_ms", "mean"),
        constellation_ms=("constellation_time_ms", "mean"),
        oracle_ms=("oracle_time_ms", "mean"),
        regret_ms=("adaptive_regret_ms", "mean"),
        oracle_match=("oracle_match", "mean")
    )
)

workload_summary["oracle_match"] *= 100

print("\n" + workload_summary.round(4).to_string())


# ============================================================
# 18. MODEL ERROR PER SCHEMA
# ============================================================

print("\n" + "=" * 75)
print("LATENCY PREDICTION ERROR")
print("=" * 75)

for schema, target_column in targets.items():

    pred_column = f"predicted_{schema}_ms"

    actual = results[target_column]
    predicted = results[pred_column]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    print(
        f"{schema:20} "
        f"MAE={mae:.6f} ms  "
        f"RMSE={rmse:.6f} ms"
    )


# ============================================================
# 19. SAVE
# ============================================================

results.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved V4 results to:")
print(OUTPUT)

print("\n" + "=" * 75)
print("V4 COMPLETE")
print("=" * 75)
