import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


INPUT = "data/processed/training_data_v3_rich.csv"


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT)

print("=" * 70)
print("ADAPTIVE MULTI-SCHEMA MODEL - V3")
print("=" * 70)

print(f"\nTotal queries: {len(df)}")

print("\nTarget distribution:")
print(df["best_schema"].value_counts())


# ---------------------------------------------------------
# 2. Define features
# ---------------------------------------------------------

numeric_features = [
    "join_count",
    "aggregation_count",
    "predicate_count",
    "group_by_count",
    "tables_used",
    "query_length",
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
    "aggregation_type",
]


X = df[numeric_features + categorical_features]
y = df["best_schema"]


# ---------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------

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


X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


# ---------------------------------------------------------
# 5. Random Forest
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_encoded, y_train)


# ---------------------------------------------------------
# 6. Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test_encoded)


# ---------------------------------------------------------
# 7. Classification metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# 8. Confusion matrix
# ---------------------------------------------------------

labels = [
    "star",
    "snowflake",
    "fact_constellation"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{x}" for x in labels],
    columns=[f"Predicted_{x}" for x in labels]
)

print("\nConfusion Matrix:")
print(cm_df)


# ---------------------------------------------------------
# 9. Attach predictions to test data
# ---------------------------------------------------------

test_results = df.loc[X_test.index].copy()

test_results["predicted_schema"] = y_pred


# ---------------------------------------------------------
# 10. Calculate predicted-schema latency
# ---------------------------------------------------------

def get_predicted_time(row):

    if row["predicted_schema"] == "star":
        return row["star_time_ms"]

    elif row["predicted_schema"] == "snowflake":
        return row["snowflake_time_ms"]

    elif row["predicted_schema"] == "fact_constellation":
        return row["constellation_time_ms"]

    return np.nan


test_results["predicted_time_ms"] = test_results.apply(
    get_predicted_time,
    axis=1
)


# ---------------------------------------------------------
# 11. Baselines
# ---------------------------------------------------------

test_results["fixed_star_time_ms"] = test_results["star_time_ms"]

test_results["oracle_time_ms"] = test_results["best_time_ms"]


# ---------------------------------------------------------
# 12. Model regret
# ---------------------------------------------------------

test_results["model_regret_ms"] = (
    test_results["predicted_time_ms"]
    - test_results["oracle_time_ms"]
)


# ---------------------------------------------------------
# 13. Improvement calculations
# ---------------------------------------------------------

fixed_star_mean = test_results["fixed_star_time_ms"].mean()
model_mean = test_results["predicted_time_ms"].mean()
oracle_mean = test_results["oracle_time_ms"].mean()

fixed_star_improvement = (
    (fixed_star_mean - model_mean)
    / fixed_star_mean
) * 100

oracle_gap = (
    (model_mean - oracle_mean)
    / oracle_mean
) * 100


# ---------------------------------------------------------
# 14. Results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LATENCY RESULTS")
print("=" * 70)

print(f"\nFixed Star mean latency      : {fixed_star_mean:.4f} ms")
print(f"ML-selected mean latency     : {model_mean:.4f} ms")
print(f"Oracle mean latency          : {oracle_mean:.4f} ms")

print(
    f"\nML improvement over Star     : "
    f"{fixed_star_improvement:.2f}%"
)

print(
    f"ML gap from Oracle           : "
    f"{oracle_gap:.2f}%"
)

print(
    f"Mean model regret            : "
    f"{test_results['model_regret_ms'].mean():.4f} ms"
)

print(
    f"Median model regret          : "
    f"{test_results['model_regret_ms'].median():.4f} ms"
)


# ---------------------------------------------------------
# 15. Schema selection distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTED SCHEMA DISTRIBUTION")
print("=" * 70)

print(
    test_results["predicted_schema"]
    .value_counts()
)


# ---------------------------------------------------------
# 16. Feature importance
# ---------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.head(15).to_string(index=False)
)


# ---------------------------------------------------------
# 17. Save results
# ---------------------------------------------------------

output_file = (
    "data/processed/"
    "adaptive_model_v3_results.csv"
)

test_results.to_csv(
    output_file,
    index=False
)

print("\nSaved predictions to:")
print(output_file)

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)
