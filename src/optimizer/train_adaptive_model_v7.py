import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.generate_data_v7 import get_scale_label
from src.benchmark.run_benchmark_v7 import bootstrap_ci

RESULTS_DIR = "results"
QUERY_SUMMARY_FILE = os.path.join(RESULTS_DIR, "scale_query_summary.csv")
ADAPTIVE_QUERY_FILE = os.path.join(RESULTS_DIR, "adaptive_model_query_results.csv")
ADAPTIVE_SUMMARY_FILE = os.path.join(RESULTS_DIR, "adaptive_model_summary.csv")
FEATURE_IMP_FILE = os.path.join(RESULTS_DIR, "feature_importance.csv")

SCHEMAS = ["star", "snowflake", "fact_constellation"]


def build_feature_matrices(train_df, test_df, feature_cols, cat_cols):
    """One-hot encode categorical features and assemble dense feature matrices."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat = encoder.fit_transform(train_df[cat_cols])
    test_cat = encoder.transform(test_df[cat_cols])

    train_num = train_df[feature_cols].fillna(0.0).astype(float).values
    test_num = test_df[feature_cols].fillna(0.0).astype(float).values

    X_train = np.hstack([train_num, train_cat])
    X_test = np.hstack([test_num, test_cat])

    all_names = list(feature_cols) + list(encoder.get_feature_names_out(cat_cols))
    return X_train, X_test, all_names


def evaluate_adaptive_model(scale_rows=100_000, features_csv=None, append_results=True):
    """
    Train and evaluate the adaptive schema selection model for a given scale
    using rigorous Leave-One-Workload-Out cross-validation.
    Ensures zero target leakage and strict pre-execution feature reliance.
    """
    scale_label = get_scale_label(scale_rows)
    if features_csv is None:
        features_csv = f"data/processed/features_{scale_label.lower()}.csv"

    print("=" * 80)
    print(f"ADAPTIVE CONTROLLER EVALUATION - SCALE {scale_label} ({scale_rows:,} rows)")
    print("=" * 80)

    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features file not found: {features_csv}")
    if not os.path.exists(QUERY_SUMMARY_FILE):
        raise FileNotFoundError(f"Query summary not found: {QUERY_SUMMARY_FILE}. Run benchmark first.")

    features_df = pd.read_csv(features_csv)
    query_summary = pd.read_csv(QUERY_SUMMARY_FILE)

    # Filter query summary to the current dataset size
    qs_scale = query_summary[query_summary["dataset_size"] == scale_rows].copy()
    if qs_scale.empty:
        raise ValueError(f"No benchmark data found for dataset_size={scale_rows} in {QUERY_SUMMARY_FILE}")

    # Merge features with measured performance
    df = features_df.merge(
        qs_scale[[
            "query_id", "workload", "star_mean_ms", "snowflake_mean_ms",
            "fc_mean_ms", "oracle_schema", "oracle_mean_ms"
        ]],
        on=["query_id", "workload"],
        how="inner"
    )

    print(f"Loaded {len(df)} query observations for scale {scale_label}.")

    # Define targets: Relative Cost vs Reference Star Schema
    # Target values: < 1.0 means candidate is faster than Star; > 1.0 means candidate is slower.
    df["target_snow_relative"] = df["snowflake_mean_ms"] / df["star_mean_ms"]
    df["target_fc_relative"] = df["fc_mean_ms"] / df["star_mean_ms"]

    # Identify candidate features (strictly pre-execution!)
    exclude_cols = {
        "query_id", "dataset_size", "scale_label", "workload", "query_type",
        "star_mean_ms", "snowflake_mean_ms", "fc_mean_ms",
        "oracle_schema", "oracle_mean_ms", "target_snow_relative", "target_fc_relative",
        "query_length"  # intentionally exclude query length to avoid generator bias
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = ["workload"]
    if "query_type" in df.columns and df["query_type"].nunique() > 1:
        cat_cols.append("query_type")

    print(f"Model Input Features: {len(feature_cols)} numeric + {len(cat_cols)} categorical.")
    print("Cross-Validation: Leave-One-Workload-Out across workloads.")

    workloads = sorted(df["workload"].unique())
    fold_results = []
    feature_importances_snow = []
    feature_importances_fc = []

    for held_out in workloads:
        train_df = df[df["workload"] != held_out].copy()
        test_df = df[df["workload"] == held_out].copy()

        X_train, X_test, feature_names = build_feature_matrices(train_df, test_df, feature_cols, cat_cols)

        # 1. Train Snowflake Relative-Cost Model
        rf_snow = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
        )
        rf_snow.fit(X_train, train_df["target_snow_relative"].values)
        pred_snow_rel = rf_snow.predict(X_test)
        feature_importances_snow.append(rf_snow.feature_importances_)

        # 2. Train Fact Constellation Relative-Cost Model
        rf_fc = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
        )
        rf_fc.fit(X_train, train_df["target_fc_relative"].values)
        pred_fc_rel = rf_fc.predict(X_test)
        feature_importances_fc.append(rf_fc.feature_importances_)

        # Pre-execution decision rule:
        # Reference Star cost is 1.0 (by definition of relative cost).
        # We select the schema with the minimal predicted relative cost.
        pred_matrix = np.column_stack([
            np.ones(len(test_df)),  # Star relative cost = 1.0
            pred_snow_rel,          # Predicted Snowflake relative cost
            pred_fc_rel             # Predicted FC relative cost
        ])
        pred_indices = np.argmin(pred_matrix, axis=1)
        pred_schemas = [SCHEMAS[i] for i in pred_indices]

        # Extract measured latencies
        test_star_lat = test_df["star_mean_ms"].values
        test_snow_lat = test_df["snowflake_mean_ms"].values
        test_fc_lat = test_df["fc_mean_ms"].values
        test_oracle_lat = test_df["oracle_mean_ms"].values
        test_oracle_schema = test_df["oracle_schema"].values

        actual_lat_matrix = np.column_stack([test_star_lat, test_snow_lat, test_fc_lat])
        chosen_latencies = actual_lat_matrix[np.arange(len(test_df)), pred_indices]

        for i, (_, row) in enumerate(test_df.iterrows()):
            fold_results.append({
                "dataset_size": scale_rows,
                "scale_label": scale_label,
                "query_id": row["query_id"],
                "workload": row["workload"],
                "held_out_workload": held_out,
                "predicted_schema": pred_schemas[i],
                "oracle_schema": test_oracle_schema[i],
                "oracle_match": int(pred_schemas[i] == test_oracle_schema[i]),
                "pred_snow_rel_cost": pred_snow_rel[i],
                "pred_fc_rel_cost": pred_fc_rel[i],
                "adaptive_latency_ms": chosen_latencies[i],
                "star_latency_ms": test_star_lat[i],
                "snowflake_latency_ms": test_snow_lat[i],
                "fc_latency_ms": test_fc_lat[i],
                "oracle_latency_ms": test_oracle_lat[i],
                "regret_ms": chosen_latencies[i] - test_oracle_lat[i]
            })

    results_df = pd.DataFrame(fold_results)

    # Save feature importances
    avg_imp_snow = np.mean(feature_importances_snow, axis=0)
    avg_imp_fc = np.mean(feature_importances_fc, axis=0)
    feat_imp_df = pd.DataFrame({
        "scale_label": scale_label,
        "feature": feature_names,
        "importance_snowflake_rel": avg_imp_snow,
        "importance_fc_rel": avg_imp_fc,
        "importance_mean": (avg_imp_snow + avg_imp_fc) / 2.0
    }).sort_values("importance_mean", ascending=False)
    feat_imp_df.to_csv(FEATURE_IMP_FILE, index=False)

    # Save / append adaptive query-level results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if append_results and os.path.exists(ADAPTIVE_QUERY_FILE):
        existing_aq = pd.read_csv(ADAPTIVE_QUERY_FILE)
        existing_aq = existing_aq[existing_aq["dataset_size"] != scale_rows]
        combined_aq = pd.concat([existing_aq, results_df], ignore_index=True)
        combined_aq.to_csv(ADAPTIVE_QUERY_FILE, index=False)
    else:
        results_df.to_csv(ADAPTIVE_QUERY_FILE, index=False)

    # Compute Statistical Aggregates
    star_mean = results_df["star_latency_ms"].mean()
    snow_mean = results_df["snowflake_latency_ms"].mean()
    fc_mean = results_df["fc_latency_ms"].mean()
    best_fixed_mean = min(star_mean, snow_mean, fc_mean)
    best_fixed_name = min(
        [("star", star_mean), ("snowflake", snow_mean), ("fact_constellation", fc_mean)],
        key=lambda x: x[1]
    )[0]

    adaptive_mean = results_df["adaptive_latency_ms"].mean()
    adaptive_median = results_df["adaptive_latency_ms"].median()
    adaptive_std = results_df["adaptive_latency_ms"].std()

    oracle_mean = results_df["oracle_latency_ms"].mean()
    mean_regret = results_df["regret_ms"].mean()
    match_rate = results_df["oracle_match"].mean() * 100.0

    # Percentage differences
    adaptive_vs_star_pct = ((star_mean - adaptive_mean) / star_mean) * 100.0
    adaptive_vs_best_pct = ((best_fixed_mean - adaptive_mean) / best_fixed_mean) * 100.0
    oracle_vs_star_pct = ((star_mean - oracle_mean) / star_mean) * 100.0
    adaptive_gap_oracle_pct = ((adaptive_mean - oracle_mean) / oracle_mean) * 100.0

    # 95% Bootstrap Confidence Intervals for paired differences
    # Positive difference indicates Adaptive is faster.
    ci_star = bootstrap_ci(results_df["star_latency_ms"].values - results_df["adaptive_latency_ms"].values)
    best_fixed_vals = results_df[f"{best_fixed_name}_latency_ms" if best_fixed_name != "fact_constellation" else "fc_latency_ms"].values
    ci_best = bootstrap_ci(best_fixed_vals - results_df["adaptive_latency_ms"].values)

    summary_row = {
        "dataset_size": scale_rows,
        "scale_label": scale_label,
        "queries": len(results_df),
        "star_mean_ms": star_mean,
        "snowflake_mean_ms": snow_mean,
        "fc_mean_ms": fc_mean,
        "best_fixed_schema": best_fixed_name,
        "best_fixed_mean_ms": best_fixed_mean,
        "adaptive_mean_ms": adaptive_mean,
        "adaptive_median_ms": adaptive_median,
        "adaptive_std_ms": adaptive_std,
        "oracle_mean_ms": oracle_mean,
        "adaptive_vs_star_percent": adaptive_vs_star_pct,
        "adaptive_vs_best_fixed_percent": adaptive_vs_best_pct,
        "oracle_vs_star_percent": oracle_vs_star_pct,
        "adaptive_gap_to_oracle_percent": adaptive_gap_oracle_pct,
        "oracle_match_rate_percent": match_rate,
        "mean_regret_ms": mean_regret,
        "ci_adaptive_star_lower_ms": ci_star[0],
        "ci_adaptive_star_upper_ms": ci_star[1],
        "ci_adaptive_best_lower_ms": ci_best[0],
        "ci_adaptive_best_upper_ms": ci_best[1],
    }

    summary_df = pd.DataFrame([summary_row])
    if append_results and os.path.exists(ADAPTIVE_SUMMARY_FILE):
        existing_as = pd.read_csv(ADAPTIVE_SUMMARY_FILE)
        existing_as = existing_as[existing_as["dataset_size"] != scale_rows]
        combined_as = pd.concat([existing_as, summary_df], ignore_index=True)
        combined_as.sort_values("dataset_size", inplace=True)
        combined_as.to_csv(ADAPTIVE_SUMMARY_FILE, index=False)
    else:
        summary_df.to_csv(ADAPTIVE_SUMMARY_FILE, index=False)

    print("\nAdaptive Controller Results:")
    print(f"  Fixed Star Mean         : {star_mean:.4f} ms")
    print(f"  Fixed Snowflake Mean    : {snow_mean:.4f} ms")
    print(f"  Fixed FC Mean           : {fc_mean:.4f} ms")
    print(f"  Best Fixed ({best_fixed_name}) : {best_fixed_mean:.4f} ms")
    print(f"  Adaptive Mean           : {adaptive_mean:.4f} ms")
    print(f"  Oracle Upper Bound Mean : {oracle_mean:.4f} ms")
    print("-" * 60)
    print(f"  Adaptive vs Star        : {adaptive_vs_star_pct:+.3f}% (95% CI: [{ci_star[0]:.4f}, {ci_star[1]:.4f}] ms)")
    print(f"  Adaptive vs Best Fixed  : {adaptive_vs_best_pct:+.3f}% (95% CI: [{ci_best[0]:.4f}, {ci_best[1]:.4f}] ms)")
    print(f"  Oracle vs Star          : {oracle_vs_star_pct:+.3f}%")
    print(f"  Adaptive Gap to Oracle  : {adaptive_gap_oracle_pct:.3f}%")
    print(f"  Oracle Match Rate       : {match_rate:.2f}%")
    print(f"  Mean Regret             : {mean_regret:.4f} ms")
    print("=" * 80)

    return summary_row


def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptive schema selector.")
    parser.add_argument("--rows", type=int, default=100_000, help="Dataset size in sales rows")
    parser.add_argument("--features", type=str, default=None, help="Features CSV path")
    args = parser.parse_args()

    evaluate_adaptive_model(scale_rows=args.rows, features_csv=args.features)


if __name__ == "__main__":
    main()
