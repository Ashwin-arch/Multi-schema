import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder


RICH_FILE = "data/processed/training_data_v3_rich.csv"
PLAN_FILE = "data/processed/query_plan_features_v5.csv"

RESULT_FILE = "data/processed/v5_leave_one_workload_out_results.csv"
SUMMARY_FILE = "data/processed/v5_leave_one_workload_out_summary.csv"


SEMANTIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "workload",
    "query_type",
    "aggregation_type",
]

TARGETS = [
    "star_time_ms",
    "snowflake_time_ms",
    "constellation_time_ms",
]

SCHEMAS = [
    "star",
    "snowflake",
    "fact_constellation",
]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_feature_matrix(train_df, test_df):
    """
    Build a leakage-safe feature matrix.

    The OneHotEncoder is fitted ONLY on the training data.
    """

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_cat = encoder.fit_transform(
        train_df[CATEGORICAL_FEATURES]
    )

    test_cat = encoder.transform(
        test_df[CATEGORICAL_FEATURES]
    )

    train_semantic = train_df[SEMANTIC_FEATURES].astype(float).values
    test_semantic = test_df[SEMANTIC_FEATURES].astype(float).values

    plan_features = [
        c for c in train_df.columns
        if c not in (
            ["query_id"]
            + SEMANTIC_FEATURES
            + CATEGORICAL_FEATURES
            + TARGETS
            + [
                "best_schema",
                "best_time_ms",
                "oracle_improvement_percent",
            ]
        )
    ]

    train_plan = train_df[plan_features].astype(float).values
    test_plan = test_df[plan_features].astype(float).values

    X_train = np.hstack([
        train_semantic,
        train_cat,
        train_plan
    ])

    X_test = np.hstack([
        test_semantic,
        test_cat,
        test_plan
    ])

    return X_train, X_test, plan_features, encoder


def main():

    print("=" * 80)
    print("V5 LEAVE-ONE-WORKLOAD-OUT GENERALIZATION EVALUATION")
    print("=" * 80)

    rich = pd.read_csv(RICH_FILE)
    plan = pd.read_csv(PLAN_FILE)

    print(f"\nRich training rows : {len(rich)}")
    print(f"Plan feature rows  : {len(plan)}")

    df = rich.merge(
        plan,
        on=["query_id", "workload"],
        how="inner",
        suffixes=("", "_plan")
    )

    print(f"Merged rows        : {len(df)}")

    # ---------------------------------------------------------
    # Remove duplicated columns created by merge, if any.
    # ---------------------------------------------------------

    duplicated_suffixes = [
        c for c in df.columns
        if c.endswith("_plan")
    ]

    for col in duplicated_suffixes:
        base = col[:-5]

        if base in df.columns:
            df.drop(columns=[col], inplace=True)

    # ---------------------------------------------------------
    # Remove query_length intentionally.
    # ---------------------------------------------------------

    if "query_length" in df.columns:
        df.drop(columns=["query_length"], inplace=True)

    # ---------------------------------------------------------
    # Remove leakage / target-derived columns.
    # ---------------------------------------------------------

    leakage_columns = [
        "best_schema",
        "best_time_ms",
        "oracle_improvement_percent",
    ]

    for col in leakage_columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # ---------------------------------------------------------
    # Detect plan features.
    # ---------------------------------------------------------

    excluded = set(
        ["query_id"]
        + SEMANTIC_FEATURES
        + CATEGORICAL_FEATURES
        + TARGETS
    )

    plan_features = [
        c for c in df.columns
        if c not in excluded
    ]

    # Only numerical plan features.
    plan_features = [
        c for c in plan_features
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"\nSemantic features : {len(SEMANTIC_FEATURES)}")
    print(f"Categorical       : {len(CATEGORICAL_FEATURES)}")
    print(f"Plan features     : {len(plan_features)}")

    print("\nWorkload distribution:")
    print(df["workload"].value_counts().sort_index())

    # ---------------------------------------------------------
    # Validate missing values.
    # ---------------------------------------------------------

    feature_columns = (
        SEMANTIC_FEATURES
        + CATEGORICAL_FEATURES
        + plan_features
        + TARGETS
    )

    missing = df[feature_columns].isnull().sum()

    if missing.sum() > 0:
        print("\nWARNING: Missing values detected:")
        print(missing[missing > 0])

        # Numeric imputation using training-independent
        # median would technically leak test information.
        # Therefore fail instead of silently introducing leakage.
        raise ValueError(
            "Missing values detected. Fix them before evaluation."
        )

    # ---------------------------------------------------------
    # Leave-one-workload-out evaluation.
    # ---------------------------------------------------------

    workloads = sorted(df["workload"].unique())

    all_results = []
    summary_rows = []

    for held_out in workloads:

        print("\n" + "-" * 80)
        print(f"HOLDING OUT WORKLOAD: {held_out}")
        print("-" * 80)

        train_df = df[df["workload"] != held_out].copy()
        test_df = df[df["workload"] == held_out].copy()

        print(f"Training rows : {len(train_df)}")
        print(f"Testing rows  : {len(test_df)}")

        X_train, X_test, _, _ = build_feature_matrix(
            train_df,
            test_df
        )

        print(f"Final feature count: {X_train.shape[1]}")

        predictions = {}

        for target in TARGETS:

            model = RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train,
                train_df[target].values
            )

            pred = model.predict(X_test)

            predictions[target] = pred

            mae = mean_absolute_error(
                test_df[target].values,
                pred
            )

            error_rmse = rmse(
                test_df[target].values,
                pred
            )

            print(
                f"{target:25s} "
                f"MAE={mae:.6f} ms | "
                f"RMSE={error_rmse:.6f} ms"
            )

        # -----------------------------------------------------
        # Predicted adaptive choice.
        # -----------------------------------------------------

        prediction_matrix = np.column_stack([
            predictions["star_time_ms"],
            predictions["snowflake_time_ms"],
            predictions["constellation_time_ms"],
        ])

        predicted_indices = np.argmin(
            prediction_matrix,
            axis=1
        )

        predicted_schema = [
            SCHEMAS[i]
            for i in predicted_indices
        ]

        # -----------------------------------------------------
        # Actual measured latency matrix.
        # -----------------------------------------------------

        actual_matrix = np.column_stack([
            test_df["star_time_ms"].values,
            test_df["snowflake_time_ms"].values,
            test_df["constellation_time_ms"].values,
        ])

        oracle_indices = np.argmin(
            actual_matrix,
            axis=1
        )

        oracle_schema = [
            SCHEMAS[i]
            for i in oracle_indices
        ]

        # -----------------------------------------------------
        # Adaptive latency.
        # -----------------------------------------------------

        adaptive_latency = actual_matrix[
            np.arange(len(test_df)),
            predicted_indices
        ]

        oracle_latency = np.min(
            actual_matrix,
            axis=1
        )

        star_latency = test_df["star_time_ms"].values
        snowflake_latency = test_df["snowflake_time_ms"].values
        constellation_latency = test_df["constellation_time_ms"].values

        best_fixed_latency = np.minimum(
            np.minimum(
                star_latency,
                snowflake_latency
            ),
            constellation_latency
        )

        # -----------------------------------------------------
        # Regret.
        # -----------------------------------------------------

        regret = adaptive_latency - oracle_latency

        # -----------------------------------------------------
        # Per-query results.
        # -----------------------------------------------------

        for i, (_, row) in enumerate(test_df.iterrows()):

            result = {
                "query_id": row["query_id"],
                "workload": row["workload"],
                "held_out_workload": held_out,

                "predicted_schema": predicted_schema[i],
                "oracle_schema": oracle_schema[i],

                "predicted_star_ms": prediction_matrix[i, 0],
                "predicted_snowflake_ms": prediction_matrix[i, 1],
                "predicted_constellation_ms": prediction_matrix[i, 2],

                "actual_star_ms": star_latency[i],
                "actual_snowflake_ms": snowflake_latency[i],
                "actual_constellation_ms": constellation_latency[i],

                "adaptive_latency_ms": adaptive_latency[i],
                "oracle_latency_ms": oracle_latency[i],
                "regret_ms": regret[i],

                "oracle_match": int(
                    predicted_schema[i] == oracle_schema[i]
                ),
            }

            all_results.append(result)

        # -----------------------------------------------------
        # Summary.
        # -----------------------------------------------------

        adaptive_mean = np.mean(adaptive_latency)
        oracle_mean = np.mean(oracle_latency)

        star_mean = np.mean(star_latency)
        snowflake_mean = np.mean(snowflake_latency)
        constellation_mean = np.mean(constellation_latency)

        best_fixed_mean = min(
            star_mean,
            snowflake_mean,
            constellation_mean
        )

        adaptive_vs_star = (
            (star_mean - adaptive_mean)
            / star_mean
            * 100
        )

        adaptive_vs_best_fixed = (
            (best_fixed_mean - adaptive_mean)
            / best_fixed_mean
            * 100
        )

        oracle_vs_star = (
            (star_mean - oracle_mean)
            / star_mean
            * 100
        )

        adaptive_gap_to_oracle = (
            (adaptive_mean - oracle_mean)
            / oracle_mean
            * 100
        )

        mean_regret = np.mean(regret)
        median_regret = np.median(regret)

        mean_regret_percent = (
            np.mean(
                regret / oracle_latency * 100
            )
        )

        match_rate = (
            np.mean(
                np.array(predicted_schema)
                == np.array(oracle_schema)
            )
            * 100
        )

        summary = {
            "held_out_workload": held_out,
            "test_queries": len(test_df),

            "fixed_star_ms": star_mean,
            "fixed_snowflake_ms": snowflake_mean,
            "fixed_constellation_ms": constellation_mean,
            "best_fixed_ms": best_fixed_mean,

            "adaptive_ml_ms": adaptive_mean,
            "oracle_ms": oracle_mean,

            "adaptive_vs_star_percent": adaptive_vs_star,
            "adaptive_vs_best_fixed_percent":
                adaptive_vs_best_fixed,
            "oracle_vs_star_percent": oracle_vs_star,
            "adaptive_gap_to_oracle_percent":
                adaptive_gap_to_oracle,

            "mean_regret_ms": mean_regret,
            "median_regret_ms": median_regret,
            "mean_regret_percent": mean_regret_percent,

            "oracle_match_rate_percent": match_rate,
        }

        summary_rows.append(summary)

        print("\nLatency:")
        print(f"  Fixed Star              : {star_mean:.6f} ms")
        print(f"  Fixed Snowflake         : {snowflake_mean:.6f} ms")
        print(f"  Fixed Fact Constellation: {constellation_mean:.6f} ms")
        print(f"  Best Fixed              : {best_fixed_mean:.6f} ms")
        print(f"  Adaptive ML             : {adaptive_mean:.6f} ms")
        print(f"  Oracle                  : {oracle_mean:.6f} ms")

        print("\nPerformance:")
        print(
            f"  Adaptive vs Star        : "
            f"{adaptive_vs_star:.3f}%"
        )
        print(
            f"  Adaptive vs Best Fixed  : "
            f"{adaptive_vs_best_fixed:.3f}%"
        )
        print(
            f"  Oracle vs Star          : "
            f"{oracle_vs_star:.3f}%"
        )
        print(
            f"  Adaptive gap to Oracle  : "
            f"{adaptive_gap_to_oracle:.3f}%"
        )

        print("\nRegret:")
        print(f"  Mean regret             : {mean_regret:.6f} ms")
        print(
            f"  Median regret           : "
            f"{median_regret:.6f} ms"
        )
        print(
            f"  Mean regret percent     : "
            f"{mean_regret_percent:.3f}%"
        )
        print(
            f"  Oracle match rate       : "
            f"{match_rate:.2f}%"
        )

        print("\nPredicted schema distribution:")
        print(
            pd.Series(predicted_schema)
            .value_counts()
            .reindex(SCHEMAS, fill_value=0)
        )

        print("\nOracle schema distribution:")
        print(
            pd.Series(oracle_schema)
            .value_counts()
            .reindex(SCHEMAS, fill_value=0)
        )

    # ---------------------------------------------------------
    # Save results.
    # ---------------------------------------------------------

    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Overall held-out performance.
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("OVERALL LEAVE-ONE-WORKLOAD-OUT RESULTS")
    print("=" * 80)

    print("\nPer-workload summary:")
    print(
        summary_df[
            [
                "held_out_workload",
                "test_queries",
                "best_fixed_ms",
                "adaptive_ml_ms",
                "oracle_ms",
                "adaptive_vs_best_fixed_percent",
                "adaptive_gap_to_oracle_percent",
                "mean_regret_ms",
                "oracle_match_rate_percent",
            ]
        ].to_string(index=False)
    )

    total_queries = len(results_df)

    overall_adaptive = results_df[
        "adaptive_latency_ms"
    ].mean()

    overall_oracle = results_df[
        "oracle_latency_ms"
    ].mean()

    overall_star = results_df[
        "actual_star_ms"
    ].mean()

    overall_snowflake = results_df[
        "actual_snowflake_ms"
    ].mean()

    overall_constellation = results_df[
        "actual_constellation_ms"
    ].mean()

    overall_best_fixed = min(
        overall_star,
        overall_snowflake,
        overall_constellation
    )

    overall_regret = (
        results_df["regret_ms"].mean()
    )

    overall_median_regret = (
        results_df["regret_ms"].median()
    )

    overall_match = (
        results_df["oracle_match"].mean()
        * 100
    )

    print(f"\nTotal held-out queries : {total_queries}")

    print("\nAverage latency:")
    print(
        f"  Fixed Star              : "
        f"{overall_star:.6f} ms"
    )
    print(
        f"  Fixed Snowflake         : "
        f"{overall_snowflake:.6f} ms"
    )
    print(
        f"  Fixed Fact Constellation: "
        f"{overall_constellation:.6f} ms"
    )
    print(
        f"  Best Fixed              : "
        f"{overall_best_fixed:.6f} ms"
    )
    print(
        f"  Adaptive ML             : "
        f"{overall_adaptive:.6f} ms"
    )
    print(
        f"  Oracle                  : "
        f"{overall_oracle:.6f} ms"
    )

    print("\nOverall comparison:")

    print(
        f"  Adaptive vs Star        : "
        f"{(overall_star - overall_adaptive) / overall_star * 100:.3f}%"
    )

    print(
        f"  Adaptive vs Best Fixed  : "
        f"{(overall_best_fixed - overall_adaptive) / overall_best_fixed * 100:.3f}%"
    )

    print(
        f"  Oracle vs Star          : "
        f"{(overall_star - overall_oracle) / overall_star * 100:.3f}%"
    )

    print(
        f"  Adaptive gap to Oracle  : "
        f"{(overall_adaptive - overall_oracle) / overall_oracle * 100:.3f}%"
    )

    print("\nOverall regret:")
    print(
        f"  Mean regret             : "
        f"{overall_regret:.6f} ms"
    )

    print(
        f"  Median regret           : "
        f"{overall_median_regret:.6f} ms"
    )

    print(
        f"  Mean regret percent     : "
        f"{results_df['regret_ms'].div(results_df['oracle_latency_ms']).mean() * 100:.3f}%"
    )

    print(
        f"  Oracle match rate       : "
        f"{overall_match:.2f}%"
    )

    print("\nFiles written:")
    print(f"  {RESULT_FILE}")
    print(f"  {SUMMARY_FILE}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
