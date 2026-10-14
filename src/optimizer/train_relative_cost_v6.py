import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder


RICH_FILE = "data/processed/training_data_v3_rich.csv"
PLAN_FILE = "data/processed/query_plan_features_v5.csv"

RESULT_FILE = "data/processed/v6_relative_cost_results.csv"
SUMMARY_FILE = "data/processed/v6_relative_cost_summary.csv"


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

SCHEMAS = [
    "star",
    "snowflake",
    "fact_constellation",
]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_features(train_df, test_df):

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

    train_semantic = (
        train_df[SEMANTIC_FEATURES]
        .astype(float)
        .values
    )

    test_semantic = (
        test_df[SEMANTIC_FEATURES]
        .astype(float)
        .values
    )

    # ---------------------------------------------------------
    # Identify schema-specific plan features.
    # ---------------------------------------------------------

    plan_columns = [
        c for c in train_df.columns
        if c.startswith("star_")
        or c.startswith("snowflake_")
        or c.startswith("fact_constellation_")
    ]

    plan_columns = [
        c for c in plan_columns
        if pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # ---------------------------------------------------------
    # Base plan features.
    # ---------------------------------------------------------

    train_plan = train_df[
        plan_columns
    ].astype(float).values

    test_plan = test_df[
        plan_columns
    ].astype(float).values

    # ---------------------------------------------------------
    # Cross-schema differential features.
    # ---------------------------------------------------------

    differential_names = []
    train_differentials = []
    test_differentials = []

    comparable_bases = [
        "plan_scan_count",
        "plan_join_count",
        "plan_hash_join_count",
        "plan_nested_loop_join_count",
        "plan_merge_join_count",
        "plan_filter_count",
        "plan_aggregate_count",
        "plan_group_by_count",
        "plan_order_by_count",
        "plan_limit_count",
        "plan_projection_count",
        "plan_left_join_count",
        "plan_inner_join_count",
        "estimated_cardinality_max",
        "estimated_cardinality_min",
        "estimated_cardinality_mean",
        "cardinality_node_count",
        "plan_operator_count",
        "plan_depth",
        "plan_text_length",
    ]

    schema_prefixes = {
        "star": "star_",
        "snowflake": "snowflake_",
        "fact_constellation": "fact_constellation_",
    }

    for base in comparable_bases:

        star_col = schema_prefixes["star"] + base
        snowflake_col = schema_prefixes["snowflake"] + base
        fc_col = schema_prefixes["fact_constellation"] + base

        required = [
            star_col,
            snowflake_col,
            fc_col,
        ]

        if not all(
            c in train_df.columns
            for c in required
        ):
            continue

        # Snowflake relative to Star
        name = f"snowflake_minus_star_{base}"
        differential_names.append(name)

        train_differentials.append(
            (
                train_df[snowflake_col].values
                - train_df[star_col].values
            ).reshape(-1, 1)
        )

        test_differentials.append(
            (
                test_df[snowflake_col].values
                - test_df[star_col].values
            ).reshape(-1, 1)
        )

        # Fact Constellation relative to Star
        name = f"fc_minus_star_{base}"
        differential_names.append(name)

        train_differentials.append(
            (
                train_df[fc_col].values
                - train_df[star_col].values
            ).reshape(-1, 1)
        )

        test_differentials.append(
            (
                test_df[fc_col].values
                - test_df[star_col].values
            ).reshape(-1, 1)
        )

        # Snowflake relative to FC
        name = f"snowflake_minus_fc_{base}"
        differential_names.append(name)

        train_differentials.append(
            (
                train_df[snowflake_col].values
                - train_df[fc_col].values
            ).reshape(-1, 1)
        )

        test_differentials.append(
            (
                test_df[snowflake_col].values
                - test_df[fc_col].values
            ).reshape(-1, 1)
        )

    if train_differentials:
        train_diff = np.hstack(
            train_differentials
        )

        test_diff = np.hstack(
            test_differentials
        )
    else:
        train_diff = np.empty(
            (len(train_df), 0)
        )

        test_diff = np.empty(
            (len(test_df), 0)
        )

    X_train = np.hstack([
        train_semantic,
        train_cat,
        train_plan,
        train_diff,
    ])

    X_test = np.hstack([
        test_semantic,
        test_cat,
        test_plan,
        test_diff,
    ])

    feature_names = (
        SEMANTIC_FEATURES
        + list(encoder.get_feature_names_out(
            CATEGORICAL_FEATURES
        ))
        + plan_columns
        + differential_names
    )

    return (
        X_train,
        X_test,
        feature_names
    )


def main():

    print("=" * 80)
    print("V6 RELATIVE-COST + CROSS-SCHEMA PLAN MODEL")
    print("=" * 80)

    rich = pd.read_csv(RICH_FILE)
    plan = pd.read_csv(PLAN_FILE)

    df = rich.merge(
        plan,
        on=["query_id", "workload"],
        how="inner",
        suffixes=("", "_plan")
    )

    # Remove duplicate merge columns.
    duplicated = [
        c for c in df.columns
        if c.endswith("_plan")
    ]

    for col in duplicated:
        base = col[:-5]

        if base in df.columns:
            df.drop(
                columns=[col],
                inplace=True
            )

    # Remove query-length feature intentionally.
    if "query_length" in df.columns:
        df.drop(
            columns=["query_length"],
            inplace=True
        )

    # Remove target-derived fields.
    for col in [
        "best_schema",
        "best_time_ms",
        "oracle_improvement_percent",
    ]:
        if col in df.columns:
            df.drop(
                columns=[col],
                inplace=True
            )

    print(f"\nDataset rows : {len(df)}")

    print("\nWorkload distribution:")
    print(
        df["workload"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Create relative-cost targets.
    #
    # Star is the reference schema.
    #
    # Values > 1 mean the candidate is slower than Star.
    # Values < 1 mean the candidate is faster than Star.
    # ---------------------------------------------------------

    df["snowflake_relative_cost"] = (
        df["snowflake_time_ms"]
        / df["star_time_ms"]
    )

    df["fc_relative_cost"] = (
        df["constellation_time_ms"]
        / df["star_time_ms"]
    )

    target_columns = [
        "snowflake_relative_cost",
        "fc_relative_cost",
    ]

    # ---------------------------------------------------------
    # Leave-one-workload-out evaluation.
    # ---------------------------------------------------------

    workloads = sorted(
        df["workload"].unique()
    )

    all_results = []
    summary_rows = []

    for held_out in workloads:

        print("\n" + "-" * 80)
        print(
            f"HOLDING OUT WORKLOAD: {held_out}"
        )
        print("-" * 80)

        train_df = df[
            df["workload"] != held_out
        ].copy()

        test_df = df[
            df["workload"] == held_out
        ].copy()

        X_train, X_test, feature_names = (
            build_features(
                train_df,
                test_df
            )
        )

        print(
            f"Training rows : {len(train_df)}"
        )

        print(
            f"Testing rows  : {len(test_df)}"
        )

        print(
            f"Final features: {X_train.shape[1]}"
        )

        predictions = {}

        # -----------------------------------------------------
        # Train relative-cost models.
        # -----------------------------------------------------

        for target in target_columns:

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

            predictions[target] = (
                model.predict(X_test)
            )

            mae = mean_absolute_error(
                test_df[target].values,
                predictions[target]
            )

            error_rmse = rmse(
                test_df[target].values,
                predictions[target]
            )

            print(
                f"{target:28s} "
                f"MAE={mae:.6f} | "
                f"RMSE={error_rmse:.6f}"
            )

        # -----------------------------------------------------
        # Reconstruct predicted absolute costs.
        # -----------------------------------------------------

        actual_star = (
            test_df["star_time_ms"]
            .values
        )

        predicted_star = actual_star

        predicted_snowflake = (
            predicted_star
            * predictions[
                "snowflake_relative_cost"
            ]
        )

        predicted_fc = (
            predicted_star
            * predictions[
                "fc_relative_cost"
            ]
        )

        predicted_matrix = np.column_stack([
            predicted_star,
            predicted_snowflake,
            predicted_fc,
        ])

        predicted_indices = np.argmin(
            predicted_matrix,
            axis=1
        )

        predicted_schema = [
            SCHEMAS[i]
            for i in predicted_indices
        ]

        # -----------------------------------------------------
        # Actual costs.
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

        adaptive_latency = actual_matrix[
            np.arange(len(test_df)),
            predicted_indices
        ]

        oracle_latency = np.min(
            actual_matrix,
            axis=1
        )

        star_latency = actual_matrix[:, 0]
        snowflake_latency = actual_matrix[:, 1]
        fc_latency = actual_matrix[:, 2]

        # -----------------------------------------------------
        # Store query-level results.
        # -----------------------------------------------------

        for i, (_, row) in enumerate(
            test_df.iterrows()
        ):

            all_results.append({
                "query_id": row["query_id"],
                "workload": row["workload"],
                "held_out_workload": held_out,

                "predicted_schema":
                    predicted_schema[i],

                "oracle_schema":
                    oracle_schema[i],

                "predicted_snowflake_relative_cost":
                    predictions[
                        "snowflake_relative_cost"
                    ][i],

                "predicted_fc_relative_cost":
                    predictions[
                        "fc_relative_cost"
                    ][i],

                "actual_snowflake_relative_cost":
                    row[
                        "snowflake_relative_cost"
                    ],

                "actual_fc_relative_cost":
                    row[
                        "fc_relative_cost"
                    ],

                "actual_star_ms":
                    star_latency[i],

                "actual_snowflake_ms":
                    snowflake_latency[i],

                "actual_constellation_ms":
                    fc_latency[i],

                "adaptive_latency_ms":
                    adaptive_latency[i],

                "oracle_latency_ms":
                    oracle_latency[i],

                "regret_ms":
                    adaptive_latency[i]
                    - oracle_latency[i],

                "oracle_match":
                    int(
                        predicted_schema[i]
                        == oracle_schema[i]
                    ),
            })

        # -----------------------------------------------------
        # Summary metrics.
        # -----------------------------------------------------

        adaptive_mean = (
            np.mean(adaptive_latency)
        )

        oracle_mean = (
            np.mean(oracle_latency)
        )

        star_mean = np.mean(
            star_latency
        )

        snowflake_mean = np.mean(
            snowflake_latency
        )

        fc_mean = np.mean(
            fc_latency
        )

        best_fixed = min(
            star_mean,
            snowflake_mean,
            fc_mean
        )

        regret = (
            adaptive_latency
            - oracle_latency
        )

        match_rate = (
            np.mean(
                np.array(predicted_schema)
                == np.array(oracle_schema)
            )
            * 100
        )

        adaptive_vs_star = (
            (star_mean - adaptive_mean)
            / star_mean
            * 100
        )

        adaptive_vs_best = (
            (best_fixed - adaptive_mean)
            / best_fixed
            * 100
        )

        oracle_vs_star = (
            (star_mean - oracle_mean)
            / star_mean
            * 100
        )

        gap_to_oracle = (
            (adaptive_mean - oracle_mean)
            / oracle_mean
            * 100
        )

        summary_rows.append({
            "held_out_workload":
                held_out,

            "test_queries":
                len(test_df),

            "fixed_star_ms":
                star_mean,

            "fixed_snowflake_ms":
                snowflake_mean,

            "fixed_constellation_ms":
                fc_mean,

            "best_fixed_ms":
                best_fixed,

            "adaptive_ml_ms":
                adaptive_mean,

            "oracle_ms":
                oracle_mean,

            "adaptive_vs_star_percent":
                adaptive_vs_star,

            "adaptive_vs_best_fixed_percent":
                adaptive_vs_best,

            "oracle_vs_star_percent":
                oracle_vs_star,

            "adaptive_gap_to_oracle_percent":
                gap_to_oracle,

            "mean_regret_ms":
                np.mean(regret),

            "median_regret_ms":
                np.median(regret),

            "mean_regret_percent":
                np.mean(
                    regret
                    / oracle_latency
                    * 100
                ),

            "oracle_match_rate_percent":
                match_rate,
        })

        print("\nLatency:")
        print(
            f"  Fixed Star              : "
            f"{star_mean:.6f} ms"
        )

        print(
            f"  Fixed Snowflake         : "
            f"{snowflake_mean:.6f} ms"
        )

        print(
            f"  Fixed Fact Constellation: "
            f"{fc_mean:.6f} ms"
        )

        print(
            f"  Best Fixed              : "
            f"{best_fixed:.6f} ms"
        )

        print(
            f"  Adaptive ML             : "
            f"{adaptive_mean:.6f} ms"
        )

        print(
            f"  Oracle                  : "
            f"{oracle_mean:.6f} ms"
        )

        print("\nPerformance:")

        print(
            f"  Adaptive vs Star        : "
            f"{adaptive_vs_star:.3f}%"
        )

        print(
            f"  Adaptive vs Best Fixed  : "
            f"{adaptive_vs_best:.3f}%"
        )

        print(
            f"  Oracle vs Star          : "
            f"{oracle_vs_star:.3f}%"
        )

        print(
            f"  Adaptive gap to Oracle  : "
            f"{gap_to_oracle:.3f}%"
        )

        print("\nRegret:")

        print(
            f"  Mean regret             : "
            f"{np.mean(regret):.6f} ms"
        )

        print(
            f"  Median regret           : "
            f"{np.median(regret):.6f} ms"
        )

        print(
            f"  Mean regret percent     : "
            f"{np.mean(regret / oracle_latency * 100):.3f}%"
        )

        print(
            f"  Oracle match rate       : "
            f"{match_rate:.2f}%"
        )

        print("\nPredicted schema distribution:")
        print(
            pd.Series(predicted_schema)
            .value_counts()
            .reindex(
                SCHEMAS,
                fill_value=0
            )
        )

        print("\nOracle schema distribution:")
        print(
            pd.Series(oracle_schema)
            .value_counts()
            .reindex(
                SCHEMAS,
                fill_value=0
            )
        )

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Overall results.
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("V6 OVERALL LEAVE-ONE-WORKLOAD-OUT RESULTS")
    print("=" * 80)

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

    overall_adaptive = (
        results_df[
            "adaptive_latency_ms"
        ].mean()
    )

    overall_oracle = (
        results_df[
            "oracle_latency_ms"
        ].mean()
    )

    overall_star = (
        results_df[
            "actual_star_ms"
        ].mean()
    )

    overall_snowflake = (
        results_df[
            "actual_snowflake_ms"
        ].mean()
    )

    overall_fc = (
        results_df[
            "actual_constellation_ms"
        ].mean()
    )

    overall_best_fixed = min(
        overall_star,
        overall_snowflake,
        overall_fc
    )

    overall_regret = (
        results_df["regret_ms"]
        .mean()
    )

    overall_median_regret = (
        results_df["regret_ms"]
        .median()
    )

    overall_match = (
        results_df["oracle_match"]
        .mean()
        * 100
    )

    print("\nTotal queries:")
    print(
        f"  {len(results_df)}"
    )

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
        f"{overall_fc:.6f} ms"
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
    print(
        f"  {RESULT_FILE}"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print("\nV6 evaluation complete.")


if __name__ == "__main__":
    main()
