import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
ADAPTIVE_SUMMARY_FILE = os.path.join(RESULTS_DIR, "adaptive_model_summary.csv")
QUERY_SUMMARY_FILE = os.path.join(RESULTS_DIR, "scale_query_summary.csv")
ADAPTIVE_QUERY_FILE = os.path.join(RESULTS_DIR, "adaptive_model_query_results.csv")
FINAL_TABLE_CSV = os.path.join(RESULTS_DIR, "final_research_summary_table.csv")
FINAL_TABLE_MD = os.path.join(RESULTS_DIR, "final_research_summary_table.md")

# Publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def format_scale_labels(sizes):
    labels = []
    for s in sizes:
        if s >= 1_000_000:
            labels.append(f"{s // 1_000_000}M")
        elif s >= 1_000:
            labels.append(f"{s // 1_000}K")
        else:
            labels.append(str(s))
    return labels


def generate_all_figures():
    """Generate all 7 publication-quality scalability figures and research tables."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not os.path.exists(ADAPTIVE_SUMMARY_FILE):
        print(f"Error: {ADAPTIVE_SUMMARY_FILE} not found. Run benchmark and adaptive model first.")
        return

    summary_df = pd.read_csv(ADAPTIVE_SUMMARY_FILE).sort_values("dataset_size")
    query_df = pd.read_csv(QUERY_SUMMARY_FILE) if os.path.exists(QUERY_SUMMARY_FILE) else None
    adaptive_query_df = pd.read_csv(ADAPTIVE_QUERY_FILE) if os.path.exists(ADAPTIVE_QUERY_FILE) else None

    sizes = summary_df["dataset_size"].values
    scale_labels = format_scale_labels(sizes)
    x = np.arange(len(sizes))

    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES & SCALABILITY TABLES")
    print("=" * 70)
    print(f"Evaluated scales: {', '.join(scale_labels)}")

    # -------------------------------------------------------------
    # 1. Mean Latency vs Dataset Size (Star, Snowflake, FC, Adaptive, Oracle)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, summary_df["star_mean_ms"], marker="o", linewidth=2, label="Star Schema", color="#1f77b4")
    ax.plot(x, summary_df["snowflake_mean_ms"], marker="s", linewidth=2, label="Snowflake Schema", color="#ff7f0e")
    ax.plot(x, summary_df["fc_mean_ms"], marker="^", linewidth=2, label="Fact Constellation", color="#2ca02c")
    ax.plot(x, summary_df["adaptive_mean_ms"], marker="D", linewidth=2.5, linestyle="--", label="Adaptive (V7 Controller)", color="#9467bd")
    ax.plot(x, summary_df["oracle_mean_ms"], marker="*", linewidth=1.5, linestyle=":", label="Oracle (Theoretical Optimal)", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_xlabel("Warehouse Scale (Sales Rows)")
    ax.set_ylabel("Mean Execution Latency (ms)")
    ax.set_title("Figure 1: Mean Query Latency vs. Warehouse Scale")
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "1_mean_latency_vs_scale.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"  [Saved] {fig1_path}")

    # -------------------------------------------------------------
    # 2. Oracle Improvement vs Dataset Size
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, summary_df["oracle_vs_star_percent"], marker="o", linewidth=2, color="#d62728", label="Oracle vs. Star")
    if "oracle_vs_best_fixed_percent" in summary_df.columns:
        ax.plot(x, summary_df["oracle_vs_best_fixed_percent"], marker="s", linewidth=2, linestyle="--", color="#e377c2", label="Oracle vs. Best Fixed")

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_xlabel("Warehouse Scale (Sales Rows)")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Figure 2: Theoretical Maximum Gain (Oracle Upper Bound) vs. Scale")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "2_oracle_improvement_vs_scale.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"  [Saved] {fig2_path}")

    # -------------------------------------------------------------
    # 3. Adaptive Improvement vs Dataset Size
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, summary_df["adaptive_vs_star_percent"], marker="D", linewidth=2, color="#9467bd", label="Adaptive vs. Star")
    ax.plot(x, summary_df["adaptive_vs_best_fixed_percent"], marker="v", linewidth=2, linestyle="--", color="#17becf", label="Adaptive vs. Best Fixed")

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_xlabel("Warehouse Scale (Sales Rows)")
    ax.set_ylabel("Relative Gain / Loss (%)")
    ax.set_title("Figure 3: Adaptive Controller Performance vs. Baseline Schemas")
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "3_adaptive_improvement_vs_scale.png")
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"  [Saved] {fig3_path}")

    # -------------------------------------------------------------
    # 4. Adaptive Gap to Oracle vs Dataset Size
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, summary_df["adaptive_gap_to_oracle_percent"], marker="P", linewidth=2, color="#8c564b", label="Adaptive Regret Gap (%)")
    if "oracle_match_rate_percent" in summary_df.columns:
        ax2 = ax.twinx()
        ax2.plot(x, summary_df["oracle_match_rate_percent"], marker="X", linewidth=2, linestyle=":", color="#2ca02c", label="Oracle Match Rate (%)")
        ax2.set_ylabel("Match Rate (%)", color="#2ca02c")
        ax2.grid(False)

    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_xlabel("Warehouse Scale (Sales Rows)")
    ax.set_ylabel("Gap to Theoretical Optimal (%)", color="#8c564b")
    ax.set_title("Figure 4: Optimization Gap and Decision Fidelity vs. Scale")
    fig.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "4_adaptive_gap_to_oracle_vs_scale.png")
    fig.savefig(fig4_path)
    plt.close(fig)
    print(f"  [Saved] {fig4_path}")

    # -------------------------------------------------------------
    # 5. Schema-Selection Distribution vs Dataset Size
    # -------------------------------------------------------------
    if adaptive_query_df is not None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        dist_data = []
        for s in sizes:
            sub = adaptive_query_df[adaptive_query_df["dataset_size"] == s]
            counts = sub["predicted_schema"].value_counts(normalize=True).to_dict()
            dist_data.append({
                "star": counts.get("star", 0.0) * 100,
                "snowflake": counts.get("snowflake", 0.0) * 100,
                "fact_constellation": counts.get("fact_constellation", 0.0) * 100
            })
        dist_df = pd.DataFrame(dist_data)

        width = 0.55
        p1 = ax.bar(x, dist_df["star"], width, label="Star", color="#1f77b4")
        p2 = ax.bar(x, dist_df["snowflake"], width, bottom=dist_df["star"], label="Snowflake", color="#ff7f0e")
        p3 = ax.bar(x, dist_df["fact_constellation"], width, bottom=dist_df["star"] + dist_df["snowflake"], label="Fact Constellation", color="#2ca02c")

        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        ax.set_xlabel("Warehouse Scale (Sales Rows)")
        ax.set_ylabel("Controller Selection Share (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Figure 5: Adaptive Schema-Selection Distribution Across Scales")
        ax.legend(frameon=True, loc="upper right")
        fig.tight_layout()
        fig5_path = os.path.join(FIGURES_DIR, "5_schema_selection_distribution_vs_scale.png")
        fig.savefig(fig5_path)
        plt.close(fig)
        print(f"  [Saved] {fig5_path}")

    # -------------------------------------------------------------
    # 6. Latency Variability vs Dataset Size (Standard Deviation)
    # -------------------------------------------------------------
    if query_df is not None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        std_star = []
        std_snow = []
        std_fc = []
        for s in sizes:
            sub = query_df[query_df["dataset_size"] == s]
            std_star.append(sub["star_std_ms"].mean())
            std_snow.append(sub["snowflake_std_ms"].mean())
            std_fc.append(sub["fc_std_ms"].mean())

        ax.plot(x, std_star, marker="o", linewidth=2, label="Star Schema Std", color="#1f77b4")
        ax.plot(x, std_snow, marker="s", linewidth=2, label="Snowflake Schema Std", color="#ff7f0e")
        ax.plot(x, std_fc, marker="^", linewidth=2, label="Fact Constellation Std", color="#2ca02c")

        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        ax.set_xlabel("Warehouse Scale (Sales Rows)")
        ax.set_ylabel("Average Query Latency Std (ms)")
        ax.set_title("Figure 6: Execution Latency Variability Across Scales")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig6_path = os.path.join(FIGURES_DIR, "6_latency_variability_vs_scale.png")
        fig.savefig(fig6_path)
        plt.close(fig)
        print(f"  [Saved] {fig6_path}")

    # -------------------------------------------------------------
    # 7. Per-Workload Latency vs Dataset Size
    # -------------------------------------------------------------
    if query_df is not None:
        workloads = sorted(query_df["workload"].unique())
        fig, ax = plt.subplots(figsize=(8.5, 5))
        markers = ["o", "s", "^", "D", "v"]
        for idx, w in enumerate(workloads):
            w_latencies = []
            for s in sizes:
                sub = query_df[(query_df["dataset_size"] == s) & (query_df["workload"] == w)]
                w_latencies.append(sub["star_mean_ms"].mean())
            ax.plot(x, w_latencies, marker=markers[idx % len(markers)], linewidth=2, label=f"Workload: {w}")

        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        ax.set_xlabel("Warehouse Scale (Sales Rows)")
        ax.set_ylabel("Star Schema Mean Latency (ms)")
        ax.set_title("Figure 7: Analytical Workload Latency Scaling Behavior")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig7_path = os.path.join(FIGURES_DIR, "7_workload_latency_vs_scale.png")
        fig.savefig(fig7_path)
        plt.close(fig)
        print(f"  [Saved] {fig7_path}")

    # -------------------------------------------------------------
    # Comprehensive Final Research Summary Table
    # -------------------------------------------------------------
    table_rows = []
    for _, r in summary_df.iterrows():
        s = int(r["dataset_size"])
        label = format_scale_labels([s])[0]
        table_rows.append({
            "Dataset Size": f"{label} ({s:,})",
            "Queries": int(r["queries"]),
            "Star Mean (ms)": f"{r['star_mean_ms']:.4f}",
            "Snowflake Mean (ms)": f"{r['snowflake_mean_ms']:.4f}",
            "Fact Constellation Mean (ms)": f"{r['fc_mean_ms']:.4f}",
            "Best Fixed Mean (ms)": f"{r['best_fixed_mean_ms']:.4f}",
            "Adaptive Mean (ms)": f"{r['adaptive_mean_ms']:.4f}",
            "Oracle Mean (ms)": f"{r['oracle_mean_ms']:.4f}",
            "Adaptive vs Star %": f"{r['adaptive_vs_star_percent']:+.2f}%",
            "Adaptive vs Best Fixed %": f"{r['adaptive_vs_best_fixed_percent']:+.2f}%",
            "Oracle vs Star %": f"{r['oracle_vs_star_percent']:.2f}%",
            "Adaptive Gap to Oracle %": f"{r['adaptive_gap_to_oracle_percent']:.2f}%",
            "Oracle Match Rate %": f"{r['oracle_match_rate_percent']:.2f}%",
        })

    final_table_df = pd.DataFrame(table_rows)
    final_table_df.to_csv(FINAL_TABLE_CSV, index=False)

    md_table = final_table_df.to_markdown(index=False)
    with open(FINAL_TABLE_MD, "w") as f:
        f.write("# Adaptive Multi-Schema Scalability Experiment: Final Research Summary Table\n\n")
        f.write(md_table)
        f.write("\n")

    print("\n" + "=" * 90)
    print("FINAL RESEARCH SUMMARY TABLE")
    print("=" * 90)
    print(final_table_df.to_string(index=False))
    print("=" * 90)
    print(f"Summary table saved to:\n  {FINAL_TABLE_CSV}\n  {FINAL_TABLE_MD}")


def main():
    generate_all_figures()


if __name__ == "__main__":
    main()
