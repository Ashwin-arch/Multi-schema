# Adaptive Multi-Schema Data Warehouse: Scalability Experiment

## Research Objective

This project implements a reproducible scalability experiment for the research prototype:
"Adaptive Multi-Schema Data Warehouse: An ML-Based Query-Aware Schema Selection Framework"

The research investigates whether query structure and pre-execution query-plan characteristics can be used to select among three physically materialized warehouse schemas (Star, Snowflake, Fact Constellation) to achieve lower execution cost than a fixed-schema strategy.

**Crucially**: The system does NOT claim one schema is globally superior. The research question is:
"Can query structure and pre-execution query-plan characteristics be used to select among multiple warehouse schemas with lower execution cost than a fixed-schema strategy?"

## Environment Requirements

- Python 3.10+
- Google Colab-compatible (Linux environment)
- Required packages: duckdb, pandas, numpy, scikit-learn, matplotlib, tabulate, seaborn

## Installation

```bash
pip install -r requirements.txt
```

## Experiment Versions

This project contains multiple experimental versions, each in isolated directories:

### V7: Baseline Scalability Experiment ("Scale Sensitivity of Adaptive Multi-Schema Selection")
- **Workload**: 600 queries (5 classes, V3 generator)
- **Scales**: 100K, 500K, 1M, 5M, 10M
- **Dimensions**: 1K customers, 60 products, 365 dates
- **Key Finding**: Oracle advantage 1.8–4.5%, adaptive never beats fixed Star

### V8: Performance-Aware Experiment ("Performance-Aware Adaptive Multi-Schema Selection")
- **Workload**: 223 schema-sensitive queries (6 classes, V8b generator)
- **Scales**: 100K, 500K, 1M, 5M, 10M (plus 1M with 10K cust/1K prod)
- **Dimensions**: configurable (default 1K/60; tested 10K/1K at 1M)
- **All schemas include fact_returns**; all queries run on all schemas
- **Key Finding**: Oracle advantage 2.4–5.3%, adaptive never beats fixed Star

---

## V7: Baseline Scalability Experiment

### Dataset Generation

Generate deterministic datasets at specified scales:

```bash
python src/data/generate_data_v7.py --rows 100000
python src/data/generate_data_v7.py --rows 500000
python src/data/generate_data_v7.py --rows 1000000
python src/data/generate_data_v7.py --rows 5000000
python src/data/generate_data_v7.py --rows 10000000  # if resources permit
```

Uses a fixed random seed (SEED=42) for full reproducibility.

### Schema Construction

Build all three schemas for a given scale:

```bash
python src/schemas/build_schemas_v7.py --rows 100000
```

Schemas:
- **Star**: fact_sales, dim_customer, dim_product, dim_date
- **Snowflake**: fact_sales, dim_customer, dim_region, dim_segment, dim_product, dim_category, dim_date
- **Fact Constellation**: fact_sales, fact_returns, dim_customer, dim_product, dim_date

### Workload

- 600 analytical queries covering: aggregation, filtered_aggregation, join_heavy, mixed, selective
- Validate workload compatibility:

```bash
python src/workload/validate_workload_v7.py --rows 100000
```

### Feature Extraction

Extract pre-execution semantic and cross-schema query plan features:

```bash
python src/workload/feature_extractor_v7.py --rows 100000
```

113 numeric + 2 categorical features per query (schema-independent + cross-schema plan differences).

### Benchmarking

Run multi-schema benchmarking with warmup and repetitions:

```bash
python src/benchmark/run_benchmark_v7.py --rows 100000 --repeats 5 --warmup 1
```

Records: query_id, dataset_size, workload, schema, repetition, latency_ms, success/failure.

### Adaptive Model Evaluation

Train and evaluate adaptive controller (Leave-One-Workload-Out):

```bash
python src/optimizer/train_adaptive_model_v7.py --rows 100000
```

Predicts relative costs (snowflake/star, fact_constellation/star). Star cost fixed at 1.0.
No actual latency used as input feature (leakage-safe).

### Scalability Analysis

Generate publication-quality figures and tables:

```bash
python src/analysis/scalability_analysis_v7.py
```

Produces 7 figures + final summary table.

### Resource Verification

Check RAM/disk feasibility before large scales:

```bash
python src/utils/system_check.py
```

Estimates calibrated from measured footprints on reference machine.

### V7 Key Outputs

- `data/raw/` - Generated CSV data files
- `data/processed/` - DuckDB databases, features, workload files
- `results/scale_raw_results.csv` - Per-execution timing records
- `results/scale_query_summary.csv` - Query-level summaries
- `results/scale_summary.csv` - Dataset-level aggregates
- `results/adaptive_model_summary.csv` - Adaptive controller performance
- `results/adaptive_model_query_results.csv` - Per-query predictions
- `results/resource_usage.csv` - Measured resource consumption (all stages)
- `results/figures/` - Publication-quality plots (7 figures)
- `results/final_research_summary_table.{csv,md}` - Final comprehensive table

### V7 Key Findings

| Scale | Star Mean (ms) | Snowflake Mean (ms) | FC Mean (ms) | Oracle vs Star | Adaptive vs Star | Oracle Match Rate |
|-------|---------------|---------------------|--------------|----------------|------------------|-------------------|
| 100K  | 5.86          | 7.00                | 5.92         | 3.58%          | -0.20%           | 46.50%            |
| 500K  | 7.77          | 8.98                | 7.81         | 4.49%          | +0.07%           | 48.50%            |
| 1M    | 10.92         | 12.04               | 11.08        | 3.07%          | -0.28%           | 47.67%            |
| 5M    | 29.55         | 31.12               | 29.66        | 1.91%          | -0.06%           | 46.00%            |
| 10M   | 51.89         | 53.45               | 51.93        | 1.83%          | -0.06%           | 43.83%            |

**Answer to the research question: NO.** Increasing warehouse scale does NOT make adaptive multi-schema selection more useful.

- **Oracle advantage shrinks with scale**: 3.58% (100K) → 1.83% (10M). Schema-dependent performance differences *decrease* as data grows.
- **Adaptive controller does not beat fixed Star**: Across all scales, adaptive mean latency is statistically indistinguishable from (or slightly worse than) the fixed Star schema. Confidence intervals consistently cross zero.
- **Oracle match rate plateaus**: ~44-49% across scales — the model does not improve its selection accuracy at larger scales.
- **Mean regret grows**: From 0.22 ms (100K) to 0.98 ms (10M) — absolute cost of mispredictions increases.

---

## V8: Performance-Aware Experiment

### Dataset Generation (Configurable Dimensions)

```bash
# Default (1K customers, 60 products)
python src/data/generate_data_v8.py --rows 1000000

# Larger dimensions (tested at 1M scale)
python src/data/generate_data_v8.py --rows 1000000 --num-customers 10000 --num-products 1000
```

### Schema Construction (All Schemas Include fact_returns)

```bash
python src/schemas/build_schemas_v7.py --rows 1000000
```

All three schemas now include `fact_returns` for cross-schema query compatibility.

### Workload Generation (Schema-Sensitive, 223 Queries)

```bash
python src/workload/generate_workload_v8b.py
```

Generates 223 queries across 6 schema-targeted classes:
- star_oriented (40): wide scans, minimal joins
- snowflake_oriented (70): normalized dimension paths (region→segment→customer)
- fact_constellation_oriented (33): multi-fact sales+returns analysis
- selective (25): point lookups
- multi_dimensional (20): 3-4 GROUP BY columns
- mixed_complex (35): realistic analytical patterns

All queries use canonical denormalized column names; adapter translates to each schema's native structure.

### Validation (All Queries on All Schemas)

```bash
python src/workload/validate_workload_v8.py --workload data/processed/workload_v8b.json --rows 1000000
```

### Feature Extraction

```bash
python src/workload/feature_extractor_v8b.py --rows 1000000 --workload data/processed/workload_v8b.json
```

### Benchmarking (Balanced: All Queries on All Schemas)

```bash
python src/benchmark/run_benchmark_v8.py --rows 1000000 --workload data/processed/workload_v8b.json --repeats 3 --warmup 1
```

Runs each query on all three schemas (223 queries × 3 schemas × 3 reps = 2,007 executions).

### Adaptive Model Evaluation

```bash
python src/optimizer/train_adaptive_model_v8b.py --rows 1000000
```

Random Forest with LOWO CV, 113 pre-execution features.

### V8 Key Outputs

- `results/v8/scale_raw_results.csv` — 33,405 execution records
- `results/v8/scale_query_summary.csv` — Per-query summaries
- `results/v8/scale_summary.csv` — Scale-level aggregates
- `results/v8/adaptive_model_*.csv` — Adaptive model results
- `results/v8/resource_usage.csv` — Measured resources

### V8 Key Findings

| Scale (Dim) | Star Mean | Snow Mean | FC Mean | Best Fixed | Oracle | Oracle vs Star | Adaptive vs Star |
|-------------|-----------|-----------|---------|------------|--------|----------------|------------------|
| 100K        | 6.47      | 7.30      | 6.51    | Star       | 6.13   | 5.29%          | -0.20%           |
| 500K        | 10.40     | 11.46     | 10.55   | Star       | 9.87   | 5.12%          | +0.07%           |
| 1M (1K/60)  | 16.18     | 17.36     | 16.14   | FC         | 15.33  | 5.29%          | -0.34%           |
| 1M (10K/1K) | 17.71     | 18.82     | 17.86   | Star       | 17.19  | 2.98%          | -0.34%           |
| 5M          | 60.14     | 61.26     | 60.55   | Star       | 58.70  | 2.40%          | —                |
| 10M         | 125.53    | 126.85    | 125.67  | Star       | 120.69 | 3.86%          | —                |

**Adaptive Model (1M scale):**
- Adaptive vs Star: -0.34% (95% CI: [-0.15, 0.02] ms)
- Oracle vs Star: +2.98%
- Oracle Match Rate: 33.6%
- Mean Regret: 0.59 ms

**Answer: NO.** Adaptive schema selection provides no measurable runtime improvement.

---

## Why Schema Selection Fails (Root Cause Analysis)

1. **Physical Plans Nearly Identical**: For non-multi-fact queries, Star and Fact Constellation produce *identical* hash-join plans. Snowflake adds 4 extra joins (customer→region, customer→segment, product→category) with zero benefit because compatibility views force the full denormalized path.

2. **Dimensions Too Small**: Even with 10K customers/1K products, all dimension tables fit in DuckDB's cache. Hash joins on 10K-row tables are near-instant. No nested-loop joins are triggered.

3. **No Physical Design Differentiation**: No indexes, partitioning, clustering, or sorting. All schemas use identical DuckDB defaults (columnar, compressed).

3. **Workload Doesn't Stress Differences**: No high-selectivity point queries (would favor indexes/nested-loop), no deep hierarchy drill-down, multi-fact queries rare (33/223).

**Increasing dimension cardinality (1K→10K, 60→1K) REDUCED Oracle advantage** (5.3% → 3.0%) because larger dimension tables make all schemas slower proportionally without changing relative costs.

---

## Reproducibility

All random operations use deterministic seeds (SEED=42). The experiment follows strict separation between pre-execution features and post-execution latency targets to prevent information leakage. Generation, build, and benchmark stages all log measured resource usage to `results/resource_usage.csv`.

## Limitations

- **Single DBMS (DuckDB)**: Findings are DuckDB-specific. Other engines with different join optimizers / index support may show different schema-sensitivity patterns.
- **Synthetic Workload**: TPC-DS-inspired but limited to 5 templates; real workloads may have different join/filter distributions.
- **No Index Tuning**: All schemas run unindexed; production warehouses typically index dimension keys.
- **Hardware-Dependent**: Results measured on a specific machine (7.5 GB RAM, NVMe). Absolute latencies will vary; relative trends are the scientific contribution.
- **Model Simplicity**: Random Forest with 113 features; more sophisticated architectures (gradient boosting, neural) were not explored.

## Research Question Answered

"Does increasing warehouse scale make adaptive multi-schema selection more useful?"

**No.** The empirical evidence across five dataset sizes (100K → 10M rows) and two workload designs (V7: 600 queries; V8: 223 schema-sensitive queries) shows that:

1. The performance gap between schemas *narrows* with scale, not widens.
2. The adaptive controller fails to consistently outperform a simple fixed-Star strategy at any scale.
3. The oracle upper bound (always choosing the best schema per query) shrinks from 3.6% to 1.8% advantage over Star (V7) and remains at 2.4–5.3% (V8).
4. Increasing dimension cardinality does not create meaningful schema-dependent differences under DuckDB's hash-join architecture.

For this workload and DBMS, the Star schema is a robust default. Adaptive schema selection does not provide measurable benefit at any tested scale.

## Recommendations for Future Work

Given the negative result, we recommend pivoting to research directions with larger optimization opportunity:

1. **Adaptive Physical Design** — index selection, partitioning, materialized view selection (optimization opportunity: 10-50%+)
2. **Query Routing** — route to different engines (OLAP vs OLTP) based on query shape
3. **Hybrid Schema Generation** — synthesize per-query schemas on-the-fly
4. **Learned Cardinality Estimation** — fix optimizer misestimates that affect all schemas equally
5. **If Schema Selection Still Targeted**: increase dimension cardinalities 100× (100K+ customers), add physical design differentiation (indexes, partitioning, clustering), create index-friendly workload with high-selectivity point lookups.