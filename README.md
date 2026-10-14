# Adaptive Multi-Schema Data Warehouse: An ML-Based Query-Aware Schema Selection Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DuckDB](https://img.shields.io/badge/database-DuckDB-orange.svg)](https://duckdb.org/)

A reproducible conference-level research prototype and empirical benchmark evaluating whether machine learning and pre-execution query-plan features can dynamically route incoming analytical queries to the optimal physically materialized warehouse schema (Star, Snowflake, or Fact Constellation).

---

## 📌 Table of Contents

- [Research Overview](#-research-overview)
- [Warehouse Architectures](#-warehouse-architectures)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [How to Run the Project](#-how-to-run-the-project)
  - [Quick Start (End-to-End Walkthrough)](#quick-start-end-to-end-walkthrough)
  - [Step-by-Step Execution Pipeline (V8)](#step-by-step-execution-pipeline-v8)
  - [Running the Baseline Scalability Experiment (V7)](#running-the-baseline-scalability-experiment-v7)
- [Experimental Results](#-experimental-results)
  - [V8 Performance-Aware Benchmark](#v8-performance-aware-benchmark)
  - [V7 Baseline Scalability Benchmark](#v7-baseline-scalability-benchmark)
- [Scientific Findings & Root Cause Analysis](#-scientific-findings--root-cause-analysis)
- [Future Research Directions](#-future-research-directions)
- [License](#-license)

---

## 🔬 Research Overview

In modern data warehousing, data modeling traditionally commits to a single schema paradigm—such as a denormalized **Star Schema**, a normalized **Snowflake Schema**, or a multi-fact **Fact Constellation**.

This research investigates the following question:

> **"Can query structure and pre-execution query-plan characteristics be used to select among multiple physically materialized warehouse schemas with lower execution cost than a fixed-schema strategy?"**

To answer this question, the framework:
1. Materializes three distinct physical schemas simultaneously on DuckDB.
2. Formulates synthetic analytical workloads ranging from aggregation-heavy to normalized dimension traversals and multi-fact analyses.
3. Extracts 113 pre-execution, leakage-safe features (syntactic query properties and cross-schema query plan estimates) without measuring actual query runtimes.
4. Trains an adaptive Random Forest controller using **Leave-One-Workload-Out (LOWO)** cross-validation to predict relative execution costs across schemas.
5. Compares the adaptive selector against the theoretical **Oracle** (best possible schema choice per query) and fixed single-schema baselines across data scales: **100K, 500K, 1M, 5M, and 10M rows**.

---

## 🏛️ Warehouse Architectures

All three schemas are physically materialized as distinct DuckDB databases:

| Schema Model | Fact Tables | Dimension Tables | Characteristics |
| :--- | :--- | :--- | :--- |
| **Star Schema** | `fact_sales` | `dim_customer`, `dim_product`, `dim_date` | Fully denormalized dimensions; minimizes join depth. |
| **Snowflake Schema** | `fact_sales` | `dim_customer`, `dim_region`, `dim_segment`, `dim_product`, `dim_category`, `dim_date` | Normalized dimension hierarchies; eliminates redundancy, increases join depth. |
| **Fact Constellation** | `fact_sales`, `fact_returns` | `dim_customer`, `dim_product`, `dim_date` | Multi-fact tables sharing conformed dimensions for cross-process analytics. |

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                   # Dimension tables & base sales CSVs (deterministic generation)
│   └── processed/             # Workload definitions (JSON), feature matrices, training sets
├── src/
│   ├── analysis/              # Publication figures and scalability analysis
│   │   └── scalability_analysis_v7.py
│   ├── benchmark/             # Multi-schema execution runners & repeatability test suites
│   │   ├── run_benchmark_v7.py
│   │   ├── run_benchmark_v8.py
│   │   └── run_v6_repeatability.py
│   ├── data/                  # Deterministic synthetic data generators (fixed SEED=42)
│   │   ├── generate_data_v7.py
│   │   └── generate_data_v8.py
│   ├── optimizer/             # Adaptive ML controllers, feature preparation, and LOWO CV
│   │   ├── train_adaptive_model_v7.py
│   │   ├── train_adaptive_model_v8b.py
│   │   └── adaptive_selector.py
│   ├── schemas/               # DuckDB physical DDL and ingestion scripts
│   │   ├── build_schemas_v7.py
│   │   ├── star_schema.py
│   │   ├── snowflake_schema.py
│   │   └── fact_constellation.py
│   ├── utils/                 # Resource tracking and system memory/disk guards
│   │   ├── resource_logging.py
│   │   └── system_check.py
│   └── workload/              # Workload generators, query validators, and feature extractors
│       ├── generate_workload_v8b.py
│       ├── validate_workload_v8.py
│       ├── feature_extractor_v8b.py
│       └── query_plan_extractor_v5.py
├── results/
│   ├── figures/               # Generated evaluation plots (PDF/PNG)
│   ├── v8/                    # Scale summaries, model summaries, and feature importances (V8)
│   ├── scale_summary.csv      # Aggregate scale latency tables
│   └── final_research_summary_table.md
├── DIAGNOSTIC_REPORT_V7.md    # Detailed audit and diagnostic report for V7
├── V8_EXPERIMENT_REPORT.md    # Comprehensive analysis and metrics for V8
├── README_COLAB.md            # Google Colab execution notes
├── requirements.txt           # Python package dependencies
└── LICENSE                    # MIT License
```

---

## ⚙️ Prerequisites & Installation

### 1. Requirements
- Python 3.10 or higher
- Recommended RAM: ≥ 8 GB (for scales up to 1M–5M)
- OS: Linux (Ubuntu/Debian recommended), macOS, or Windows WSL2

### 2. Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Ashwin-arch/Multi-schema.git
cd Multi-schema

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### Quick Start (End-to-End Walkthrough)

To execute a complete pipeline run at the **100K scale** (~1–2 minutes):

```bash
# 1. Generate synthetic warehouse datasets
python src/data/generate_data_v8.py --rows 100000

# 2. Construct the Star, Snowflake, and Fact Constellation DuckDB databases
python src/schemas/build_schemas_v7.py --rows 100000

# 3. Generate and validate the schema-sensitive workload (223 queries)
python src/workload/generate_workload_v8b.py
python src/workload/validate_workload_v8.py --workload data/processed/workload_v8b.json --rows 100000

# 4. Extract 113 pre-execution plan and structural features
python src/workload/feature_extractor_v8b.py --rows 100000 --workload data/processed/workload_v8b.json

# 5. Execute benchmark across all three schemas (with warmup)
python src/benchmark/run_benchmark_v8.py --rows 100000 --workload data/processed/workload_v8b.json --repeats 3 --warmup 1

# 6. Train adaptive ML controller using Leave-One-Workload-Out CV
python src/optimizer/train_adaptive_model_v8b.py --rows 100000
```

---

### Step-by-Step Execution Pipeline (V8)

The V8 experiment is the primary, performance-aware benchmark using 223 schema-sensitive queries and configurable dimension cardinalities.

#### Step 1: Pre-Execution Resource Check
Verify that available system RAM and disk capacity are sufficient for the desired dataset scale:
```bash
python src/utils/system_check.py
```

#### Step 2: Deterministic Data Generation
Generate raw CSV datasets with customer, product, date, and sales records.
```bash
# Standard dimensions (1,000 customers, 60 products)
python src/data/generate_data_v8.py --rows 1000000

# Or higher dimension cardinality (10,000 customers, 1,000 products)
python src/data/generate_data_v8.py --rows 1000000 --num-customers 10000 --num-products 1000
```
*Supported scales: `--rows 100000`, `500000`, `1000000`, `5000000`, `10000000`.*

#### Step 3: Physical Schema Materialization
Instantiate the physical Star, Snowflake, and Fact Constellation DuckDB databases under `data/processed/`:
```bash
python src/schemas/build_schemas_v7.py --rows 1000000
```

#### Step 4: Workload Generation & Cross-Schema Compatibility Check
Generate the 223 queries across 6 workload classes (`star_oriented`, `snowflake_oriented`, `fact_constellation_oriented`, `selective`, `multi_dimensional`, `mixed_complex`):
```bash
python src/workload/generate_workload_v8b.py
python src/workload/validate_workload_v8.py --workload data/processed/workload_v8b.json --rows 1000000
```

#### Step 5: Feature Extraction
Extract 113 pre-execution features (no runtime leakage) from DuckDB `EXPLAIN` query plans and SQL syntax:
```bash
python src/workload/feature_extractor_v8b.py --rows 1000000 --workload data/processed/workload_v8b.json
```

#### Step 6: Multi-Schema Execution Benchmarking
Run all queries against all three schemas with timed warm-ups:
```bash
python src/benchmark/run_benchmark_v8.py --rows 1000000 --workload data/processed/workload_v8b.json --repeats 3 --warmup 1
```
Results are saved to `results/v8/scale_raw_results.csv` and `results/v8/scale_summary.csv`.

#### Step 7: Train & Evaluate the Adaptive Controller
Train the Random Forest relative-cost predictor using Leave-One-Workload-Out cross-validation:
```bash
python src/optimizer/train_adaptive_model_v8b.py --rows 1000000
```

---

### Running the Baseline Scalability Experiment (V7)

The V7 experiment evaluated 600 queries (5 classes) across five scales (100K to 10M rows).

```bash
# Data generation and schema build
python src/data/generate_data_v7.py --rows 1000000
python src/schemas/build_schemas_v7.py --rows 1000000

# Workload validation and feature extraction
python src/workload/validate_workload_v7.py --rows 1000000
python src/workload/feature_extractor_v7.py --rows 1000000

# Benchmark and evaluate adaptive model
python src/benchmark/run_benchmark_v7.py --rows 1000000 --repeats 5 --warmup 1
python src/optimizer/train_adaptive_model_v7.py --rows 1000000

# Generate publication figures (produces 7 plots in results/figures/)
python src/analysis/scalability_analysis_v7.py
```

---

## 📊 Experimental Results

### V8 Performance-Aware Benchmark

Evaluated with 223 schema-sensitive queries (balanced execution: 223 queries × 3 schemas × 3 repetitions = 2,007 runs per scale):

| Scale (Dimensions) | Star Mean (ms) | Snowflake Mean (ms) | Fact Constellation (ms) | Best Fixed Schema | Oracle Mean (ms) | Oracle Advantage vs Star | Adaptive vs Star |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100K** (1K / 60) | 6.47 | 7.30 | 6.51 | **Star** | 6.13 | **+5.29%** | -0.20% |
| **500K** (1K / 60) | 10.40 | 11.46 | 10.55 | **Star** | 9.87 | **+5.12%** | +0.07% |
| **1M** (1K / 60) | 16.18 | 17.36 | 16.14 | **FC** | 15.33 | **+5.29%** | -0.34% |
| **1M** (10K / 1K) | 17.71 | 18.82 | 17.86 | **Star** | 17.19 | **+2.98%** | -0.34% |
| **5M** (1K / 60) | 60.14 | 61.26 | 60.55 | **Star** | 58.70 | **+2.40%** | — |
| **10M** (1K / 60) | 125.53 | 126.85 | 125.67 | **Star** | 120.69 | **+3.86%** | — |

**At 1M Scale (Adaptive Controller Performance):**
- **Adaptive vs. Star Baseline**: -0.34% (95% CI: `[-0.15, 0.02]` ms — statistically indistinguishable from Star)
- **Oracle Match Rate**: 33.6%
- **Mean Regret**: 0.59 ms

---

### V7 Baseline Scalability Benchmark

Evaluated with 600 synthetic queries (V3 template generator) across 5 scales:

| Scale | Star Mean (ms) | Snowflake Mean (ms) | FC Mean (ms) | Oracle vs. Star | Adaptive vs. Star | Oracle Match Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100K** | 5.86 | 7.00 | 5.92 | +3.58% | -0.20% | 46.50% |
| **500K** | 7.77 | 8.98 | 7.81 | +4.49% | +0.07% | 48.50% |
| **1M** | 10.92 | 12.04 | 11.08 | +3.07% | -0.28% | 47.67% |
| **5M** | 29.55 | 31.12 | 29.66 | +1.91% | -0.06% | 46.00% |
| **10M** | 51.89 | 53.45 | 51.93 | +1.83% | -0.06% | 43.83% |

---

## 💡 Scientific Findings & Root Cause Analysis

### Core Finding
**Increasing warehouse scale does NOT make adaptive multi-schema selection more effective.**
Under modern vectorized columnar query engines (such as DuckDB), adaptive schema selection provides no statistically significant latency benefit over simply choosing a fixed **Star Schema**.

### Root Cause Analysis

1. **Equivalence of Physical Execution Plans**:
   - For all single-fact queries, the query planner produces identical hash-join execution graphs on both Star and Fact Constellation schemas.
   - Snowflake schemas incur additional join operators without filtering leverage, yielding consistently higher overhead.
2. **Dimension Table Cache Resident**:
   - Even with 10,000 customers and 1,000 products, dimension tables remain fully cached in RAM. Vectorized hash joins execute in microseconds, rendering normalization benefits negligible.
3. **Absence of Physical Index Differentiation**:
   - In standard columnar OLAP formats, all schemas share identical columnar compression and sequential scans. Without specialized indexes, clustered sorting, or projection constraints, schema differences cannot manifest physically.
4. **Compression of the Oracle Opportunity**:
   - As fact table size scales from 100K to 10M rows, the theoretical Oracle headroom contracts from 5.3% down to 2.4–3.8%. A machine learning selector cannot overcome misprediction penalties within this narrow margin.

---

## 🔮 Future Research Directions

Given the empirical evidence demonstrating the limits of pure logical schema selection, promising high-impact alternatives include:
1. **Adaptive Physical Design**: Dynamically managing secondary indexes, zone maps, sorting orders, and materialized views (expected gains: 10%–50%+).
2. **Heterogeneous Query Routing**: Routing queries across distinct specialized execution backends (e.g., vectorized OLAP vs. indexed point-lookup engines).
3. **Automated Partitioning & Pruning**: Partitioning large fact tables along time and geographical dimensions based on query workload trends.

---

## 📄 License

This project is open-source software licensed under the [MIT License](LICENSE).
