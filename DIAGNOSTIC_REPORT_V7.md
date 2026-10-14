# Diagnostic Report: Why V7 Adaptive Selection Shows No Measurable Improvement

**Project:** Adaptive Multi-Schema Data Warehouse  
**Version:** V7 (Baseline Experiment A: "Scale Sensitivity of Adaptive Multi-Schema Selection")  
**Date:** 2026-10-04

---

## Executive Summary

The V7 experiment successfully executed across 5 scales (100K–10M rows) with 600 queries. The findings are clear and honest:

| Scale | Oracle vs Star | Oracle vs Best Fixed | Oracle Match Rate |
|-------|---------------|---------------------|-------------------|
| 100K  | 3.58%         | 3.58%               | 46.5%             |
| 500K  | 4.49%         | 4.49%               | 48.5%             |
| 1M    | 3.07%         | 3.07%               | 47.7%             |
| 5M    | 1.91%         | 1.91%               | 46.0%             |
| 10M   | 1.83%         | 1.83%               | 43.8%             |

**The adaptive controller never statistically outperforms fixed Star.** The oracle improvement decreases with scale. This is not a model failure—it is a **workload/schema mismatch**.

---

## Root Cause Analysis

### 1. Fact Constellation Is Effectively Unused

- **0 / 600 queries** reference `fact_returns`
- The Fact Constellation schema includes `fact_returns` (1.24M rows at 10M scale) but the workload never joins it
- **Result:** Star and Fact Constellation have **identical physical plans** for every query—both scan `fact_sales` and join the same 3 dimensions
- Oracle sees Star and FC as tied; Snowflake is strictly slower

### 2. Snowflake Adds Pure Overhead

The Snowflake schema normalizes:
- `dim_customer` → `dim_region` + `dim_segment` (2 extra joins)
- `dim_product` → `dim_category` (1 extra join)

**Query plan evidence** (Q001, aggregation by year):
- Star/FC: 3 hash joins (fact_sales → customer, product, date)
- Snowflake: **7 hash joins** (3 base + 2 for customer path + 1 for product path + 1 for region)

The compatibility views (`vw_customer`, `vw_product`) force the optimizer to traverse the full normalized path even though the query only filters on `region`/`segment` names. No query benefits from the normalized structure because:
- No query drills from region → segment → customer
- No query filters on `region_id`/`segment_id` directly (always on names)
- Dimension tables are tiny (4 regions, 3 segments, 3 categories); the "smaller scan" benefit is negligible

### 3. Workload Lacks Schema-Discriminating Patterns

| Pattern | Current Workload | Needed for Schema Separation |
|---------|-----------------|------------------------------|
| Multi-fact joins (sales + returns) | 0 queries | **Required** for FC advantage |
| Deep dimension traversal (region→segment→customer) | 0 queries | **Required** for Snowflake advantage |
| Fact-to-fact analysis | 0 queries | **Required** for FC advantage |
| High-cardinality GROUP BY (product_name, customer_name) | 14 queries (3 cols) | Helpful for Star |
| Selective filters on normalized keys | 0 queries | **Required** for Snowflake |
| Time-series sales vs returns | 0 queries | **Required** for FC advantage |

**Join distribution:** 67% of queries use 3 joins (full star join); 16% use 2 joins; 7% use 1 join; 9% use 0 joins (selective scans).

### 4. Dimension Cardinalities Are Too Small for Normalization to Matter

| Dimension | Cardinality | In-Memory? |
|-----------|-------------|------------|
| dim_region | 4 | Yes |
| dim_segment | 3 | Yes |
| dim_category | 3 | Yes |
| dim_customer | 1,000 | Yes |
| dim_product | 60 | Yes |
| dim_date | 365 | Yes |

All dimension tables fit entirely in DuckDB's buffer. The theoretical benefit of Snowflake (scanning smaller normalized tables instead of wide denormalized ones) is **zero** because:
- The denormalized dimensions are already tiny
- Hash joins on 1K/60-row tables are near-instant
- The extra join overhead dominates

### 5. Query Adapter Masks Physical Differences

The `adapt_sql()` function in `validate_workload_v7.py` rewrites Snowflake queries to use `vw_customer`/`vw_product` views. These views **materialize the full denormalized path** (customer→region→segment, product→category). The optimizer sees:
- Star: `fact_sales → dim_customer → dim_product → dim_date` (3 joins)
- Snowflake: `fact_sales → vw_customer → vw_product → dim_date` (views expand to 7 joins)

The logical query is identical; the physical difference is only extra join operators in Snowflake.

---

## Schema Plan Comparison (Representative Queries)

### Aggregation Query (Q001) — GROUP BY d.year
| Schema | Joins | Plan Operators | Estimated Rows After Joins |
|--------|-------|----------------|---------------------------|
| Star | 3 HASH_JOIN | 13 | ~8.3M |
| Snowflake | 7 HASH_JOIN | 25 | ~8.3M |
| Fact Constellation | 3 HASH_JOIN | 13 | ~8.3M |

**Result:** Identical output cardinality; Snowflake does 2.3× more join work.

### Selective Query (Q005) — region='South' filter
| Schema | Joins | Filter Pushdown |
|--------|-------|-----------------|
| Star | 1 HASH_JOIN (fact_sales→dim_customer) | Filter on `region='South'` applied during dim_customer scan |
| Snowflake | 3 HASH_JOIN (fact_sales→dim_customer→dim_region→dim_segment) | Filter on `region='South'` pushed to dim_region scan (1 row), but still requires 2 extra joins |

**Result:** Snowflake does 3× more joins for same filter.

---

## Why Scale Doesn't Help

| Scale | Fact Rows | Dim Rows | Join Cost Ratio (Snowflake/Star) |
|-------|-----------|----------|----------------------------------|
| 100K  | 100K      | Fixed    | ~2.3× |
| 1M    | 1M        | Fixed    | ~2.3× |
| 10M   | 10M       | Fixed    | ~2.3× |

Scaling the fact table increases **absolute** latency proportionally for all schemas, but the **relative** overhead of Snowflake's extra joins remains constant. The Oracle advantage (which only comes from Star vs Snowflake noise) actually **shrinks** because the signal-to-noise ratio worsens.

---

## Conclusion: What Must Change for V8

To make adaptive schema selection scientifically meaningful, we need:

1. **Queries that USE fact_returns** — multi-fact analytical patterns (sales vs returns, return rates, customer return behavior)
2. **Queries that benefit from normalized dimensions** — deep hierarchy traversal, selective filters on `region_id`/`segment_id`, drill-down patterns
3. **Queries where Snowflake's smaller dimension scans win** — increase dimension cardinalities (10K+ customers, 1K+ products) so that scanning `dim_region` (10 rows) vs `dim_customer` (10K rows) matters
4. **Remove the compatibility view crutch** — let Snowflake queries express the normalized path naturally
5. **Workload classes explicitly designed for each schema's strength**

---

## Recommended V8 Workload Design

| Class | Target Schema | Example Patterns |
|-------|--------------|------------------|
| A. Star-oriented | Star | Wide aggregations, full fact scans, low-selectivity GROUP BY |
| B. Snowflake-oriented | Snowflake | Region→Segment→Customer drill-down, filters on normalized keys, high-cardinality dimension scans |
| C. Fact Constellation | FC | Sales + Returns joins, return rate by product/region, multi-fact time series |
| D. Selective | Any | Point lookups, PK filters |
| E. Large aggregation | Star | Full fact scans, minimal dimensions |
| F. Multi-dimensional grouping | All | 3-4 GROUP BY columns, cross-dimensional analysis |

**Critical:** Do NOT hard-code expected winners. Let the Oracle experiment reveal the truth.

---

## Next Steps

1. **PHASE 2-3:** Design `generate_workload_v8.py` with schema-sensitive query templates
2. **PHASE 4:** Run Oracle Gap Experiment on 1M scale (fast feedback)
3. **PHASE 5:** Evaluate — if Oracle improvement < 5%, iterate workload design
4. **PHASE 6+:** Only if Oracle succeeds, build V8 adaptive model

**Files to create/modify:**
- `src/workload/generate_workload_v8.py` — new schema-sensitive workload
- `src/workload/validate_workload_v8.py` — validation without forced view rewrites
- `src/workload/feature_extractor_v8.py` — features for new query patterns
- `src/data/generate_data_v8.py` — richer dimensions if justified
- `src/optimizer/train_adaptive_model_v8.py` — new model (only after Oracle success)
- `results/v8/` — isolated results directory