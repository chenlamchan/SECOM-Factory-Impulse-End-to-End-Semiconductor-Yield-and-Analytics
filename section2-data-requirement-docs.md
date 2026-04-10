### Section 2 - Data Contract & Requirements

#### 2.1 Source Data Profile — SECOM Raw

| Property       | Value                                                                             |
|----------------|-----------------------------------------------------------------------------------|
| Source         | UCI Machine Learning Repository — SECOM dataset                                   |
| Format         | Two CSV files: `secom.data` (features), `secom_labels.data` (labels + timestamps) |
| Observations   | 1567 rows                                                                         |
| Features       | 590 sensor measurement columns                                                    |
| Label          | Binary: -1 (pass), 1 (fail)                                                       |
| Failure rate   | ~6.35% (93 failures out of 1567) — severe class imbalance                         |
| Missing values | ~5.4% of all values — NOT random (MCAR assumption invalid)                        |
| Timestamp      | Included in labels file — format: `DD/MM/YYYY HH:MM:SS`                           |
| Duplicates     | Partial duplicates exist — require investigation                                  |
| Sensor naming  | Anonymous: columns named `feature_001` through `feature_590`                      |

**Known data quality issues (pre-documented, not discovered at runtime):**

1. **High missingness columns:** Some columns exceed 50% missing — these are candidates for drop, not imputation
2. **Near-zero variance columns:** Several sensors show minimal variation — low predictive value
3. **Class imbalance:** 93 positives vs 1474 negatives — any model metric must account for this
4. **No column metadata:** Sensor meanings are unknown — engineering domain knowledge unavailable
5. **Timestamp gaps:** Not all timestamps are evenly spaced — time-based features require care

#### 2.2 Layer Definitions and Contracts

### Layer: RAW

**Purpose:** Exact copy of source data. Never transformed. Append-only.

| Property           | Rule                                              |
|--------------------|---------------------------------------------------|
| Format             | Parquet (columnar, compressed)                    |
| Partitioning       | By ingestion date: `ingestion_date=YYYY-MM-DD`    |
| Schema enforcement | None — accept source as-is                        |
| Transformations    | None permitted                                    |
| Retention          | Permanent                                         |
| Lineage metadata   | `source_file`, `ingestion_timestamp`, `row_count` |

### Layer: BRONZE

**Purpose:** Schema-normalised, typed, labelled. Minimally cleaned — no imputation yet.

| Property           | Rule                                                          |
|--------------------|---------------------------------------------------------------|
| Format             | Delta Lake / Iceberg                                          |
| Partitioning       | By `ingestion_date`                                           |
| Schema enforcement | Strict — reject schema-violating records, log to quarantine   |
| Transformations    | Column renaming, type casting, label merge, timestamp parsing |
| Retention          | 90 days rolling                                               |
| Lineage metadata   | `source_raw_path`, `bronze_created_at`, `pipeline_version`    |

**Schema definition (bronze layer):**

```
observation_id      : STRING      NOT NULL  -- synthetic PK: "{date}_{row_index}"
timestamp           : TIMESTAMP   NOT NULL  -- parsed from labels file
label               : INTEGER     NOT NULL  -- -1 (pass) or 1 (fail)
feature_001..590    : DOUBLE      NULLABLE  -- raw sensor values, nulls preserved
ingestion_date      : DATE        NOT NULL  -- partition column
source_file         : STRING      NOT NULL  -- lineage: origin filename
pipeline_version    : STRING      NOT NULL  -- lineage: pipeline code version
```

### Layer: SILVER

**Purpose:** Analysis-ready. Imputed, feature-engineered, quality-assured. No modelling yet.

| Property           | Rule                                                                                    |
|--------------------|-----------------------------------------------------------------------------------------|
| Format             | Delta Lake / Iceberg                                                                    |
| Partitioning       | By `label` (for model training efficiency)                                              |
| Schema enforcement | Strict                                                                                  |
| Transformations    | Imputation, feature selection, scaling, engineered features                             |
| Retention          | 90 days rolling                                                                         |
| Lineage metadata   | `bronze_source_version`, `imputation_strategy`, `features_dropped`, `silver_created_at` |

**Schema definition (silver layer, post feature selection):**

```
observation_id          : STRING     NOT NULL
timestamp               : TIMESTAMP  NOT NULL
label                   : INTEGER    NOT NULL
feature_[selected]      : DOUBLE     NOT NULL  -- nulls not permitted post-imputation
imputation_flag_[col]   : BOOLEAN    NOT NULL  -- TRUE if value was imputed
features_dropped_count  : INTEGER    NOT NULL  -- metadata
bronze_source_version   : STRING     NOT NULL
silver_created_at       : TIMESTAMP  NOT NULL
```

### Layer: GOLD [2 subsets: One table for Reporting (Bronze -> Gold) and One Table for model (Bronze -> Silver -> Gold)]

**Purpose:** Model-ready and business-ready. Table 1: Final feature set, encoded, scaled, documented. Table 2: For reporting, no imputation.

| Property           | Rule                                                                                          |
|--------------------|-----------------------------------------------------------------------------------------------|
| Format             | Delta Lake / Iceberg + Parquet export for model consumption, and Delta Lake/Iceberg for       |
| Partitioning       | By `split` (train / test) for Model Table, By `ingestion_date` for reporting                  |
| Schema enforcement | Strict                                                                                        |
| Transformations    | Train/test split, scaling, SMOTE or weight encoding for imbalance, feature importance ranking |
| Retention          | Permanent (model training artifact)                                                           |
| Lineage metadata   | Full chain from raw → bronze → silver → gold versions                                         |

#### 2.3 SLA Definitions

#### 2.4 Quarantine Rules

Any record failing a CRITICAL assertion is written to a quarantine table, not silently dropped.

```
quarantine_table schema:
  observation_id   : STRING
  layer            : STRING   -- "bronze" | "silver" | "gold"
  failed_assertion : STRING
  raw_record       : STRING   -- JSON serialised original record
  quarantine_ts    : TIMESTAMP
  pipeline_run_id  : STRING
```

