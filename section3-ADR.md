## ADR-001: Storage Format for All Layers

**Status:** ACCEPTED  

**Context:**  
Need a storage format for bronze/silver/gold layers that supports: schema enforcement, ACID transactions, time travel (for pipeline reruns and debugging), efficient columnar reads, and compatibility with PySpark, pandas, and DuckDB.

**Decision:**  
Use **Iceberg** for bronze, silver, gold layers. Use raw **Parquet** for the raw layer only (no transformation = no need for ACID).

**Alternatives Considered:**

| Option                     | Rejected Reason                                                                                                        |
|----------------------------|------------------------------------------------------------------------------------------------------------------------|
| Delta Lake                 | Iceberg is free and open source. Delta Lake has stronger PySpark integration and wider ecosystem familiarity for portfolio readers |
| Plain Parquet (all layers) | No ACID, no time travel, no schema enforcement — fails NFR for pipeline reliability                                    |
| Apache Hudi                | Strong for streaming/upserts; overkill for batch-only pipeline at this scale                                           |

**Consequences:**
- Time travel enables safe pipeline reruns by version
- Schema evolution supported without full rewrites
- DuckDB can read Delta Lake tables directly for ad hoc analysis

---

## ADR-002: Orchestration Tool

**Status:** ACCEPTED  

**Context:**  
Need pipeline orchestration with: dependency management, retry logic, scheduling capability, observability, and local development support without heavy infrastructure.

**Decision:**  
Use **Apache Airflow** (local via Astro CLI or Docker) for orchestration.

**Alternatives Considered:**

| Option               | Rejected Reason                                                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Prefect 2.x          | Excellent developer experience, but less universal recognition for portfolio; Airflow remains the most commonly asked-about tool in DE interviews |
| Dagster              | Best-in-class asset-oriented model; considered for post-MVP upgrade. Too steep a setup curve for MVP timeline                                     |
| Luigi                | Dated; limited community; no strong reason to choose                                                                                              |
| Cron + shell scripts | No dependency management, no retry, no UI — fails observability NFR                                                                               |

**Consequences:**
- Airflow adds local setup overhead (Docker or Astro CLI required)
- DAG-based mental model is well-understood by interviewers
- Will document Dagster as the upgrade path in README

---

## ADR-003: Transformation Framework

**Status:** ACCEPTED  

**Context:**  
Need a transformation framework for bronze → silver → gold logic. Must support: column-level lineage, testable transformation functions, compatibility with Delta Lake, and readable code for portfolio reviewers.

**Decision:**  
Use **PySpark** for all transformation logic, with **dbt (dbt-spark)** for SQL-expressible transformations at the silver and gold layers.

**Alternatives Considered:**

| Option      | Rejected Reason                                                                             |
|-------------|---------------------------------------------------------------------------------------------|
| Pandas only | Single-node; cannot demonstrate scalability; not industry-standard for production DE        |
| dbt only    | dbt transforms SQL; Python UDFs needed for imputation, ML feature engineering               |
| Polars      | Fast, modern, but lower portfolio recognition vs PySpark; considered for post-MVP benchmark |

**Consequences:**
- PySpark requires local Spark setup (use PySpark package, not full cluster)
- dbt adds a second tool to explain in portfolio — worth it for demonstrating SQL + Python hybrid pattern
- Test coverage split: pytest for PySpark functions, dbt tests for SQL logic

---

## ADR-004: Data Quality Enforcement Tool

**Status:** ACCEPTED

**Context:**  
Need to enforce Data Contract assertions at each layer automatically, with: readable assertion definitions, integration with pipeline run, failure-blocking capability, and result logging.

**Decision:**  
Use **Great Expectations (GX)** for data quality enforcement. Assertions defined as Expectation Suites, one per layer.

**Alternatives Considered:**

| Option            | Rejected Reason                                                                                                          |
|-------------------|--------------------------------------------------------------------------------------------------------------------------|
| Soda Core         | Cleaner YAML syntax, actively maintained — strong alternative; rejected only because GX has higher portfolio recognition |
| dbt tests         | Covers SQL-layer tests well; cannot cover Spark DataFrame assertions pre-materialisation                                 |
| Custom assertions | Reinventing the wheel; no standard reporting format                                                                      |

**Consequences:**
- GX Expectation Suites stored in `gx/expectations/` directory
- Pipeline fails at layer boundary if CRITICAL assertions fail
- WARNING-level failures log to monitoring table but do not block
- Quarantine table populated on CRITICAL failures

---

## ADR-005: Missing Value Strategy

**Status:** ACCEPTED  

**Context:**  
SECOM dataset has ~5.4% missing values. Distribution of missingness is not random (MCAR assumption invalid based on EDA). Imputation strategy must be: honest (not hiding data quality issues), documented, and appropriate for high-dimensional sensor data.

**Decision:**  
Three-tier strategy: DROP columns with >50% missing → DROP near-zero variance columns → MEDIAN IMPUTE remaining missing values (global median, not per-label to avoid leakage).

**Alternatives Considered:**

| Option                     | Rejected Reason                                                                    |
|----------------------------|------------------------------------------------------------------------------------|
| KNN imputation             | Too computationally expensive for 590 dimensions at MVP; considered for post-MVP   |
| MICE / Multiple imputation | Statistically superior; complexity not justified at MVP pipeline stage             |
| Mean imputation            | Less robust to outliers than median; sensor data is frequently skewed              |
| Forward-fill (time-based)  | Assumes temporal relationship between missing values — not validated for SECOM     |
| Zero-fill                  | Introduces false signal; semantically incorrect for continuous sensor measurements |

**Consequences:**
- Drop threshold (50%) is a documented decision, not a magic number — revisable with evidence
- Imputation flags added as boolean columns for each imputed cell — model can use these as features
- Global median computed on full training set only, applied to test set (no leakage)

---

## ADR-006: Class Imbalance Handling

**Status:** ACCEPTED  

**Context:**  
SECOM has ~6.35% positive (failure) class. Naive model will achieve >93% accuracy by predicting all-negative — useless for the business case. Must handle imbalance at either data level or model level.

**Decision:**  
Use **class weighting** at model training level (`class_weight='balanced'` in sklearn). Do NOT use SMOTE or oversampling at the data pipeline level for MVP.

**Alternatives Considered:**

| Option                         | Rejected Reason                                                                         |
|--------------------------------|-----------------------------------------------------------------------------------------|
| SMOTE oversampling in pipeline | Generates synthetic data — complicates lineage and contract assertions; post-MVP option |
| Undersampling majority         | Discards real data; information loss on an already small dataset                        |
| Threshold tuning only          | Insufficient alone without addressing training signal                                   |
| Cost-sensitive learning        | Subsumed by class weighting at MVP scale                                                |

**Consequences:**
- Gold layer does NOT include SMOTE data — pipeline outputs are always real observations
- Class weight parameter documented in model config, not hardcoded
- Primary evaluation metric is F1 (positive class) and PR-AUC, not accuracy
- SMOTE variant deferred to post-MVP with full lineage impact assessment
