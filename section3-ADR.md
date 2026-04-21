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

---

### ADR-007: Message Broker for Data Ingestion Buffering

**Status:** ACCEPTED

**Context:**
The system requires a reliable handshake between the **Data Generation** module (simulating sensor output) and the **Data Ingestion** pipeline (Airflow). We need a mechanism to:
1.  **Prevent Data Loss:** If Airflow is offline or a task fails, the "file ready" signals must be persisted.
2.  **State-Based Polling:** Airflow needs to track progress via a "last processed" timestamp, but requires a buffer to handle bursts of synthetic data generation.
3.  **Local Resource Efficiency:** The entire stack must run via Docker Compose on a single machine without consuming the high memory/CPU overhead associated with enterprise-scale brokers.

**Decision:**
Use **NATS with JetStream** enabled as the primary message broker and persistence buffer.

**Rationale:**
* **Simplicity:** NATS is a single 20MB binary. It is "cloud-native" and requires zero external dependencies (unlike Kafka which requires ZooKeeper or KRaft).
* **JetStream Persistence:** By using JetStream, we gain "at-least-once" delivery guarantees. If the Airflow consumer fails, the message remains in the NATS stream until acknowledged.
* **SQS-like Experience:** NATS provides a simple Subject-based messaging model that mimics the "lightweight" feel of AWS SQS, making it ideal for managing file-path metadata and timestamps.

**Alternatives Considered:**

| Option | Rejected Reason |
| :--- | :--- |
| **Apache Kafka** | Overkill for this scale. Requires significant JVM memory (2GB+) and complex configuration (partitions, offsets, ZK/KRaft) that complicates the "Single Command Execution" goal. |
| **RabbitMQ** | Heavier resource footprint (Erlang VM). While feature-rich, the management of exchanges/bindings adds unnecessary complexity for a simple file-tracking buffer. |
| **Redis (Pub/Sub)** | Lacks native persistence and "replay" capabilities in its standard Pub/Sub mode. While Redis Streams exist, NATS JetStream offers a more robust "store-and-forward" model for this specific use case. |
| **Direct File Polling** | Without a broker, Airflow would need to scan the disk constantly. This is brittle and risks missing files if the naming convention or timestamps have minor inconsistencies during high-frequency generation. |

**Consequences:**
* **Lightweight Footprint:** The NATS container will typically use < 50MB of RAM, leaving more resources for ML model training and Airflow workers.
* **Decoupling:** The Data Generator doesn't need to know if Airflow is running; it simply publishes "File Created" events to the NATS stream.
* **Observability:** We can use the `nats` CLI or a simple UI to inspect the buffer, making debugging of the ingestion pipeline significantly easier during development.
* **State Management:** Airflow will use a "Pull Consumer" pattern, fetching the next available file metadata from NATS based on the stream's sequence, ensuring no file is processed twice or skipped.

---
### ADR-008: Compute Engine Pattern for Raw to Bronze Ingestion

**Status:** ACCEPTED

**Context:**
We need to ingest event-driven, raw Parquet files from an S3-compatible object store (MinIO) into our Iceberg Bronze layer. The raw dataset contains ~590 sensor features and is subject to potential schema drift. The ingestion process requires reading the files, appending basic Medallion lineage metadata (e.g., `ingestion_timestamp`), and appending the records to the Iceberg catalog. We need a solution that minimizes compute overhead, handles dynamic schemas gracefully, and keeps Airflow strictly constrained to a control plane (orchestration only) without running heavy data processing tasks on the Airflow workers.

**Decision:**
Implement a **Containerized Task Pattern**. We will use PyIceberg and PyArrow inside a lightweight, ephemeral Docker container. Airflow will utilize the `DockerOperator` to spin up the container, pass the raw S3 file paths dynamically as arguments, execute the Python ingestion script, and immediately tear the container down upon completion. 

**Alternatives Considered:**

| Option | Rejected Reason |
| :--- | :--- |
| **PySpark (`SparkSubmitOperator`)** | Massive JVM overhead, requires maintaining an active Spark cluster (master/worker nodes), and is severe overkill for a simple file-to-table read/append operation. |
| **Trino Pushdown (`SQLExecuteQueryOperator`)** | Trino is an MPP engine optimized for structured tables. Querying ad-hoc S3 paths requires spoofing a file-based Hive Metastore and writing rigid, upfront DDLs. It is clunky for raw object storage and breaks easily if the raw schema drifts. |
| **Local Python in Airflow (`PythonOperator`)** | Violates the architectural principle of treating Airflow purely as an orchestrator. Running PyArrow/PyIceberg directly on Airflow workers risks memory bloat and dependency conflicts. |

**Consequences:**
* **Performance:** Lightning-fast execution. PyArrow reads Parquet files directly into memory via C++ bindings, drastically reducing ingestion latency compared to spinning up Spark executors.
* **Flexibility:** Dynamic schema inference via PyArrow eliminates the need for rigid DDL definitions or Hive metastore hacks for the raw layer.
* **Resource Efficiency:** Zero idle compute. The container exists only for the seconds it takes to process the files.
* **Trade-off:** Introduces the minor operational overhead of building and maintaining a custom Docker image for the PyIceberg script.

---
This is a great way to formalize the architectural decisions we just made. Documenting these logic choices is vital because "why" a specific SPC math was chosen is often lost over time.

Based on our discussion regarding the transition from global statistics to an asset-specific, 2-pass baseline approach, here is the ADR.

---

### ADR-009: Asset-Specific 2-Pass Phase I SPC Baseline

**Status:** ACCEPTED

**Context:** The initial SPC implementation calculated global control limits ($\mu, \sigma$) across the entire factory and all sensors, ignoring the unique calibration baselines of individual testers and lines. Furthermore, the baseline calculation was "polluted" by anomalies and was designed to aggregate all historical data continuously. This led to "limit chasing" (where limits adapt to bad data, hiding drift) and inflated variance due to outliers. We also lacked parity between the SQL engine and Python engine regarding Nelson Rule 4 (14 alternating points).

**Decision:** Implement an **Phase I Frozen Baseline** pattern using a **2-Pass Filtered Calculation** at the asset-grain.

1.  **Asset-Specific Partitioning:** Every window function and join for stats and violations is now partitioned by `(line_id, tester_id, sensor_id)`.
2.  **2-Pass Baseline Logic:** * **Pass 1:** Calculate "raw" stats for the first $N$ observations per asset.
    * **Pass 2:** Re-calculate final stats after excluding any observations from that window that exceed $\pm3\sigma_{raw}$ to prevent inflated variance from outliers.
3.  **Frozen Limits:** Once the Phase I baseline is established, limits are "frozen" and stored. Future observations (Phase II) are evaluated against these static limits to ensure drift is detectable.
4.  **Rule 4 Implementation:** Added SQL logic to detect 12 consecutive sign flips in direction, representing 14 alternating points.
5.  **Dbt Parameterization:** The baseline sample size (e.g., 100) is now controlled via `dbt_project.yml` variables for easy tuning.

**Alternatives Considered:**

| Option | Rejected Reason |
| :--- | :--- |
| **Continuous Moving Average** | Causes "limit chasing." If the process drifts slowly, the mean follows it, and the violation is never triggered (the boiling frog problem). |
| **Median-based SPC** | While robust to outliers, it is mathematically disconnected from the Standard Deviation ($\sigma$) formula and lacks sensitivity to small, important process shifts. |
| **Global Factory Limits** | Ignores physical differences and calibrations between individual testers, leading to high false-alarm rates on specific machines. |

**Consequences:**
* **Statistical Integrity:** The 2-pass method ensures that the "normal" process variation is not artificially widened by the very anomalies the system is supposed to catch.
* **Sensitivity:** The system is now significantly more sensitive to tool-specific drift and wear.
* **Operational Flexibility:** Data scientists can tune the baseline sensitivity (sample size) globally through dbt variables without modifying core SQL models.
* **Parity:** The SQL monitoring engine now perfectly mirrors the logic used in the Python-based `SPCEngine`.

---
Since this is a fundamental design choice for your control strategy, it deserves its own **ADR**. This documents why you are monitoring every single wafer (Individual X-Chart) rather than averaging them by lot ($\bar{X}$-Chart), especially given the skewness concerns we discussed.

---

### ADR-010: Selection of Individual (X) Charting vs. Subgroup ($\bar{X}$) Charting

**Status:** ACCEPTED

**Context:** The SECOM dataset provides 100% metrology data (one record per wafer) rather than sampled data. We need to decide if the SPC engine should monitor individual wafers (X-Chart) or aggregate wafers into batches/lots ($\bar{X}$-Chart). While $\bar{X}$-charts are more robust to non-normal (skewed) distributions due to the Central Limit Theorem, the business requirement for SECOM analytics is to identify specific wafer-level failures that lead to scrap.

**Decision:** Implement **Individual (X) Control Charts** as the primary monitoring pattern for the Gold Layer. Every observation in the `silver_secom_reporting` table is evaluated as a standalone point.

**Rationale:**
* **Metrology Density:** Since the data is already captured for 100% of the wafers, an Individual Chart provides the highest resolution of process visibility.
* **Sensitivity to Catastrophic Failure:** A single wafer with a massive electrical short or physical defect might be "hidden" if averaged into a batch of 24 good wafers. Individual monitoring ensures these "spikes" trigger Rule 1 violations immediately.
* **Pipeline Simplicity:** Monitoring individuals avoids the overhead of complex window-aggregation logic (calculating lot means) before applying Nelson Rules.

**Alternatives Considered:**

| Option | Rejected Reason |
| :--- | :--- |
| **Subgroup ($\bar{X}$) Charting** | While better at handling skewed data, it significantly reduces the ability to pinpoint which specific wafer in a lot caused a process deviation. It is better suited for high-volume sampling where not every wafer is measured. |
| **Combined X/$\bar{X}$ Monitoring** | Leads to "Alarm Fatigue." If a lot is drifting, the system would trigger 25 individual alarms plus 1 batch alarm for the same root cause, overwhelming the operator dashboard. |

**Consequences:**
* **Skewness Management:** Because individual sensor data in SECOM can be skewed, we must rely on the **ADR-009 2-Pass Baseline** or mathematical transformations (like Log-normal) rather than relying on the Central Limit Theorem to "fix" the distribution.
* **Noise Sensitivity:** Individual charts are more sensitive to "jitter" (measurement noise). This is mitigated by the `_apply_mutations` logic in the generation engine which controls jitter variance.
* **Nelson Rule Application:** Rules like Rule 2 (9 points on one side) and Rule 3 (6 trending) are applied across the wafer sequence, which effectively detects the same "lot-level" drift that an $\bar{X}$-chart would, but with wafer-level granularity.

---
### ADR-011: Selection of Nelson Rules for Automated Process Monitoring

**Status:** ACCEPTED

**Context:** We need a standardized logic to detect "out-of-control" states in the SECOM sensor data. While several SPC rule sets exist (Western Electric, Wheeler, AIAG), we must choose a set that balances the sensitivity to catch equipment drift against the risk of "false alarms" (Type I errors) in a high-volume semiconductor environment.

**Decision:** Implement a subset of the **Nelson Rules** (specifically Rules 1, 2, 3, 4, and 5) as the primary detection logic.



**Rationale:**
* **Industry Standard:** Nelson rules are the widely accepted successor to the Western Electric (WECO) rules, providing a more comprehensive set of patterns for modern, computerized metrology systems.
* **Pattern Versatility:** * **Rule 1 (Beyond 3$\sigma$):** Detects "Special Cause" variation or immediate catastrophic failure.
    * **Rules 2 & 3 (Run and Trend):** Detect gradual tool wear or chemical depletion that a single-point check would miss.
    * **Rule 4 (Alternating):** Specifically targets systematic issues like "hunting" in automated feedback loops or valve oscillations.
    * **Rule 5 (Warning Zone):** Acts as an early-warning system before a process officially crosses the 3$\sigma$ failure line.
* **Computational Compatibility:** The Nelson rules are mathematically structured as "rolling windows," making them highly efficient to implement in SQL window functions (Trino) and Python/Pandas rolling vectors.

**Alternatives Considered:**

| Option | Rejected Reason |
| :--- | :--- |
| **Western Electric (WECO) Rules** | Only includes 4 rules. It lacks the "Trend" and "Alternating" detection capabilities required for complex semiconductor gas/thermal sensors. |
| **Simple 3$\sigma$ Thresholding** | Only catches "gross" failures. By the time a 3$\sigma$ violation occurs in a fab, you may have already produced an entire lot of scrap wafers that could have been caught earlier by a "Run" or "Trend" rule. |
| **Machine Learning (Anomaly Detection)** | While powerful, ML lacks the **transparency** of SPC. Operators need to know *why* a machine stopped (e.g., "The mean shifted," not "The black-box score was 0.98"). |

**Consequences:**
* **Specificity Tuning:** Using more rules increases the chance of "false positives" due to natural noise. This is why we have parameterized the `jitter_variance` in the generator to ensure the baseline is stable.
* **Compute Grain:** The rules are applied sequentially. We have ensured that the `rn` (row number) used for these rules is strictly partitioned by asset to prevent "Rule 2" or "Rule 3" from accidentally counting points across different machines.
* **Dashboard Logic:** The dashboard must now display the specific **Rule Number** and **Name** so maintenance teams know whether to look for a sudden spike (Rule 1) or a slow drift (Rule 3).

---
## ADR-012: Derivation of Global Specification Limits via Golden Line Benchmarking

**Status:** ACCEPTED  
**Context:** The SECOM dataset lacks explicit engineering **Specification Limits** ($USL$/$LSL$). In a multi-line simulation (Lines A, B, and C), evaluating process capability ($Cpk$) requires a universal standard of "good." Without a fixed reference, each line is only measured against its own history, which masks performance gaps between different pieces of equipment.

**Decision:** Implement a **Golden Tool** benchmarking strategy within the `gold_sensor_stats` dbt model. **Line A** (running raw, non-synthetic SECOM data) is designated as the factory standard. 
* **Global Specification Limits ($USL$/$LSL$):** Calculated strictly from the first 100 observations of **Line A**.
* **Local Control Limits ($UCL$/$LCL$):** Calculated independently for every individual Line/Tester/Sensor combination based on their own first 100 observations.

**Logic:**
The global standard for each sensor is reverse-engineered from Line A’s baseline, targeting a $Cpk = 1.33$:
$$USL = \mu_{LineA} + (3.99 \times \sigma_{LineA})$$
$$LSL = \mu_{LineA} - (3.99 \times \sigma_{LineA})$$

**Rationale:**
* **Golden Tool Concept:** In real-world manufacturing, product specifications are determined by design intent or the performance of the most stable "Golden" machine. Using Line A as this reference provides a grounded, non-arbitrary standard.
* **Cross-Line Benchmarking:** By applying Line A’s specs to Lines B and C, the $Cpk$ gauge accurately reflects if synthetic drifts or equipment aging in those lines are making them less capable than the "ideal" state represented by Line A.
* **Statistical Stability:** Since Line A uses raw SECOM data without synthetic injections, it provides the most authentic representation of the process's natural "Best Demonstrated Performance."
* **Dual-Layer Monitoring:** * **Local Layer:** Nelson Rules trigger based on a machine's own $UCL/LCL$, catching sudden shifts.
    * **Global Layer:** $Cpk$ gauges trigger based on Line A's $USL/LSL$, catching chronic underperformance.

**Alternatives Considered:**
| Option | Rejected Reason |
| :--- | :--- |
| **Local Spec Derivation** | If Line B is inherently "noisier" than Line A, deriving specs from Line B data would "forgive" that noise, resulting in a misleadingly high $Cpk$. |
| **Universal Control Limits** | Rejected because different testers often have unique mechanical signatures (offsets). Forcing Line A’s $UCL/LCL$ on Line B would cause constant "false alarm" Nelson Rule violations. |

**Consequences:**
* **Line A Dependency:** The `gold_sensor_stats` dbt model must ensure Line A data is processed first or is present in the `silver` table before final limits are broadcast.
* **Deterministic Calibration:** Both the Golden Specs and Local Controls are "frozen" after the Phase I window (100 samples) to prevent the "shifting baseline" problem where the standard moves as the process degrades.
* **UI Interpretation:** Capability gauges across all lines now answer the specific question: *"How well is this specific machine meeting the high standard set by our Golden Line?"*

---
**ADR-013: Virtual Feature Store for Batch ML Engineering**

**Status:** ACCEPTED

**Context:** The SECOM (Semiconductor Manufacturing) project involves high-dimensional sensor data (590+ features) characterized by high noise, missing values, and the need for complex preprocessing (imputation, scaling). Since the project is strictly **batch processing** (e.g., generating quality reports for batches of wafers), we do not require the sub-millisecond latency provided by in-memory databases. We need a mechanism that:
1. **Ensures Reproducibility:** Allows "Time Travel" to query data exactly as it existed during a specific production window.
2. **Handles Schema Evolution:** Manages 500+ sensor columns that may change, disappear, or be renamed over time.
3. **Optimizes for Training:** Provides a "Point-in-Time" correct view of features to prevent data leakage during model training.

**Decision:** Use **Apache Iceberg** as the table format (Storage) and **dbt (data build tool)** as the transformation engine to create a "Headless" Feature Store.

**Rationale:**
* **Point-in-Time Correctness:** Iceberg’s native snapshotting allows us to perform "as-of" queries, ensuring that the features used for training perfectly match the historical state of the manufacturing line.
* **SQL-as-Code:** dbt allows us to define feature engineering logic (e.g., sensor normalization) in version-controlled SQL, treating the feature store as a governed software product.
* **Schema Flexibility:** Iceberg supports full schema evolution (adding/dropping sensor columns) without requiring a full rewrite of the historical dataset, which is critical for high-dimensional manufacturing data.
* **Infrastructure Simplicity:** By avoiding a specialized feature store service (like Feast) or an online database (like Redis), we reduce the system's memory footprint and operational overhead.

**Alternatives Considered:**

| Option | Rejected Reason |
| :--- | :--- |
| **Feast + Redis** | Overkill for batch processing. Adds significant overhead (registry management + RAM costs) to solve a "latency" problem that doesn't exist in this use case. |
| **Standalone Redis** | Lacks persistence for historical data and "Time Travel" capabilities. Redis is a serving layer, not a data management layer for 500+ features. |
| **Direct CSV/Parquet** | Lacks ACID transactions and governance. Managing schema changes across 590 columns manually in raw files is brittle and prone to "Data Swamp" issues. |
| **Featureform** | While it supports Iceberg, adding another orchestration layer is unnecessary for a personal project where dbt + Iceberg already provides a clean "Virtual" feature store. |

**Consequences:**
* **Batch-Only Performance:** Querying features will take seconds rather than milliseconds. This is acceptable for the SECOM batch reports but would require an additional "Online" sink (like Redis) if the project pivots to real-time monitoring.
* **Storage Efficiency:** Features are stored as compressed Parquet files within the Iceberg format, drastically reducing disk space compared to an in-memory or row-based SQL database.
* **Simplified Pipeline:** The "Feature Store" is simply a collection of dbt models. There is no separate API to maintain; the "Feature Registry" is effectively the dbt documentation and schema.yml files.
* **ML Integration:** Model training scripts can point directly to Iceberg tables via DuckDB or PyIceberg, providing a high-performance path for large-scale feature extraction.

---
This is a great approach. By shifting from a **Hard Delete** (dropping columns physically) to a **Soft Delete** (keeping all data but filtering via metadata), you've essentially created a "Versioned Feature Contract."

Here is the ADR following your template, documenting the transition to a manifest-driven, resilient feature engineering strategy.

---

## ADR-014: Manifest-Driven "Soft" Feature Selection and Imputation

**Status:** ACCEPTED

**Context:** In the SECOM manufacturing environment, sensor reliability is variable. A sensor that fails the "Missingness Threshold" (>40% nulls) today might be repaired and provide high-quality data tomorrow. Hard-dropping columns during the extraction phase (`extract_features.py`) creates a brittle pipeline where the schema of historical snapshots changes unpredictably, making model retraining and "Time Travel" difficult. Furthermore, the prediction service needs a deterministic way to handle missing values at inference time to prevent runtime errors and training-serving skew.

**Decision:** We will implement a **"Soft Feature Selection"** strategy. The PySpark extraction job will persist all 590+ sensors in the Iceberg snapshot, while a companion `feature_manifest.json` will act as the source of truth for which features are "active." Additionally, the training pipeline will calculate and store baseline medians for every active feature within this manifest.

**Rationale:**
* **Pipeline Robustness:** By keeping all features in the Iceberg snapshot, we avoid data loss. If selection logic changes (e.g., we lower the variance threshold), we can update the manifest without re-running the heavy extraction job.
* **Training-Inference Alignment:** Using a manifest file ensures that the prediction service uses the exact same feature list and imputation values (medians) that the model was trained on. This is a critical industry practice to prevent "stale" model behavior.
* **Resilience via Fallbacks:** Persisting medians provides a "safety net." If a sensor breaks in production, the prediction service can use the training-time median as a temporary placeholder, allowing the system to remain operational (graceful degradation) rather than failing.
* **Explainability (CFE) Baseline:** The medians stored in the manifest provide a "normal" baseline required for generating Counterfactual Explanations, helping engineers understand why a specific wafer was flagged as a failure compared to the "typical" state.

**Implementation Details:**
1.  **Extraction:** `extract_features.py` identifies "selected_features" but writes the full wide-table to Iceberg.
2.  **Profiling:** `prepare_features.py` calculates the 50th percentile (`percentile_approx`) exclusively from the **training split** to avoid data leakage.
3.  **Contract:** The `feature_manifest.json` is passed as a build artifact to the model training and prediction service tasks.

**Alternatives Considered:**
* **Hard-Dropping Columns:** Rejected because it requires expensive data rewrites if feature selection criteria are tuned.
* **Real-time Imputation:** Rejected because calculating medians on-the-fly in a prediction service adds unnecessary latency and requires access to historical data windows at runtime.
* **Constant Value Imputation (e.g., 0):** Rejected because it can introduce significant bias in manufacturing sensor data where "0" might be a valid, extreme measurement.

**Consequences:**
* **Manifest Dependency:** The `feature_manifest.json` becomes a critical path artifact. If it is lost or corrupted, the inference service cannot map input data to model features.
* **Storage Overhead:** Keeping "dropped" features in the Iceberg table increases storage footprint slightly, though Iceberg/Parquet compression mitigates this for columns with high null counts.
* **Traceability:** Every model version is now explicitly tied to a manifest version, providing an audit trail of exactly what sensors and imputation values were used for any historical prediction.

---
