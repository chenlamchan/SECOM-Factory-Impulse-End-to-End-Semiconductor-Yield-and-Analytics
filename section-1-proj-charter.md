### One Page Charter:

#### 1.1 Problem Statement
Semiconductor manufacturing produces thousands of sensor measurements per wafer across hundreds of process steps. Identifying which signals predict yield failure is currently manual, slow, and reactive. 
Failures are discovered at end-of-line inspection, too late to intervene, resulting in scrap cost and throughput loss.

The business question: 
- Can we predict wafer yield failure from in-process sensor data, early enough for intervention?
- Can we reduce the test time by reducing the number of test variables in overall?

---
#### 1.2 Project Objective
Build an end-to-end data platform that

1. Data Generation: Based on the raw SECOM dataset, create a synthetic data generating module that stay intact (outliers, missing data, inconsistency, data distribution etc.), to simulate the actual data generation at the source.
2. Data Ingestion: Ingests and cleans raw SECOM data into a scalable data storage and pipelines.
3. Data Reporting: Report the current total output, yield, process capability, PPM and related business metrics through dashboard, and update accordingly.
4. ML Analytics:
   - Data Processing: Applies feature engineering, dimensionality reduction, feature selection technique to identify predictive signals.
   - Data Preparation: Produce a model-ready gold layer dataset with documented lineage.
   - Model Building & Training: Train model to achieve the business requirements with experiment logged.
   - Model Evaluation: Evaluate the model with test dataset with result logged. (Iterative experiment)
   - Model Validation: Exposes yield failure predictions and key signal analysis based on the ML model, and to compare the test (features) involved, the predicted output, yield andother related business metrics comparing to full test.
  
This is a portfolio project demonstrating data engineering with ML analytics, system design and AI-augmented development workflow. 

---
#### 1.3 Success Metrics (To agents and the system) - Experimental

| Metric                                               | Target | How Measured                                        |
|------------------------------------------------------|--------|-----------------------------------------------------|
| Pipeline runs end-to-end from raw to gold            | Yes    | Single command execution (Local Dockerized for MVP) |
| Data contract assertions pass at all layers          | 100%   | Great Expectations                                  |
| Missing value handling is documented and consistent  | Yes    | DRD + lineage metadata                              |
| Failure prediction F1 score (positive class)         | ≥ 0.65 | Model evaluation notebook                           |
| Top 20 predictive features identified and documented | Yes    | Feature importance output                           |
| README tells the system story with 3+ ADRs explained | Yes    | Human review                                        |

---
#### 1.4 Scope Boundary

**In scope:**
- Batch ingestion pipeline from raw CSV to gold layer
- Medallion architecture: raw → bronze → silver → gold
- Missing value imputation strategy (documented)
- Feature engineering and selection layer
- Baseline classification model (failure prediction)
- Data quality contract enforcement at each layer
- Reproducible pipeline execution
- Documentation for the portfolio and architectural decision

**Explicitly out of scope (MVP):**
- Real-time / streaming ingestion
- Production deployment or serving API
- Retraining pipeline / MLOps
- Multi-dataset support
- Alerting and monitoring infrastructure

---
#### 1.5 Constraints

| Constraint  | Detail                                                              |
|-------------|---------------------------------------------------------------------|
| Dataset     | SECOM UCI — fixed, cannot add new source data                       |
| Environment | Local development (Docker Compose                                   |
| Stack       | Must be demonstrable in a GitHub repo with clear setup instructions |

---
#### 1.6 Done Criteria

