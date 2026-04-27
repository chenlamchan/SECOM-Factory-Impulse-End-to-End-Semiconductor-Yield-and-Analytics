```.
├── airflow/                 # Workflow Orchestration
│   ├── dags/                # DAG definitions and business logic
│   └── common/              # Shared python utilities for Airflow
├── analytics/               # Data Transformation Layer (dbt)
│   ├── models/              # staging, silver, and gold models
│   ├── macros/              # Custom dbt Jinja macros
│   └── dbt_project.yml      # dbt configuration
├── infrastructure/          # Platform, Storage & Observability
│   ├── airflow/             # Airflow initiation scripts
│   ├── minio/               # Object storage bucket initialization
│   └── postgres/            # Database init scripts and DDLs
├── notebooks/               # Exploratory Data Analysis (EDA)
├── reports/                 # Documentation, Architecture & Dashboard exports
│   └── figures/             # System diagrams and screenshots
├── secrets/                 # Local credentials (git-ignored)
├── services/                # Core Application Services (Dockerized)
│   ├── airflow/             # Airflow custom images & dependencies
│   ├── dbt/                 # dbt runtime environment
│   ├── ingestor/            # Raw-to-Bronze ingestion logic
│   ├── streamlit/           # Frontend Analytics Dashboards
│   ├── synthetic-gen/       # Data simulator & daemon logic
│   └── trino/               # Distributed SQL engine & catalog config
├── docker-compose.yml       # Full-stack container orchestration
├── Makefile                 # Shortcuts for build, up, down, and clean
├── README.md                # Project overview and setup guide
└── .env                     # Template for environment variables
```
