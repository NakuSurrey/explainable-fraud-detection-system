# Explainable Fraud Detection System

## Business Problem

**Reducing manual audit hours and false positives in financial transaction monitoring while ensuring FCA Consumer Duty compliance.**

Financial institutions spend thousands of analyst hours each year manually reviewing flagged transactions. The vast majority of these alerts are false positives — legitimate transactions incorrectly flagged by rule-based systems. This creates two critical problems: genuine fraud slips through overwhelmed teams, and operational costs balloon without proportional risk reduction.

Traditional "black box" machine learning models can improve detection accuracy, but they introduce a new problem: **auditability**. In regulated industries, a model that cannot explain *why* it flagged a transaction is legally and operationally useless. Regulators, compliance officers, and risk management teams need transparent, traceable decision-making — not opaque probability scores.

This project solves both problems simultaneously.

### The Cost of Inaction

| Problem | Business Impact |
|---------|-----------------|
| High false positive rates | Wasted analyst hours, customer friction, delayed legitimate transactions |
| Missed fraud (false negatives) | Direct financial losses, regulatory penalties, reputational damage |
| Unexplainable AI decisions | Non-compliance with FCA Consumer Duty, potential algorithmic discrimination fines |
| No feedback mechanism | Models degrade over time as fraud patterns evolve (concept drift) |

### ROI and Operational Efficiency

This system is designed to deliver measurable value:

- **Reduced manual review volume** by prioritizing high-confidence fraud alerts with explainable risk scores, allowing analysts to focus on genuinely suspicious transactions rather than chasing false positives.
- **Faster investigation cycles** through plain-English explanations of why each transaction was flagged, eliminating the guesswork that slows down manual review.
- **Regulatory compliance by design** with full auditability of every algorithmic decision, SHAP-based bias detection, and GDPR-compliant data handling.
- **Fraud ring detection** using network graph analytics to surface coordinated fraud that transaction-level models alone would miss.
- **Continuous improvement** via a human-in-the-loop feedback mechanism that captures analyst corrections and enables model retraining.

---

## Solution Overview

An enterprise-grade machine learning pipeline that:

1. **Detects** fraudulent financial transactions using XGBoost and LightGBM, optimized for heavily imbalanced data using SMOTE and class weighting.
2. **Explains** every decision using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations), providing both local (single transaction) and global (overall model) interpretability.
3. **Visualizes** results through an interactive Streamlit dashboard where non-technical stakeholders can input transactions and instantly see risk scores alongside plain-English explanations.
4. **Maps fraud rings** using NetworkX graph analytics to identify coordinated fraudulent activity across connected accounts.
5. **Learns** from human investigators through a built-in feedback loop that captures corrections for future model retraining.

### Why a White-Box Approach

This project deliberately prioritizes **transparency over complexity**. While deep learning models may offer marginal accuracy improvements on certain datasets, they come with significant drawbacks in regulated environments:

- They cannot provide feature-level explanations required by compliance teams.
- Their inference latency and computational cost are orders of magnitude higher.
- They are difficult to audit, version, and reproduce.

XGBoost and LightGBM were chosen because their inference latency is measured in **milliseconds**, making them suitable for real-time credit card processing while keeping cloud computing costs extremely low. Combined with SHAP's TreeExplainer, they provide mathematically rigorous explanations for every prediction.

---

## Tech Stack

| Component | Technology | Reasoning |
|-----------|------------|-----------|
| Primary Models | XGBoost + LightGBM | Millisecond inference, excellent on tabular data, native SHAP support |
| Explainability | SHAP (TreeExplainer) + LIME | Gold standard for local and global XAI on tree-based models |
| Imbalance Handling | SMOTE + Class Weighting | Dual approach; SMOTE applied after train/test split to prevent data leakage |
| Scaling | RobustScaler | Resistant to outliers common in financial transaction data |
| Evaluation Metric | AUPRC (primary), Precision, Recall | Correct metrics for highly imbalanced data — standard accuracy is not used |
| Graph Analytics | NetworkX + PyVis | Fraud ring detection and interactive visualization |
| Dashboard | Streamlit | Fastest Python data app framework for non-technical users |
| API | FastAPI | Modern async framework with auto-generated OpenAPI documentation |
| CI/CD | GitHub Actions | Automated testing on every push |
| Containerization | Docker | Full reproducibility across environments |
| Cloud Deployment | Microsoft Azure | Enterprise-grade hosting with free student credits |
| Configuration | Single config.yaml | Centralized configuration — no hardcoded paths anywhere |
| Logging | Python logging module | Centralized pipeline.log — no print() statements |

---

## Architecture

The system follows a **decoupled microservices architecture**. Every phase operates independently on a "Read → Process → Save Artifact" principle. If one component fails, no other component is affected, and the project never needs to be restarted.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION LAYER                          │
│                  config.yaml  ←  Single Source of Truth              │
│                  logger.py   ←  Centralized Logging                 │
└──────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  DATA LAYER   │  │  MODEL LAYER    │  │ DEPLOYMENT LAYER │
│               │  │                 │  │                  │
│ Raw Vault     │  │ XGBoost Train   │  │ FastAPI Server   │
│     ↓         │  │ LightGBM Train  │  │     ↓            │
│ Data Cleaning │  │     ↓           │  │ Streamlit UI     │
│     ↓         │  │ Model Compare   │  │     ↓            │
│ Feature Eng.  │  │     ↓           │  │ Human Feedback   │
│     ↓         │  │ Stress Testing  │  │     ↓            │
│ SMOTE/Split   │  │     ↓           │  │ Feedback DB      │
│     ↓         │  │ SHAP/LIME Gen.  │  │                  │
│ Graph (NX)    │  │                 │  │ GitHub Actions   │
└───────────────┘  └─────────────────┘  └──────────────────┘
```

### Phase Independence

Each phase reads artifacts saved by previous phases and produces its own artifacts. This means:

- If the Streamlit dashboard crashes → the trained model and API remain operational.
- If model training fails → the cleaned/engineered data is preserved and does not need to be regenerated.
- If data preprocessing has an error → the raw data vault is never overwritten; simply fix the script and re-run.
- If SHAP computation is slow → the dashboard loads pre-computed explainer objects instead of calculating on-the-fly.

---

## Project Phases

| # | Phase | Purpose | Key Artifact |
|---|-------|---------|-------------|
| 0 | Environment Setup | Project skeleton, config, logging, dependencies | config.yaml, logger.py, requirements.txt |
| 1 | Business Logic & Compliance | Documentation, regulatory alignment, architecture | README.md, compliance_notes.md |
| 2 | Data Ingestion | Download and secure raw dataset | data/raw/creditcard.csv |
| 3 | Data Engineering | Clean, engineer features, split, apply SMOTE | data/processed/*.csv, models/scaler.pkl |
| 4 | Graph Generation | Map fraud rings using NetworkX | graphs/fraud_graph.gpickle |
| 5 | Model Training | Train XGBoost + LightGBM, compare, evaluate | models/best_model.pkl, metrics.json |
| 6 | Stress Testing | Adversarial robustness validation | reports/stress_test_results.json |
| 7 | Explainability | Generate SHAP + LIME explainer objects | models/shap_explainer.pkl |
| 8 | Inference API | FastAPI microservice for real-time scoring | REST API endpoint |
| 9 | Dashboard | Streamlit UI with risk scores, SHAP plots, fraud rings | src/dashboard/app.py |
| 10 | Human-in-the-Loop & CI/CD | Feedback collection, automated testing pipeline | feedback.db, test.yml |

---

## Dataset

**Kaggle Credit Card Fraud Detection Dataset**
- 284,807 transactions over two days by European cardholders
- 492 fraudulent transactions (0.172% of total) — extreme class imbalance
- 28 PCA-transformed features (V1–V28) plus Time and Amount
- Industry-standard benchmark widely recognized in financial ML

The extreme imbalance (99.83% legitimate vs. 0.17% fraud) makes standard accuracy meaningless. A model predicting "not fraud" for every transaction would achieve 99.83% accuracy while catching zero fraud. This is why AUPRC is the primary evaluation metric.

---

## Evaluation Metrics

**Standard accuracy is explicitly not used in this project.**

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| AUPRC | Model's ability to identify fraud across all thresholds | The correct metric for heavily imbalanced data; insensitive to the large legitimate class |
| Precision | Of transactions flagged as fraud, how many actually are | High precision = fewer false positives = less wasted analyst time |
| Recall | Of all actual fraud, how much did the model catch | High recall = fewer missed fraud cases = less financial loss |
| F1-Score | Harmonic mean of Precision and Recall | Balances the trade-off between catching fraud and avoiding false alarms |

---

## Explainability Framework

### Local Interpretability (Single Transaction)

For any flagged transaction, the system generates a SHAP waterfall plot and a plain-English explanation. Example output:

> **Risk Score: 94.2% — HIGH RISK**
>
> Alert triggered because:
> - Transaction amount is 5.3x the account's historical average → +38% risk contribution
> - Transaction occurred at 3:17 AM (outside normal activity window) → +22% risk contribution
> - 4 transactions from this account in the last 10 minutes → +18% risk contribution
> - IP geolocation does not match billing address region → +12% risk contribution

### Global Interpretability (Overall Model)

SHAP summary plots show which features are most important across all predictions, enabling auditors and compliance teams to understand the model's general behavior and verify it is not relying on biased or inappropriate features.

### FCA Consumer Duty Alignment

The SHAP framework directly supports compliance with the FCA's Consumer Duty guidelines by:

- Providing mathematical proof of each feature's contribution to every decision.
- Enabling bias audits to verify the model does not discriminate against specific demographics.
- Creating an auditable trail that regulators can inspect and verify.
- Ensuring customers affected by automated decisions can receive clear explanations of why.

---

## Data Engineering Safeguards

### No Data Leakage

SMOTE is applied **only to the training set, after the train/validation/test split**. This prevents synthetic samples from leaking information about the test distribution into the training process. Feature scaling is fit on the training set and applied (transform only) to validation and test sets.

### No Hardcoded Paths

Every file path, hyperparameter, and connection string is stored in a single `config.yaml` file. Every script reads from this central configuration. Changing an output directory requires updating one line — the entire 11-phase pipeline adapts instantly.

### No Silent Failures

Python's built-in `logging` module writes to a centralized `pipeline.log`. Every phase logs its start, completion, and any errors to a `phase_status.json` tracker. No `print()` statements are used anywhere in the codebase.

---

## Regulatory Compliance

### GDPR Simulation

Although this project uses a public anonymized dataset, it simulates GDPR-compliant data handling practices:

- Raw data is stored in a separate vault directory and is never overwritten.
- PCA-transformed features ensure no personally identifiable information (PII) is present.
- Data access patterns follow the principle of least privilege.
- The feedback database stores only analyst corrections, not raw customer data.

### FCA Consumer Duty

The explainability framework ensures every algorithmic decision can be:

- **Traced** — the exact input features and their SHAP contributions are logged.
- **Justified** — plain-English explanations accompany every risk score.
- **Audited** — global feature importance plots verify the model's decision patterns.
- **Checked for bias** — SHAP values can be analyzed across demographic segments to detect discrimination.

See `docs/compliance_notes.md` for detailed compliance documentation.

---

## Future Monitoring & Data Drift

In a live banking environment, fraud patterns evolve constantly. A model trained today will degrade over time as criminals adapt their techniques. This section outlines the monitoring strategy for production deployment.

### Drift Detection

- **Population Stability Index (PSI):** Calculated weekly on input feature distributions. A PSI value above **0.2** indicates significant drift and triggers a review.
- **AUPRC Decay Monitoring:** The model's AUPRC is tracked against a holdout set refreshed monthly. A decay of more than **0.05** from the baseline triggers a manual retraining review.
- **Feature Distribution Alerts:** Z-score monitoring on key features (transaction amount, frequency, time patterns) flags unexpected shifts in incoming data.

### Retraining Protocol

1. When drift thresholds are exceeded, the human-in-the-loop feedback database is exported.
2. Corrected labels from analyst feedback are merged with the original training data.
3. The decoupled pipeline is re-run from Phase 3 (Data Engineering) through Phase 7 (Explainability).
4. The new model is compared against the previous version using AUPRC on a consistent holdout set.
5. Only if the new model meets or exceeds the minimum AUPRC threshold (0.70) is it promoted to production.

### Monitoring Dashboard Metrics (Future Enhancement)

- Weekly AUPRC trend line
- Feature drift heatmap
- False positive rate by time period
- Analyst override frequency (from feedback database)

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fraud-detection-system.git
cd fraud-detection-system

# Create and activate virtual environment
python -m venv venv

# On Windows (Git Bash):
source venv/Scripts/activate

# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup (Phase 0 tests)
python -m pytest tests/test_phase0.py -v
```

### Running the Pipeline

Each phase can be run independently. If a phase fails, fix the issue and re-run only that phase.

```bash
# Phase 2: Download and ingest data
python -m src.preprocessing.data_ingestion

# Phase 3: Clean, engineer features, split data
python -m src.preprocessing.data_engineering

# Phase 4: Generate fraud ring graphs
python -m src.graph_analytics.graph_generator

# Phase 5: Train and evaluate models
python -m src.models.train

# Phase 6: Run adversarial stress tests
python -m src.stress_testing.adversarial_test

# Phase 7: Generate SHAP and LIME explainers
python -m src.explainability.generate_explainers

# Phase 8: Start the inference API
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Phase 9: Launch the dashboard
streamlit run src/dashboard/app.py

# Run all tests
python -m pytest tests/ -v
```

---

## Repository Structure

```
fraud-detection-system/
├── .github/workflows/          # CI/CD: GitHub Actions auto-testing
│   └── test.yml
├── config.yaml                 # Central configuration (all phases read from here)
├── Dockerfile                  # Containerized reproducibility
├── requirements.txt            # Pinned dependencies
├── .gitignore                  # Keeps data/models/logs out of version control
├── README.md                   # This file
│
├── data/
│   ├── raw/                    # Phase 2: Untouched dataset vault (never overwritten)
│   ├── processed/              # Phase 3: Cleaned, split, engineered data
│   └── feedback/               # Phase 10: Human correction database
│
├── models/                     # Phase 5/7: Trained models, scalers, explainers
├── logs/                       # Centralized pipeline.log + phase_status.json
├── graphs/                     # Phase 4: NetworkX fraud ring data
├── reports/                    # Phase 6: Stress test results
├── tests/                      # Test files organized by phase
│
├── docs/
│   ├── compliance_notes.md     # FCA Consumer Duty & GDPR documentation
│   └── architecture_diagram.png
│
├── src/
│   ├── utils/
│   │   └── logger.py           # Centralized logger, config loader, phase tracker
│   ├── preprocessing/          # Phase 3: Data cleaning and feature engineering
│   ├── features/               # Phase 3: Temporal feature engineering
│   ├── models/                 # Phase 5: Model training and evaluation
│   ├── explainability/         # Phase 7: SHAP and LIME generation
│   ├── api/                    # Phase 8: FastAPI inference server
│   ├── dashboard/              # Phase 9: Streamlit UI
│   ├── graph_analytics/        # Phase 4: NetworkX fraud ring mapping
│   └── stress_testing/         # Phase 6: Adversarial robustness tests
│
└── notebooks/                  # Optional: Exploration and prototyping
```

---

## License

This project is developed for educational and portfolio purposes. The dataset used is publicly available under the Open Database License (ODbL) from Kaggle.

---

## Author

Nakul Arora

*Engineered a machine learning fraud detection pipeline using XGBoost and Python, applying SMOTE to handle heavily imbalanced financial data and optimizing for AUPRC to minimize false positives. Integrated SHAP and LIME frameworks to provide local and global interpretability, transforming opaque algorithmic decisions into transparent, auditable reports suitable for compliance and risk management teams.*
