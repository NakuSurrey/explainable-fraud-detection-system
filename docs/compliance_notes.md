# Compliance & Regulatory Alignment Documentation

## Document Purpose

This document details how the Explainable Fraud Detection System aligns with key regulatory frameworks governing algorithmic decision-making in financial services. It is intended for compliance officers, risk management teams, and auditors who need to verify that the system meets regulatory obligations before deployment.

---

## 1. FCA Consumer Duty Alignment

### 1.1 Overview

The Financial Conduct Authority's Consumer Duty (PS22/9, effective July 2023) requires firms to deliver good outcomes for retail customers. When automated systems influence decisions that affect customers — such as flagging transactions as potentially fraudulent, which may lead to account freezes, delayed payments, or customer friction — firms must ensure those systems act fairly, transparently, and without bias.

This system is designed to satisfy three core Consumer Duty requirements relevant to algorithmic decision-making:

### 1.2 Requirement: Fair Treatment and Non-Discrimination

**Regulatory Expectation:** Automated systems must not produce outcomes that systematically disadvantage specific groups of customers based on protected characteristics (age, gender, ethnicity, disability, etc.).

**How This System Complies:**

SHAP (SHapley Additive exPlanations) values provide a mathematically rigorous breakdown of how each input feature contributes to every individual prediction. This enables:

- **Feature Contribution Auditing:** For every flagged transaction, the exact numerical contribution of each feature is recorded. If a feature correlated with a protected characteristic (e.g., geographic location as a proxy for ethnicity) has an outsized influence on fraud scores, it will be visible in the SHAP summary plots.

- **Demographic Segment Analysis:** SHAP values can be aggregated across demographic segments to detect systematic bias. If the model assigns higher risk scores to transactions from specific geographic regions or time zones without legitimate fraud-pattern justification, the SHAP analysis will expose this.

- **White-Box Model Selection:** XGBoost and LightGBM are inherently interpretable tree-based models. Unlike deep neural networks, their decision boundaries can be traced through individual trees, providing an additional layer of auditability beyond SHAP.

**Audit Procedure:**

1. Generate SHAP summary plots grouped by available demographic indicators.
2. Compare mean SHAP contributions across groups for each feature.
3. Flag any feature where the contribution difference between groups exceeds a predefined threshold.
4. Document findings and remediation actions (e.g., feature removal, re-weighting).

### 1.3 Requirement: Transparency and Explainability

**Regulatory Expectation:** Customers affected by automated decisions have the right to understand why a decision was made. Firms must be able to provide clear, understandable explanations.

**How This System Complies:**

- **Plain-English Explanations:** The Streamlit dashboard translates SHAP waterfall plots into human-readable statements. Instead of showing raw numerical values, it presents explanations such as: "This transaction was flagged because the amount was 5 times higher than your usual spending, and it occurred at an unusual time of day."

- **Local Interpretability (SHAP Waterfall):** For each individual transaction, the system produces a chart showing exactly which features pushed the risk score up or down, and by how much. This directly answers the question "Why was my transaction blocked?"

- **Global Interpretability (SHAP Summary):** For regulators and auditors, the system provides an overview of which features the model relies on most heavily across all predictions. This answers the question "How does the model generally make decisions?"

- **LIME Cross-Validation:** LIME provides an independent, model-agnostic explanation for each prediction by fitting a local linear approximation. Comparing LIME and SHAP explanations for the same transaction serves as a consistency check on the model's reasoning.

### 1.4 Requirement: Avoiding Foreseeable Harm

**Regulatory Expectation:** Firms must take reasonable steps to avoid causing foreseeable harm to customers through their products and services.

**How This System Complies:**

- **False Positive Mitigation:** The model is evaluated using AUPRC (Area Under the Precision-Recall Curve), which explicitly balances the trade-off between catching fraud and minimizing false positives. This protects legitimate customers from unnecessary friction (blocked cards, delayed payments).

- **Human-in-the-Loop Override:** The system includes a mechanism for human investigators to review and override the model's decisions. No transaction is automatically blocked based solely on the model's output without human review for high-stakes actions.

- **Feedback Integration:** Analyst corrections (marking false positives and confirming true fraud) are stored in a feedback database, enabling the model to be retrained with corrected labels. This reduces the likelihood of repeated errors affecting the same customers.

- **Stress Testing:** Adversarial tests verify that the model maintains acceptable performance when fraud patterns shift, reducing the risk of a degraded model causing harm through inaccurate predictions.

---

## 2. GDPR Compliance Simulation

### 2.1 Overview

The General Data Protection Regulation (EU 2016/679) governs the processing of personal data. While this project uses a publicly available, anonymized dataset (PCA-transformed features with no personally identifiable information), it simulates GDPR-compliant practices to demonstrate enterprise readiness.

### 2.2 Article 22: Automated Individual Decision-Making

**Regulatory Expectation:** Individuals have the right not to be subject to decisions based solely on automated processing that significantly affect them, and to obtain meaningful information about the logic involved.

**How This System Addresses It:**

- The SHAP and LIME explainability framework provides the "meaningful information about the logic involved" required by Article 22.
- The human-in-the-loop mechanism ensures no solely automated decision-making for consequential actions.
- Every prediction is logged with its feature contributions, creating an audit trail that supports a data subject's right to explanation.

### 2.3 Data Minimization (Article 5(1)(c))

**Simulation Approach:**

- The dataset uses PCA-transformed features, meaning the raw personal data (names, card numbers, addresses) has already been reduced to mathematical components. This aligns with the principle of processing only the minimum data necessary.
- The system does not store or process any data beyond what is required for fraud scoring and explanation.

### 2.4 Data Security and Integrity (Article 5(1)(f))

**Simulation Approach:**

- Raw data is stored in a separate vault directory (`data/raw/`) and is never overwritten by downstream processing.
- The feedback database stores only analyst corrections (transaction ID, corrected label, timestamp), not customer personal data.
- The Docker containerization ensures a consistent, isolated execution environment.
- The `config.yaml` centralizes access paths, reducing the risk of accidental data exposure through scattered hardcoded paths.

### 2.5 Purpose Limitation (Article 5(1)(b))

**Simulation Approach:**

- Data is ingested solely for fraud detection purposes.
- No secondary use, sharing, or transfer of data is implemented or facilitated.
- The system architecture does not include data export capabilities to external parties.

---

## 3. Algorithmic Auditability

### 3.1 Audit Trail

Every component of the decision-making process is traceable:

| Component | What Is Logged | Where |
|-----------|---------------|-------|
| Data Ingestion | Dataset source, row counts, timestamp | pipeline.log |
| Feature Engineering | Transformations applied, scaling parameters | pipeline.log, models/scaler.pkl |
| Model Training | Hyperparameters, training duration, convergence metrics | pipeline.log, config.yaml |
| Model Evaluation | AUPRC, Precision, Recall for both XGBoost and LightGBM | models/metrics.json |
| Predictions | Risk score, SHAP values for each feature | API response JSON |
| Human Feedback | Analyst ID, correction type, timestamp | data/feedback/feedback.db |
| Stress Testing | Test scenarios, results, pass/fail status | reports/stress_test_results.json |

### 3.2 Reproducibility

- All dependencies are pinned to exact versions in `requirements.txt`.
- The Docker container ensures identical execution environments across machines.
- The random seed (42) is set in `config.yaml` and used across all stochastic processes (train/test splits, SMOTE, model training).
- Every file path is relative and resolved through `config.yaml`, ensuring portability.

### 3.3 Model Versioning

- Trained models are saved with version identifiers (e.g., `xgboost_fraud_v1.pkl`).
- The `models/metrics.json` file tracks evaluation scores for each model version.
- The `models/model_comparison.json` file documents head-to-head performance between XGBoost and LightGBM.

---

## 4. Bias Detection Methodology

### 4.1 Pre-Deployment Bias Checks

Before the model is deployed, the following checks are performed:

1. **Feature Correlation Analysis:** Identify any input features that may serve as proxies for protected characteristics. In the Kaggle dataset, PCA features are mathematically decorrelated from the original attributes, but in a production system with raw features, this step is critical.

2. **Disparate Impact Analysis:** Compare model outcomes (fraud flag rate) across available demographic segments. A ratio below 0.8 between any two groups (the four-fifths rule) triggers a review.

3. **SHAP Fairness Audit:** Aggregate SHAP values by demographic segment. If a feature contributes disproportionately to fraud scores for a specific group, investigate whether the pattern reflects genuine fraud behavior or bias.

### 4.2 Post-Deployment Monitoring

- Track false positive rates by demographic segment over time.
- Monitor analyst override rates by segment (high override rates for a specific group may indicate model bias).
- Re-run the SHAP fairness audit after each model retraining cycle.

---

## 5. Document Control

| Field | Value |
|-------|-------|
| Document Owner | YOUR_NAME_HERE |
| Version | 1.0 |
| Created | Phase 1 |
| Review Frequency | After each model retraining cycle |
| Classification | Internal — Compliance |

---

*This document is part of the Explainable Fraud Detection System project and should be reviewed alongside the system's README.md and technical documentation.*
