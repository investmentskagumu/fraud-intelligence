# Fraud Intelligence: Financial Transaction Analysis and Risk Detection

## 1. Project Overview

Fraud Intelligence is an end-to-end financial transaction analysis and fraud detection project built with Python and SQL. It explores transaction behaviour, identifies potential fraud patterns, develops machine learning models, evaluates decision thresholds, and produces risk analysis outputs and an interactive HTML dashboard.

The project uses a synthetic financial transaction dataset. Its findings demonstrate analytical and modelling techniques and should not be interpreted as results from a live financial institution.

## 2. Objectives

- Audit and validate transaction data.
- Explore fraud prevalence and transaction patterns.
- Analyse temporal behaviour, accounts, and transaction networks.
- Engineer predictive features and assess potential data leakage.
- Train and compare baseline and advanced machine learning models.
- Evaluate model performance and fraud detection thresholds.
- Analyse false-positive and false-negative decision costs.
- Generate risk scores and prioritise transactions for investigation.
- Evaluate temporal performance and model robustness.
- Present findings through tables, figures, metrics, and an HTML dashboard.

## 3. Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- SciPy
- Matplotlib
- XGBoost
- LightGBM
- Plotly
- SQL

## 4. Project Workflow

The analysis is organised into 27 numbered stages.

| Stage | Focus |
|---|---|
| 01–04 | Data loading, audit, quality checks, and cleaning |
| 05–10 | Target analysis, exploratory analysis, fraud patterns, temporal behaviour, account analysis, and network analysis |
| 11–14 | Feature engineering, leakage audit, data splitting, and statistical analysis |
| 15–18 | Baseline models, advanced models, class imbalance, and model comparison |
| 19–21 | Threshold analysis, decision costs, and model interpretation |
| 22–25 | Risk scoring, investigation prioritisation, temporal validation, and stress testing |
| 26–27 | Final model selection analysis and dashboard generation |

The numbered scripts are located in `scripts/`.

## 5. Repository Structure

```text
FRAUD_INTELLIGENCE_001/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── splits/
├── dashboard/
├── models/
├── notebooks/
├── outputs/
│   ├── dashboard/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   └── tables/
├── reports/
├── scripts/
├── sql/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

The `outputs/` directory contains generated analysis artefacts. The interactive dashboard is located at:

`outputs/dashboard/fraud_intelligence_dashboard.html`

The `models/`, `notebooks/`, `reports/`, `dashboard/`, and `tests/` directories are available for future organised development. Their presence does not imply that they currently contain files.

## 6. Dataset

The project uses `Synthetic_Financial_datasets_log.csv`, a synthetic financial transaction dataset.

The dataset is intended for learning, experimentation, and analytical demonstration. It does not represent verified real-world customer transactions.

## 7. Installation

Use Python and PowerShell on Windows.

Clone or download the repository, then navigate to the project directory.

Create and activate a virtual environment if one is not already configured:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the listed dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 8. Running the Analysis

Run scripts from the project root, following their numbered order where a stage depends on earlier outputs.

For example:

```powershell
python .\scripts\01_load_data.py
python .\scripts\02_data_audit.py
```

**Important:** Before rerunning the full pipeline, inspect each script's input and output paths. Some stages may overwrite existing generated files or depend on outputs produced by earlier stages. The existing project outputs should be preserved.

## 9. Dashboard

Open the generated dashboard in a web browser:

`outputs/dashboard/fraud_intelligence_dashboard.html`

The dashboard presents sections for executive summary, fraud analysis, model performance, and risk intelligence.

## 10. Model Evaluation

The project evaluates fraud detection models using classification metrics, threshold analysis, and decision-cost analysis. Relevant outputs include precision, recall, F1-score, specificity, ROC-AUC, and precision-recall AUC where calculated by the relevant stage.

Fraud detection involves an important trade-off: increasing fraud recall may also increase false positives and the number of transactions requiring investigation.

Model performance should therefore be assessed against the intended operational costs and investigation capacity rather than a single metric.

## 11. Limitations

- The dataset is synthetic and may not reflect real-world fraud behaviour.
- Results depend on the dataset, feature definitions, train/test split, and evaluation method.
- Fraud is rare relative to legitimate activity, making class imbalance important.
- Thresholds and model rankings may not transfer across models or datasets.
- The project is an analytical prototype, not a production fraud detection service.
- Production use would require additional validation, monitoring, security controls, deployment engineering, and assessment on suitable real-world data.

## 12. Future Improvements

- Add automated tests for data processing and evaluation.
- Document reproducible experiment configurations.
- Save and version validated model artefacts.
- Add model-specific threshold validation.
- Improve experiment tracking and reproducibility.
- Develop deployment and monitoring workflows.
- Add documented SQL examples and technical reports.

## 13. Author

**Erick Kagumu**

Statistics and Programming | Data Analytics | Machine Learning

This project demonstrates practical work in data preparation, exploratory analysis, statistical investigation, machine learning evaluation, fraud-risk analysis, and dashboard reporting.
