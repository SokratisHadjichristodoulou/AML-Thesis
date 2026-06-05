# Explainable Anti-Money Laundering on Blockchain Networks

### A Comparative Study of Feature-Based and Graph-Based Models

**Author:** Sokratis Hadjichristodoulou  
**Programme:** BSc Data Science and Artificial Intelligence, Maastricht University  
**Supervisor:** Dr Enrique Hortal Quesada

---

## Overview

This repository contains the full implementation for a bachelor thesis investigating AML detection in blockchain systems. The study compares feature-based machine learning models and graph-based deep learning models across two cryptocurrency datasets, applies explainability methods to interpret model predictions, and proposes a wallet-level risk scoring framework for compliance decision-making.

---

## Project Structure

```
project/
├── data/
│   ├── ethereum_fraud/
│   │   └── ethereum-frauddetection-dataset/
│   │       └── transaction_dataset.csv
│   ├── elliptic_plus_plus/
│   │   ├── wallets/
│   │   │   └── wallets_features_classes_combined.csv
│   │   └── transactions/
│   │       ├── txs_features.csv
│   │       ├── txs_classes.csv
│   │       └── txs_edgelist.csv
│   └── clean_datasets/          # auto-generated on first run
├── models/                      # auto-generated experiment outputs
│   ├── rq1_feature_models/
│   ├── rq2_graph_models/
│   ├── rq3_xai_models/
│   └── rq4_risk_scoring/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_models.py
│   ├── graph_models.py
│   ├── training.py
│   ├── xai.py
│   └── risk_scoring.py
├── notebooks/
│   ├── rq1_feature_models.ipynb
│   ├── rq2_graph_models.ipynb
│   ├── rq3_xai_analysis.ipynb
│   └── rq4_wallet_risk_scoring.ipynb
└── README.md
```

---

## Datasets

### Elliptic++ Bitcoin Dataset

- **Source:** [Kaggle — Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- Over 800,000 wallet addresses, 56 wallet-level features, 1M+ temporal interactions
- Labels: licit (1), illicit (2), unknown (3)
- Used for: RQ1 (feature-based), RQ2 (graph-based), RQ3 (XAI), RQ4 (risk scoring)

### Ethereum Fraud Detection Dataset

- **Source:** [Kaggle — Ethereum Fraud Detection](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
- Account-level features including transaction counts, values, timing, and ERC20 activity
- Labels: fraudulent (1), legitimate (0)
- Used for: RQ1 (feature-based), RQ3 (XAI)

Download both datasets and place them in the `data/` directory following the structure above before running any notebooks.

---

## Installation

### Requirements

- Python 3.9+
- PyTorch (with CUDA optional)
- PyTorch Geometric

### Install dependencies

```bash
pip install -r requirements.txt
```

Key packages:

```
pandas numpy scikit-learn
xgboost lightgbm imbalanced-learn
torch torch-geometric
shap lime
matplotlib seaborn
jupyter
```

For PyTorch Geometric, follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) matching your PyTorch and CUDA version.

---

## Preprocessing

Before running any experiments, preprocess and save the clean datasets once:

```python
from src.data_loader import (
    save_preprocessed_ethereum_fraud,
    save_preprocessed_elliptic_wallets_combined,
    save_preprocessed_elliptic_transactions_combined,
    save_preprocessed_elliptic_graph,
)

save_preprocessed_ethereum_fraud()
save_preprocessed_elliptic_wallets_combined()
save_preprocessed_elliptic_transactions_combined()
save_preprocessed_elliptic_graph()
```

This only needs to be run once. Clean files are saved to `data/clean_datasets/` and reused automatically on subsequent runs.

---

## Running the Experiments

Each research question has a dedicated notebook. Run them in order.

### RQ1 — Feature-Based Models

**Notebook:** `notebooks/rq1_feature_models.ipynb`

Compares Logistic Regression, LinearSVC, Random Forest, XGBoost, and LightGBM across three datasets and three imbalance handling strategies (Simple, Balanced, SMOTE).

```python
from src.training import (
    run_random_forest_experiments,
    run_xgboost_experiments,
    run_lightgbm_experiments,
    run_logistic_regression_experiments,
    run_linear_svc_experiments,
)
```

Outputs saved to `models/rq1_feature_models/`.

---

### RQ2 — Graph-Based Models

**Notebook:** `notebooks/rq2_graph_models.ipynb`

Evaluates GCN, GraphSAGE (5 seeds, averaged), and EvolveGCN-O (seed 42) on the Elliptic++ transaction graph with temporal train-test splitting.

> **Note:** EvolveGCN-O is computationally expensive. This notebook was run on Google Colab with GPU acceleration.

```python
from src.training import (
    run_gcn_experiments,
    run_graphsage_experiments,
    run_evolve_gcn_experiments,
)
```

Outputs saved to `models/rq2_graph_models/`.

---

### RQ3 — XAI Analysis

**Notebook:** `notebooks/rq3_xai_analysis.ipynb`

Applies SHAP, local SHAP, and GNNExplainer across all datasets and models. Computes Spearman correlation between SHAP and internal feature importance.

```python
from src.xai import (
    run_shap_lightgbm,
    run_shap_random_forest,
    run_local_shap_random_forest,
    run_gnnexplainer_graphsage,
)

# Example
run_shap_lightgbm(dataset="ethereum")
run_shap_lightgbm(dataset="elliptic-wallets-combined")
run_shap_lightgbm(dataset="elliptic-transactions-combined")
```

Outputs saved to `models/rq3_xai_models/`.

---

### RQ4 — Wallet-Level Risk Scoring

**Notebook:** `notebooks/rq4_wallet_risk_scoring.ipynb`

Trains the balanced Random Forest on the Elliptic++ wallets dataset, converts predicted probabilities into 0–100 risk scores, applies LIME and counterfactual explanations, and generates a compliance output table.

```python
from src.risk_scoring import run_wallet_risk_scoring, build_compliance_output

results = run_wallet_risk_scoring(dataset="elliptic-wallets-combined")
```

**Risk categories:**

| Category | Score Range | Recommended Action                      |
| -------- | ----------- | --------------------------------------- |
| Low      | 0 – 24      | No immediate action required            |
| Medium   | 25 – 49     | Monitor wallet activity                 |
| High     | 50 – 74     | Review wallet manually                  |
| Critical | 75 – 100    | Prioritise for compliance investigation |

Outputs saved to `models/rq4_risk_scoring/`.

---

## Source Files

| File                    | Description                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| `src/data_loader.py`    | Loads, preprocesses, and splits all datasets. Handles Simple, Balanced, SMOTE, and graph splits.   |
| `src/preprocessing.py`  | Dataset-specific cleaning pipelines for Ethereum, Elliptic++ wallets, and Elliptic++ transactions. |
| `src/feature_models.py` | Model constructors for all five feature-based models with and without class balancing.             |
| `src/graph_models.py`   | PyTorch implementations of GCN, GraphSAGE, and EvolveGCN-O.                                        |
| `src/training.py`       | Experiment runners for all models. Handles evaluation, per-time-step analysis, and output saving.  |
| `src/xai.py`            | SHAP, local SHAP, LIME, GNNExplainer, and counterfactual explanation pipelines.                    |
| `src/risk_scoring.py`   | Risk score computation, category assignment, compliance output generation.                         |

---

## Reproducibility

- All feature-based experiments use `random_state=42`
- GCN and GraphSAGE are averaged across seeds 42–46 using `set_global_seed()`
- EvolveGCN-O uses seed 42 only due to computational constraints
- Temporal splits use strict time-step boundaries (train: steps 1–39, test: steps 40–49) with no data leakage
- Clean datasets are saved to disk and reloaded deterministically

---

## Results Summary

| Model               | Dataset                 | F1    | ROC-AUC   |
| ------------------- | ----------------------- | ----- | --------- |
| LightGBM (Balanced) | Ethereum                | 0.895 | 0.988     |
| LightGBM (Balanced) | Elliptic++ Transactions | 0.753 | 0.949     |
| RF (Balanced)       | Elliptic++ Wallets      | 0.775 | 0.981     |
| GraphSAGE           | Elliptic++ Transactions | ~0.50 | 0.83–0.85 |
| GCN                 | Elliptic++ Transactions | ~0.24 | 0.74–0.77 |
| EvolveGCN-O         | Elliptic++ Transactions | —     | ~0.54     |

---

## Citation

If you use this code or the thesis findings, please cite:

```
S. Hadjichristodoulou, "Explainable Anti-Money Laundering on Blockchain
Networks: A Comparative Study of Feature-Based and Graph-Based Models,"
Bachelor Thesis, Maastricht University, 2026.
```
