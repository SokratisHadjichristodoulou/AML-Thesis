import pandas as pd

from src.data_loader import (
    # Ethereum
    load_ethereum_simple,
    load_ethereum_smote,
    load_ethereum_simple_scaled,
    load_ethereum_smote_scaled,

    # Elliptic-Wallets
    load_wallet_combined_simple,
    load_wallet_combined_smote,
    load_wallet_combined_simple_scaled,
    load_wallet_combined_smote_scaled,

    # Elliptic-Transactions
    load_transactions_combined_simple,
    load_transactions_combined_smote,
    load_transactions_combined_simple_scaled,
    load_transactions_combined_smote_scaled,

    # Elliptic-Transactons- Graph
    load_transactions_graph_simple
)

from src.feature_models import (
    get_random_forest,
    get_random_forest_balanced,
    get_xgboost,
    get_xgboost_balanced,
    get_lightgbm, 
    get_lightgbm_balanced,
    get_logistic_regression,
    get_logistic_regression_balanced,
    get_svm,
    get_svm_balanced,
    get_linear_svc,
    get_linear_svc_balanced,
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------------------------------
# DATASET LOADER: Not case sensitive
# ---------------------------------------------------
DATASET_LOADERS = {
    "ethereum": {
        "simple": load_ethereum_simple,
        "smote": load_ethereum_smote,
        "scaled": load_ethereum_simple_scaled,
        "scaled_smote": load_ethereum_smote_scaled,
    },
    "elliptic-wallets-combined": {
        "simple": load_wallet_combined_simple,
        "smote": load_wallet_combined_smote,
        "scaled": load_wallet_combined_simple_scaled,
        "scaled_smote": load_wallet_combined_smote_scaled,
    },
    "elliptic-transactions-combined": {
        "simple": load_transactions_combined_simple,
        "smote": load_transactions_combined_smote,
        "scaled": load_transactions_combined_simple_scaled,
        "scaled_smote": load_transactions_combined_smote_scaled,
    },
    "elliptic-transactions-graph": {
        "simple": load_transactions_graph_simple,
    },
}

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def _evaluate(y_test, y_pred, y_proba=None):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }

def _evaluate_per_timestep(time_steps, y_test, y_pred, y_proba=None):
    """
    Compute per-timestep classification metrics on the test set.

    Parameters:
        time_steps: pd.Series or array of time-step values aligned with y_test
        y_test, y_pred: aligned arrays/Series of labels
        y_proba: aligned array of positive-class probabilities (optional)

    Returns:
        DataFrame with one row per time step:
        time_step, n_samples, n_positives, precision, recall, f1, roc_auc
    """
    df = pd.DataFrame({
        "time_step": pd.Series(time_steps).values,
        "y_true": pd.Series(y_test).values,
        "y_pred": pd.Series(y_pred).values,
    })
    if y_proba is not None:
        df["y_proba"] = pd.Series(y_proba).values

    rows = []
    for ts, group in df.groupby("time_step"):
        n_pos = int((group["y_true"] == 1).sum())
        row = {
            "time_step": int(ts),
            "n_samples": len(group),
            "n_positives": n_pos,
            "precision": precision_score(group["y_true"], group["y_pred"], zero_division=0),
            "recall": recall_score(group["y_true"], group["y_pred"], zero_division=0),
            "f1": f1_score(group["y_true"], group["y_pred"], zero_division=0),
        }
        if y_proba is not None and 0 < n_pos < len(group):
            row["roc_auc"] = roc_auc_score(group["y_true"], group["y_proba"])
        else:
            row["roc_auc"] = None
        rows.append(row)

    return pd.DataFrame(rows).sort_values("time_step").reset_index(drop=True)

def get_dataset_loader(dataset_name, setup):
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if setup not in DATASET_LOADERS[dataset_name]:
        raise ValueError(f"Unknown setup: {setup}")

    return DATASET_LOADERS[dataset_name][setup]

# ---------------------------------------------------
# Run all 3 setups for Random Forest
# ---------------------------------------------------
def run_random_forest_experiments(dataset="ethereum"):
    results = []

    # ---------------------------
    # 1. Simple (baseline)
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test = loader()

    model = get_random_forest()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "Random Forest",
        "setup": "simple",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 2. Simple + Balanced RF
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test = loader()

    model = get_random_forest_balanced()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "Random Forest",
        "setup": "simple_balanced",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 3. SMOTE
    # ---------------------------
    loader = get_dataset_loader(dataset, "smote")
    X_train, X_test, y_train, y_test = loader()

    model = get_random_forest()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "Random Forest",
        "setup": "smote",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

# ---------------------------------------------------
# Run all 3 setups for XGBoost
# ---------------------------------------------------
def run_xgboost_experiments(dataset="ethereum"):
    results = []

    # ---------------------------
    # 1. Simple
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test = loader()

    model = get_xgboost()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "XGBoost",
        "setup": "simple",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 2. Simple + Balanced
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test = loader()

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = get_xgboost_balanced(scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "XGBoost",
        "setup": "simple_balanced",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 3. SMOTE
    # ---------------------------
    loader = get_dataset_loader(dataset, "smote")
    X_train, X_test, y_train, y_test = loader()

    model = get_xgboost()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "XGBoost",
        "setup": "smote",
        **_evaluate(y_test, y_pred, y_proba)
    })

    return pd.DataFrame(results)

# ---------------------------------------------------
# Run all 3 setups for LightGBM
# ---------------------------------------------------
def run_lightgbm_experiments(dataset="ethereum"):
    results = []

    # ---------------------------
    # 1. Simple (baseline)
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test = loader()

    model = get_lightgbm()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "LightGBM",
        "setup": "simple",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 2. Simple + Balanced LightGBM
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test = loader()

    model = get_lightgbm_balanced()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "LightGBM",
        "setup": "simple_balanced",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 3. SMOTE
    # ---------------------------
    loader = get_dataset_loader(dataset, "smote")
    X_train, X_test, y_train, y_test = loader()

    model = get_lightgbm()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "LightGBM",
        "setup": "smote",
        **_evaluate(y_test, y_pred, y_proba)
    })

    return pd.DataFrame(results)

# ---------------------------------------------------
# Run Logistic Regression (scaled only)
# ---------------------------------------------------
def run_logistic_regression_experiments(dataset="ethereum"):
    results = []

    # ---------------------------
    # 1. Scaling (baseline)
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_logistic_regression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "Logistic Regression",
        "setup": "scaled",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 2. Scaling + Balnced Logistic Regresion
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_logistic_regression_balanced()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "Logistic Regression",
        "setup": "scaled_balanced",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 3. Scaling + SMOTE
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled_smote")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_logistic_regression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "Logistic Regression",
        "setup": "scaled_smote",
        **_evaluate(y_test, y_pred, y_proba)
    })

    return pd.DataFrame(results)

# ---------------------------------------------------
# Run SVM (scaled only)
# ---------------------------------------------------
def run_svm_experiments(dataset="ethereum"):
    results = []

    # ---------------------------
    # 1. Scaling (baseline)
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_svm()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "SVM",
        "setup": "scaled",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 2. Scaling + Balanced SVM
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_svm_balanced()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "SVM",
        "setup": "scaled_balanced",
        **_evaluate(y_test, y_pred, y_proba)
    })

    # ---------------------------
    # 2. Scaling + SMOTE
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled_smote")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_svm()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "SVM",
        "setup": "scaled_smote",
        **_evaluate(y_test, y_pred, y_proba)
    })

    return pd.DataFrame(results)

# ---------------------------------------------------
# Run LinearSVC (scaled only)
# ---------------------------------------------------
def run_linear_svc_experiments(dataset="ethereum"):
    results = []

    # ---------------------------
    # 1. Scaling (baseline)
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_linear_svc()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        "dataset": dataset.lower(),
        "model": "Linear SVC",
        "setup": "scaled",
        **_evaluate(y_test, y_pred)
    })

    # ---------------------------
    # 2. Scaling + Balanced
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_linear_svc_balanced()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        "dataset": dataset.lower(),
        "model": "Linear SVC",
        "setup": "scaled_balanced",
        **_evaluate(y_test, y_pred)
    })

    # ---------------------------
    # 3. Scaling + SMOTE
    # ---------------------------
    loader = get_dataset_loader(dataset, "scaled_smote")
    X_train, X_test, y_train, y_test, _ = loader()

    model = get_linear_svc()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        "dataset": dataset.lower(),
        "model": "Linear SVC",
        "setup": "scaled_smote",
        **_evaluate(y_test, y_pred)
    })

    return pd.DataFrame(results)

# ---------------------------------------------------
# GRAPH EXPERIMENTS (static feature baseline for RQ2)
# Only "simple" and "simple_balanced" setups.
# Dataset: elliptic-transactions-graph (time-based split, no SMOTE, no scaling)
# ---------------------------------------------------

# ---------------------------------------------------
# Run Random Forest on graph dataset (simple + simple_balanced)
# ---------------------------------------------------
def run_random_forest_graph_experiments(dataset="elliptic-transactions-graph"):
    results = []
    per_timestep = {}

    # ---------------------------
    # 1. Simple (baseline)
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test, train_time, test_time = loader()

    model = get_random_forest()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "Random Forest",
        "setup": "simple",
        **_evaluate(y_test, y_pred, y_proba)
    })
    per_timestep["simple"] = _evaluate_per_timestep(test_time, y_test, y_pred, y_proba)

    # ---------------------------
    # 2. Simple + Balanced RF
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test, train_time, test_time = loader()

    model = get_random_forest_balanced()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "Random Forest",
        "setup": "simple_balanced",
        **_evaluate(y_test, y_pred, y_proba)
    })
    per_timestep["simple_balanced"] = _evaluate_per_timestep(test_time, y_test, y_pred, y_proba)

    return pd.DataFrame(results), per_timestep

# ---------------------------------------------------
# Run XGBoost on graph dataset (simple + simple_balanced)
# ---------------------------------------------------
def run_xgboost_graph_experiments(dataset="elliptic-transactions-graph"):
    results = []
    per_timestep = {}

    # ---------------------------
    # 1. Simple (baseline)
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test, train_time, test_time = loader()

    model = get_xgboost()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "XGBoost",
        "setup": "simple",
        **_evaluate(y_test, y_pred, y_proba)
    })
    per_timestep["simple"] = _evaluate_per_timestep(test_time, y_test, y_pred, y_proba)

    # ---------------------------
    # 2. Simple + Balanced XGBoost
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test, train_time, test_time = loader()

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = get_xgboost_balanced(scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "XGBoost",
        "setup": "simple_balanced",
        **_evaluate(y_test, y_pred, y_proba)
    })
    per_timestep["simple_balanced"] = _evaluate_per_timestep(test_time, y_test, y_pred, y_proba)

    return pd.DataFrame(results), per_timestep

# ---------------------------------------------------
# Run LightGBM on graph dataset (simple + simple_balanced)
# ---------------------------------------------------
def run_lightgbm_graph_experiments(dataset="elliptic-transactions-graph"):
    results = []
    per_timestep = {}

    # ---------------------------
    # 1. Simple (baseline)
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test, train_time, test_time = loader()

    model = get_lightgbm()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "LightGBM",
        "setup": "simple",
        **_evaluate(y_test, y_pred, y_proba)
    })
    per_timestep["simple"] = _evaluate_per_timestep(test_time, y_test, y_pred, y_proba)


    # ---------------------------
    # 2. Simple + Balanced LightGBM
    # ---------------------------
    loader = get_dataset_loader(dataset, "simple")
    X_train, X_test, y_train, y_test, train_time, test_time = loader()

    model = get_lightgbm_balanced()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "dataset": dataset.lower(),
        "model": "LightGBM",
        "setup": "simple_balanced",
        **_evaluate(y_test, y_pred, y_proba)
    })
    per_timestep["simple_balanced"] = _evaluate_per_timestep(test_time, y_test, y_pred, y_proba)

    
    return pd.DataFrame(results), per_timestep