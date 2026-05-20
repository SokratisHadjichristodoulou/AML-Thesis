import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import random

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

    # Elliptic-Transactons-Graph
    load_transactions_graph_baseline_simple, 
    load_transactions_graph_gcn,
    load_transactions_graph_temporal,
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

from src.graph_models import (
    get_gcn,
    get_graphsage,
    get_evolve_gcn,
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
        "simple": load_transactions_graph_baseline_simple,
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

def set_global_seed(seed=42, deterministic=True):
    """
    Make training as reproducible as possible across runs.

    Parameters:
        seed: integer random seed
        deterministic: if True, ask PyTorch to use deterministic ops when possible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Optional: stricter determinism in newer PyTorch versions
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def _train_gcn(
    data, 
    hidden_channels=64, 
    dropout=0.5, 
    lr=0.01, 
    weight_decay=5e-4, 
    epochs=200, 
    class_balanced=False, 
    device="cpu", 
    verbose=False,
    seed=42, 
    deterministic=True
):
    """
    Standard full-batch GCN training loop with masking.
    Returns the trained model and final logits on the full graph.
    """
    set_global_seed(seed, deterministic=deterministic)

    model = get_gcn(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Compute class weights on training labels only (matches our balanced baselines)
    class_weights = None
    if class_balanced:
        y_train = data.y[data.train_mask]
        n_pos = (y_train == 1).sum().item()
        n_neg = (y_train == 0).sum().item()
        pos_weight = min(n_neg / max(n_pos, 1), 3.0)  # cap at 3x
        w = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        class_weights = w

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return model, logits

def _train_graphsage(
    data,
    hidden_channels=64,
    dropout=0.5, 
    lr=0.01, 
    weight_decay=5e-4, 
    epochs=200,
    class_balanced=False, 
    pos_weight_cap=3.0,
    device="cpu", 
    verbose=False, 
    seed=42, 
    deterministic=True
):
    """
    Standard full-batch GraphSAGE training loop with masking.
    Same skeleton as _train_gcn — only the model class differs.
    """
    set_global_seed(seed, deterministic=deterministic)

    model = get_graphsage(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    class_weights = None
    if class_balanced:
        y_train = data.y[data.train_mask]
        n_pos = (y_train == 1).sum().item()
        n_neg = (y_train == 0).sum().item()
        pos_weight = min(n_neg / max(n_pos, 1), pos_weight_cap)
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return model, logits

def _train_evolve_gcn(
    bundle, 
    hidden_channels=64, 
    dropout=0.5, lr=0.01, 
    weight_decay=5e-4, 
    epochs=50, 
    class_balanced=False, 
    pos_weight_cap=3.0, 
    device="cpu", 
    verbose=False, 
    seed=42, 
    deterministic=True
):
    """
    Train EvolveGCN-O on a sequence of graph snapshots.

    Loss is accumulated across all training time steps per epoch.
    """
    set_global_seed(seed, deterministic=deterministic)

    snapshots = [s.to(device) for s in bundle["snapshots"]]
    train_steps = set(bundle["train_time_steps"])

    model = get_evolve_gcn(
        in_channels=bundle["n_features"],
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)

    # Class weights from TRAIN snapshots only
    if class_balanced:
        all_train_y = torch.cat([
            s.y[s.supervision_mask] for s in snapshots if s.time_step in train_steps
        ])
        n_pos = (all_train_y == 1).sum().item()
        n_neg = (all_train_y == 0).sum().item()
        pos_weight = min(n_neg / max(n_pos, 1), pos_weight_cap)
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    else:
        class_weights = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        model.reset_weights(device=device)
        optimizer.zero_grad()

        total_loss = 0.0
        n_train_steps = 0

        for snap in snapshots:
            logits = model.forward_step(snap.x, snap.edge_index)

            if snap.time_step in train_steps:
                mask = snap.supervision_mask
                if mask.sum() > 0:
                    loss = F.cross_entropy(
                        logits[mask],
                        snap.y[mask],
                        weight=class_weights,
                    )
                    total_loss = total_loss + loss
                    n_train_steps += 1

        if n_train_steps > 0:
            total_loss = total_loss / n_train_steps
            total_loss.backward()
            optimizer.step()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | avg loss {total_loss.item():.4f}")

    model.eval()
    model.reset_weights(device=device)

    per_step_outputs = {}
    with torch.no_grad():
        for snap in snapshots:
            logits = model.forward_step(snap.x, snap.edge_index)
            per_step_outputs[snap.time_step] = {
                "logits": logits.cpu(),
                "y": snap.y.cpu(),
                "mask": snap.supervision_mask.cpu(),
            }

    return model, per_step_outputs

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

# ---------------------------------------------------
# GRAPH EXPERIMENTS (static GNN for RQ2)
# Only "simple" and "simple_balanced" setups.
# Dataset: elliptic-transactions-graph (time-based split, no SMOTE, no scaling)
# ---------------------------------------------------

# ---------------------------------------------------
# Run Simple 2 Layer GCN on graph dataset (simple + simple_balanced)
# ---------------------------------------------------
def run_gcn_experiments(
    dataset="elliptic-transactions-graph", 
    device="cpu", 
    seeds=[42, 43, 44, 45, 46], 
    deterministic=True
):
    all_runs = []
    per_timestep_all = []

    for seed in seeds:
        results = []
        per_timestep = {}

        # ---------------------------
        # 1. Simple (baseline GCN)
        # ---------------------------
        data = load_transactions_graph_gcn()
        model, logits = _train_gcn(
            data,
            class_balanced=False,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )

        y_proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred  = logits.argmax(dim=1).cpu().numpy()
        y_true  = data.y.cpu().numpy()
        mask    = data.test_mask.cpu().numpy()
        t_all   = data.time_steps.cpu().numpy()

        results.append({
            "dataset": dataset.lower(),
            "model": "GCN",
            "setup": "simple",
            **_evaluate(y_true[mask], y_pred[mask], y_proba[mask]),
        })
        per_timestep["simple"] = _evaluate_per_timestep(
            t_all[mask], y_true[mask], y_pred[mask], y_proba[mask]
        )

        # ---------------------------
        # 2. Simple + Balanced GCN
        # ---------------------------
        data = load_transactions_graph_gcn()
        model, logits = _train_gcn(
            data,
            class_balanced=True,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )

        y_proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred  = logits.argmax(dim=1).cpu().numpy()
        y_true  = data.y.cpu().numpy()
        mask    = data.test_mask.cpu().numpy()
        t_all   = data.time_steps.cpu().numpy()

        results.append({
            "dataset": dataset.lower(),
            "model": "GCN",
            "setup": "simple_balanced",
            **_evaluate(y_true[mask], y_pred[mask], y_proba[mask]),
        })
        per_timestep["simple_balanced"] = _evaluate_per_timestep(
            t_all[mask], y_true[mask], y_pred[mask], y_proba[mask]
        )

        # Save aggregate results for this seed
        df_seed = pd.DataFrame(results)
        df_seed["seed"] = seed
        all_runs.append(df_seed)

        # Save per-time-step results for this seed
        for setup_name, df_ts in per_timestep.items():
            df_ts = df_ts.copy()
            df_ts["model"] = "GCN"
            df_ts["setup"] = setup_name
            df_ts["seed"] = seed
            per_timestep_all.append(df_ts)

    # Combine all aggregate results
    df_all = pd.concat(all_runs, ignore_index=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    df_mean = df_all.groupby(["dataset", "model", "setup"])[metrics].mean()
    df_std  = df_all.groupby(["dataset", "model", "setup"])[metrics].std()

    df_summary = df_mean.copy()
    for m in metrics:
        df_summary[m] = (
            df_mean[m].round(4).astype(str) + " ± " + df_std[m].round(4).astype(str)
        )

    df_summary = df_summary.reset_index()

    # Combine all per-time-step results
    df_per_timestep = pd.concat(per_timestep_all, ignore_index=True)

    return df_summary, df_per_timestep

# ---------------------------------------------------
# Run Simple 2 Layer graphSAGE on graph dataset (simple + simple_balanced)
# ---------------------------------------------------
def run_graphsage_experiments(
    dataset="elliptic-transactions-graph", 
    device="cpu", 
    seeds=[42, 43, 44, 45, 46], 
    deterministic=True
):
    all_runs = []
    per_timestep_all = []

    for seed in seeds:
        results = []
        per_timestep = {}

        # ---------------------------
        # 1. Simple (baseline GraphSAGE)
        # ---------------------------
        data = load_transactions_graph_gcn()
        model, logits = _train_graphsage(
            data,
            class_balanced=False,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )

        y_proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred  = logits.argmax(dim=1).cpu().numpy()
        y_true  = data.y.cpu().numpy()
        mask    = data.test_mask.cpu().numpy()
        t_all   = data.time_steps.cpu().numpy()

        results.append({
            "dataset": dataset.lower(),
            "model": "GraphSAGE",
            "setup": "simple",
            **_evaluate(y_true[mask], y_pred[mask], y_proba[mask]),
        })
        per_timestep["simple"] = _evaluate_per_timestep(
            t_all[mask], y_true[mask], y_pred[mask], y_proba[mask]
        )

        # ---------------------------
        # 2. Simple + Balanced GraphSAGE
        # ---------------------------
        data = load_transactions_graph_gcn()
        model, logits = _train_graphsage(
            data,
            class_balanced=True,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )

        y_proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred  = logits.argmax(dim=1).cpu().numpy()
        y_true  = data.y.cpu().numpy()
        mask    = data.test_mask.cpu().numpy()
        t_all   = data.time_steps.cpu().numpy()

        results.append({
            "dataset": dataset.lower(),
            "model": "GraphSAGE",
            "setup": "simple_balanced",
            **_evaluate(y_true[mask], y_pred[mask], y_proba[mask]),
        })
        per_timestep["simple_balanced"] = _evaluate_per_timestep(
            t_all[mask], y_true[mask], y_pred[mask], y_proba[mask]
        )

        # Save aggregate results for this seed
        df_seed = pd.DataFrame(results)
        df_seed["seed"] = seed
        all_runs.append(df_seed)

        # Save per-time-step results for this seed
        for setup_name, df_ts in per_timestep.items():
            df_ts = df_ts.copy()
            df_ts["model"] = "GraphSAGE"
            df_ts["setup"] = setup_name
            df_ts["seed"] = seed
            per_timestep_all.append(df_ts)

    # Combine all aggregate results
    df_all = pd.concat(all_runs, ignore_index=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    df_mean = df_all.groupby(["dataset", "model", "setup"])[metrics].mean()
    df_std  = df_all.groupby(["dataset", "model", "setup"])[metrics].std()

    df_summary = df_mean.copy()
    for m in metrics:
        df_summary[m] = (
            df_mean[m].round(4).astype(str) + " ± " + df_std[m].round(4).astype(str)
        )

    df_summary = df_summary.reset_index()

    # Combine all per-time-step results
    df_per_timestep = pd.concat(per_timestep_all, ignore_index=True)

    return df_summary, df_per_timestep

# ---------------------------------------------------
# Run EvolveGCN-O on graph dataset (simple + simple_balanced)
# ---------------------------------------------------
def run_evolve_gcn_experiments(
    dataset="elliptic-transactions-graph",
    device="cpu",
    seed=42,
    deterministic=True,
):
    results = []
    per_timestep = {}

    bundle = load_transactions_graph_temporal()
    test_steps = set(bundle["test_time_steps"])

    # ---------------------------
    # 1. Simple
    # ---------------------------
    _, outputs = _train_evolve_gcn(
        bundle,
        class_balanced=False,
        device=device,
        seed=seed,
        deterministic=deterministic,
        epochs=50,
    )

    agg_y, agg_pred, agg_proba, agg_t = [], [], [], []

    for t, out in outputs.items():
        if t in test_steps:
            mask = out["mask"].numpy()
            y_true = out["y"].numpy()[mask]
            y_pred = out["logits"].argmax(dim=1).numpy()[mask]
            y_proba = F.softmax(out["logits"], dim=1)[:, 1].numpy()[mask]

            agg_y.extend(y_true)
            agg_pred.extend(y_pred)
            agg_proba.extend(y_proba)
            agg_t.extend([t] * len(y_true))

    agg_y = np.array(agg_y)
    agg_pred = np.array(agg_pred)
    agg_proba = np.array(agg_proba)
    agg_t = np.array(agg_t)

    results.append({
        "dataset": dataset.lower(),
        "model": "EvolveGCN-O",
        "setup": "simple",
        **_evaluate(agg_y, agg_pred, agg_proba),
    })

    per_timestep["simple"] = _evaluate_per_timestep(
        agg_t, agg_y, agg_pred, agg_proba
    )

    # ---------------------------
    # 2. Simple + Balanced
    # ---------------------------
    _, outputs = _train_evolve_gcn(
        bundle,
        class_balanced=True,
        device=device,
        seed=seed,
        deterministic=deterministic,
        epochs=50,
    )

    agg_y, agg_pred, agg_proba, agg_t = [], [], [], []

    for t, out in outputs.items():
        if t in test_steps:
            mask = out["mask"].numpy()
            y_true = out["y"].numpy()[mask]
            y_pred = out["logits"].argmax(dim=1).numpy()[mask]
            y_proba = F.softmax(out["logits"], dim=1)[:, 1].numpy()[mask]

            agg_y.extend(y_true)
            agg_pred.extend(y_pred)
            agg_proba.extend(y_proba)
            agg_t.extend([t] * len(y_true))

    agg_y = np.array(agg_y)
    agg_pred = np.array(agg_pred)
    agg_proba = np.array(agg_proba)
    agg_t = np.array(agg_t)

    results.append({
        "dataset": dataset.lower(),
        "model": "EvolveGCN-O",
        "setup": "simple_balanced",
        **_evaluate(agg_y, agg_pred, agg_proba),
    })

    per_timestep["simple_balanced"] = _evaluate_per_timestep(
        agg_t, agg_y, agg_pred, agg_proba
    )

    return pd.DataFrame(results), per_timestep

# ---------------------------------------------------
# XAI training wrappers for RQ3
# ---------------------------------------------------
def train_graphsage_xai(
    data,
    device="cpu",
    seed=42,
    deterministic=True,
    class_balanced=False,
):
    """
    Public wrapper for training GraphSAGE for RQ3 XAI.
    """

    model, logits = _train_graphsage(
        data=data,
        class_balanced=class_balanced,
        device=device,
        seed=seed,
        deterministic=deterministic,
    )

    return model, logits