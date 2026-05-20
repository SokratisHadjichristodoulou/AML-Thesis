import pandas as pd
from pathlib import Path
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data

from src.preprocessing import preprocess_ethereum
from src.preprocessing import preprocess_elliptic_wallets_combined
from src.preprocessing import preprocess_elliptic_transactions_combined
from src.preprocessing import preprocess_elliptic_graph

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
def _get_project_root():
    return Path(__file__).resolve().parent.parent

def _get_ethereum_raw_path():
    root = _get_project_root()
    return root / "data" / "ethereum_fraud" / "ethereum-frauddetection-dataset" / "transaction_dataset.csv"

def _get_ethereum_clean_path():
    root = _get_project_root()
    return root / "data" / "clean_datasets" / "ethereum" / "ethereum_fraud_clean.csv"

def _get_elliptic_wallets_combined_raw_path():
    root = _get_project_root()
    return root / "data" / "elliptic_plus_plus" / "wallets" / "wallets_features_classes_combined.csv"

def _get_elliptic_wallets_combined_clean_path():
    root = _get_project_root()
    return root / "data" / "clean_datasets" / "elliptic_plus_plus" / "wallets_features_classes_combined_clean.csv"

def _get_elliptic_transactions_features_raw_path():
    root = _get_project_root()
    return root / "data" / "elliptic_plus_plus" / "transactions" / "txs_features.csv"

def _get_elliptic_transactions_classes_raw_path():
    root = _get_project_root()
    return root / "data" / "elliptic_plus_plus" / "transactions" / "txs_classes.csv"

def _get_elliptic_transactions_combined_clean_path():
    root = _get_project_root()
    return root / "data" / "clean_datasets" / "elliptic_plus_plus" / "transactions_features_classes_combined_clean.csv"

def _get_elliptic_graph_clean_path():
    root = _get_project_root()
    return root / "data" / "clean_datasets" / "elliptic_plus_plus" / "transactions_graph_clean.csv"

def _get_transactions_edgelist_path():
    root = _get_project_root()
    return root / "data" / "elliptic_plus_plus" / "transactions" / "txs_edgelist.csv"

# ---------------------------------------------------
# Raw loader
# ---------------------------------------------------
def _load_raw_ethereum():
    return pd.read_csv(_get_ethereum_raw_path())

def _load_raw_elliptic_wallets_combined():
    return pd.read_csv(_get_elliptic_wallets_combined_raw_path())

def _load_raw_elliptic_transactions_features():
    return pd.read_csv(_get_elliptic_transactions_features_raw_path())

def _load_raw_elliptic_transactions_classes():
    return pd.read_csv(_get_elliptic_transactions_classes_raw_path())

def _load_raw_elliptic_transactions_combined():
    tx_features = _load_raw_elliptic_transactions_features()
    tx_classes = _load_raw_elliptic_transactions_classes()

    df = pd.merge(tx_features, tx_classes, on="txId", how="inner")

    return df

# ---------------------------------------------------
# Build + save cleaned dataset ONCE
# ---------------------------------------------------
def save_preprocessed_ethereum_fraud(force=False):
    clean_path = _get_ethereum_clean_path()

    if clean_path.exists() and not force:
        print(f"Clean file already exists: {clean_path}")
        return pd.read_csv(clean_path)

    df = _load_raw_ethereum()
    df = preprocess_ethereum(df)

    clean_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(clean_path, index=False)

    print(f"Saved cleaned dataset to: {clean_path}")
    return df

def save_preprocessed_elliptic_wallets_combined(force=False):
    clean_path = _get_elliptic_wallets_combined_clean_path()

    if clean_path.exists() and not force:
        print(f"Clean file already exists: {clean_path}")
        return pd.read_csv(clean_path)

    df = _load_raw_elliptic_wallets_combined()
    df = preprocess_elliptic_wallets_combined(df)
    
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(clean_path, index=False)

    print(f"Saved cleaned dataset to: {clean_path}")
    return df

def save_preprocessed_elliptic_transactions_combined(force=False):
    clean_path = _get_elliptic_transactions_combined_clean_path()

    if clean_path.exists() and not force:
        print(f"Clean file already exists: {clean_path}")
        return pd.read_csv(clean_path)

    df = _load_raw_elliptic_transactions_combined()
    df = preprocess_elliptic_transactions_combined(df)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)

    print(f"Saved cleaned dataset to: {clean_path}")
    return df

def save_preprocessed_elliptic_graph(force=False):
    clean_path = _get_elliptic_graph_clean_path()

    if clean_path.exists() and not force:
        print(f"Clean file already exists: {clean_path}")
        return pd.read_csv(clean_path)

    df = _load_raw_elliptic_transactions_combined()
    df = preprocess_elliptic_graph(df)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)

    print(f"Saved cleaned dataset to: {clean_path}")
    return df

# ---------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------
def _load_preprocessed_ethereum():
    clean_path = _get_ethereum_clean_path()

    if clean_path.exists():
        return pd.read_csv(clean_path)

    # If the cleaned file does not exist yet, build it once
    return save_preprocessed_ethereum_fraud()

def _load_preprocessed_elliptic_wallets_combined():
    clean_path = _get_elliptic_wallets_combined_clean_path()

    if clean_path.exists():
        return pd.read_csv(clean_path)

    # If the cleaned file does not exist yet, build it once
    return save_preprocessed_elliptic_wallets_combined()

def _load_preprocessed_elliptic_transactions_combined():
    clean_path = _get_elliptic_transactions_combined_clean_path()

    if clean_path.exists():
        return pd.read_csv(clean_path)

    # If the cleaned file does not exist yet, build it once
    return save_preprocessed_elliptic_transactions_combined()

def _load_preprocessed_elliptic_graph():
    clean_path = _get_elliptic_graph_clean_path()

    if clean_path.exists():
        return pd.read_csv(clean_path)

    # If the cleaned file does not exist yet, build it once
    return save_preprocessed_elliptic_graph()

def _load_transactions_edgelist():
    return pd.read_csv(_get_transactions_edgelist_path())

"""
def _load_raw_ethereum():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "ethereum_fraud" / "ethereum-frauddetection-dataset" / "transaction_dataset.csv"

    df = pd.read_csv(data_path)
    #  df = preprocess_ethereum(df)

    return df

def _load_preprocessed_ethereum():
    df = _load_raw_ethereum()

    return preprocess_ethereum(df)

def _load_raw_elliptic_wallets_combined():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "elliptic_plus_plus" / "wallets_features_classes_combined.csv"

    df = pd.read_csv(data_path)

    return df

def _load_preprocessed_elliptic_wallets_combined():
    df = _load_raw_elliptic_wallets_combined()

    return preprocess_elliptic_wallets_combined(df)
"""

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def _split_features_target(df, target_col="class", test_size=0.2, random_state=42, shuffle=True):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y if shuffle else None
    )

# ---------------------------------------------------
# Time-step-boundary split for RQ2
# Shared by: static feature baseline, static GNN, dynamic GNN
#
# Differs from _time_based_split_features_target:
#   - Splits on TIME-STEP boundaries (not row positions), so no time step
#     is ever cut in half. Required for clean per-timestep evaluation and
#     for GNNs (no cross-split edges within a time step).
#   - Returns train_time and test_time as separate metadata arrays so the
#     model never sees Time step as a feature, but downstream code can
#     group by time step for evaluation and temporal training.
# ---------------------------------------------------
def _time_step_boundary_split(
    df,
    target_col="class",
    time_col="Time step",
    test_size=0.2,
):
    """
    Split a temporal dataset on time-step boundaries.

    Guarantees that every row of a given time step lies entirely on one
    side of the split. The time column is always removed from X_train and
    X_test (model never sees it as a feature), but returned separately
    as train_time and test_time for:
      - Per-timestep evaluation (static baseline, static GNN, dynamic GNN)
      - Temporal training loops (dynamic GNN only)

    Parameters:
        df: DataFrame containing features, target, and time column
        target_col: name of the target column
        time_col: name of the time-step column
        test_size: fraction of UNIQUE time steps assigned to test
                   (not row fraction)

    Returns:
        X_train, X_test, y_train, y_test, train_time, test_time
        - All six objects have reset integer indices (0..n-1) aligned
          row-for-row within the train split and within the test split.
        - train_time and test_time are pd.Series of time-step values.
    """
    df = df.sort_values(time_col).reset_index(drop=True)

    # Split on time-step boundaries
    unique_steps = sorted(df[time_col].unique())
    n_test_steps = max(1, int(round(len(unique_steps) * test_size)))
    train_steps = unique_steps[:-n_test_steps]
    test_steps  = unique_steps[-n_test_steps:]

    train_df = df[df[time_col].isin(train_steps)].copy().reset_index(drop=True)
    test_df  = df[df[time_col].isin(test_steps)].copy().reset_index(drop=True)

    # Capture time steps BEFORE dropping the column
    train_time = train_df[time_col].copy()
    test_time  = test_df[time_col].copy()

    # Drop target AND time column from features
    X_train = train_df.drop(columns=[target_col, time_col])
    X_test  = test_df.drop(columns=[target_col, time_col])

    y_train = train_df[target_col]
    y_test  = test_df[target_col]

    return X_train, X_test, y_train, y_test, train_time, test_time

def _scale_selected_columns(X_train, X_test, columns_to_scale):
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train, X_test, scaler

# ---------------------------------------------------
# 1. Ethereum - Simple split (baseline)
# ---------------------------------------------------
def load_ethereum_simple(
    test_size=0.2,
    random_state=42,
    target_col="class",
    shuffle=True,
):
    df = _load_preprocessed_ethereum()

    X_train, X_test, y_train, y_test = _split_features_target(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return X_train, X_test, y_train, y_test

# ---------------------------------------------------
# 2. Ethereum - Split + SMOTE (no scaling)
# ---------------------------------------------------
def load_ethereum_smote(
    test_size=0.2,
    random_state=42,
    target_col="class",
    smote_random_state=42,
    shuffle=True,
):
    df = _load_preprocessed_ethereum()

    X_train, X_test, y_train, y_test = _split_features_target(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    smote = SMOTE(random_state=smote_random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Convert back to DataFrame (IMPORTANT for feature names)
    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)

    return X_train_sm, X_test, y_train_sm, y_test

# ---------------------------------------------------
# 3. Ethereum - Split + Scaling
# ---------------------------------------------------
def load_ethereum_simple_scaled(
    test_size=0.2,
    random_state=42,
    target_col="class",
    shuffle=True,
):
    df = _load_preprocessed_ethereum()

    X_train, X_test, y_train, y_test = _split_features_target(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ---------------------------------------------------
# 4. Ethereum - Split + Scaling + SMOTE
# For scaling-sensitive models Logistic Regression and SVM
# ---------------------------------------------------
def load_ethereum_smote_scaled(
    test_size=0.2,
    random_state=42,
    target_col="class",
    smote_random_state=42,
    shuffle=True,
):
    df = _load_preprocessed_ethereum()

    X_train, X_test, y_train, y_test = _split_features_target(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    # Step 1: Scale
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 2: SMOTE on scaled data
    smote = SMOTE(random_state=smote_random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    # Convert back to DataFrame
    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_sm, X_test_scaled, y_train_sm, y_test, scaler

# ---------------------------------------------------
# 5. Elliptic wallets combined - Simple split
# ---------------------------------------------------
def load_wallet_combined_simple(
    test_size=0.2,
    random_state=42,
    target_col="class",
    shuffle=False,
):
    df = _load_preprocessed_elliptic_wallets_combined()

    X_train, X_test, y_train, y_test = _split_features_target(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return X_train, X_test, y_train, y_test

# ---------------------------------------------------
# 6. Elliptic wallets combined - Split + SMOTE
# ---------------------------------------------------
def load_wallet_combined_smote(
    test_size=0.2,
    random_state=42,
    target_col="class",
    smote_random_state=42,
    shuffle=False,
):
    df = _load_preprocessed_elliptic_wallets_combined()

    X_train, X_test, y_train, y_test = _split_features_target(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    smote = SMOTE(random_state=smote_random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)

    return X_train_sm, X_test, y_train_sm, y_test

# ---------------------------------------------------
# 7. Elliptic wallets combined - Split + Scaling
# ---------------------------------------------------
def load_wallet_combined_simple_scaled(
    test_size=0.2,
    random_state=42,
    target_col="class",
    shuffle=False,
):
    df = _load_preprocessed_elliptic_wallets_combined()

    X_train, X_test, y_train, y_test = _split_features_target(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ---------------------------------------------------
# 8. Elliptic wallets combined - Split + Scaling + SMOTE
# For scaling-sensitive models Logistic Regression and SVM
# ---------------------------------------------------
def load_wallet_combined_smote_scaled(
    test_size=0.2,
    random_state=42,
    target_col="class",
    smote_random_state=42,
    shuffle=False,
):
    df = _load_preprocessed_elliptic_wallets_combined()

    X_train, X_test, y_train, y_test = _split_features_target(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    # Step 1: Scale on training set only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 2: Apply SMOTE after scaling, only on training set
    smote = SMOTE(random_state=smote_random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_sm, X_test_scaled, y_train_sm, y_test, scaler

# ---------------------------------------------------
# 9. Elliptic transactions features + classes combined - Simple split (time-based)
# ---------------------------------------------------
def load_transactions_combined_simple(
    test_size=0.2,
    target_col="class",
    time_col="Time step",
    drop_time_from_features=True,
):
    df = _load_preprocessed_elliptic_transactions_combined()

    X_train, X_test, y_train, y_test, _, _ = _time_step_boundary_split(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
    )

    return X_train, X_test, y_train, y_test

# ---------------------------------------------------
# 10. Elliptic transactions features + classes combined - Split + SMOTE (time-based)
# ---------------------------------------------------
def load_transactions_combined_smote(
    test_size=0.2,
    target_col="class",
    time_col="Time step",
    smote_random_state=42,
    drop_time_from_features=True,
):
    df = _load_preprocessed_elliptic_transactions_combined()

    X_train, X_test, y_train, y_test, _, _ = _time_step_boundary_split(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
    )

    smote = SMOTE(random_state=smote_random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)

    return X_train_sm, X_test, y_train_sm, y_test

# ---------------------------------------------------
# 11. Elliptic transactions features + classes combined - Split + selective scaling (time-based)
# Scale only columns 168-184 (1-based positions in feature matrix)
# ---------------------------------------------------
def load_transactions_combined_simple_scaled(
    test_size=0.2,
    target_col="class",
    time_col="Time step",
    drop_time_from_features=True,
):
    df = _load_preprocessed_elliptic_transactions_combined()

    X_train, X_test, y_train, y_test, _, _ = _time_step_boundary_split(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
    )

    # Columns 168-184
    cols_to_scale = X_train.columns[167:184]

    X_train_scaled, X_test_scaled, scaler = _scale_selected_columns(
        X_train,
        X_test,
        cols_to_scale
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ---------------------------------------------------
# 12. Elliptic transactions features + classes combined - Split + selective scaling + SMOTE (time-based)
# Scale only columns 168-184, then apply SMOTE on training set only
# ---------------------------------------------------
def load_transactions_combined_smote_scaled(
    test_size=0.2,
    target_col="class",
    time_col="Time step",
    smote_random_state=42,
    drop_time_from_features=True,
):
    df = _load_preprocessed_elliptic_transactions_combined()

    X_train, X_test, y_train, y_test, _, _ = _time_step_boundary_split(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
    )

    # Columns 168-184
    cols_to_scale = X_train.columns[167:184]

    X_train_scaled, X_test_scaled, scaler = _scale_selected_columns(
        X_train,
        X_test,
        cols_to_scale
    )

    smote = SMOTE(random_state=smote_random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)

    return X_train_sm, X_test_scaled, y_train_sm, y_test, scaler

# ---------------------------------------------------
# 13. Elliptic transactions Graph - RQ2 loader
# Single source of truth for all three RQ2 stages:
#   - Static feature baseline (uses: X_*, y_*, test_time)
#   - Static GNN              (uses: X_*, y_*, test_time)
#   - Dynamic GNN             (uses: X_*, y_*, train_time, test_time) 
#
#   - Static feature baseline (drop_unknowns=True):
#       class-3 rows removed entirely from X/y/time arrays
#   - Static GNN / Dynamic GNN (drop_unknowns=False):
#       class-3 rows kept in X/y/time, plus boolean masks returned
#       so the model can see them as graph nodes but skip them in loss
# ---------------------------------------------------
def load_transactions_graph_baseline_simple(
    test_size=0.2,
    target_col="class",
    time_col="Time step",
    drop_unknowns=True,
    unknown_label=3,
):
    df = _load_preprocessed_elliptic_graph()

    X_train, X_test, y_train, y_test, train_time, test_time = _time_step_boundary_split(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
    )

    if drop_unknowns:
        # Remove unknown-class rows entirely — feature baseline has no graph
        # structure to benefit from keeping them.
        train_mask = y_train != unknown_label
        test_mask  = y_test  != unknown_label

        X_train    = X_train.loc[train_mask].reset_index(drop=True)
        y_train    = y_train.loc[train_mask].reset_index(drop=True)
        train_time = train_time.loc[train_mask].reset_index(drop=True)

        X_test     = X_test.loc[test_mask].reset_index(drop=True)
        y_test     = y_test.loc[test_mask].reset_index(drop=True)
        test_time  = test_time.loc[test_mask].reset_index(drop=True)

        return X_train, X_test, y_train, y_test, train_time, test_time

    # GNN stages: keep unknowns, return supervision masks so the caller can
    # exclude them from loss and metrics while keeping them in the graph.
    train_supervision_mask = (y_train != unknown_label).values
    test_supervision_mask  = (y_test  != unknown_label).values

    return (
        X_train, X_test, y_train, y_test,
        train_time, test_time,
        train_supervision_mask, test_supervision_mask,
    )

# ---------------------------------------------------
# 14. Elliptic transactions Graph - GCN loader (RQ2)
# Builds a PyTorch Geometric Data object from:
#   - nodes:  transactions_graph_clean.csv (txId, Time step, features, class)
#   - edges:  txs_edgelist.csv (txId1 -> txId2)
# Unknowns (class 3) are kept as graph nodes but masked out of supervision.
# ---------------------------------------------------
def load_transactions_graph_gcn(
    test_size=0.2,
    target_col="class",
    time_col="Time step",
    id_col="txId",
    unknown_label=3,
):
    # 1. Load nodes (preprocessed graph CSV) and edges
    nodes_df = _load_preprocessed_elliptic_graph()
    edges_df = _load_transactions_edgelist()

    # 2. Sort nodes by time (same ordering as _time_step_boundary_split)
    nodes_df = nodes_df.sort_values(time_col).reset_index(drop=True)

    # 3. Build a mapping: txId -> row index (node index in the graph)
    txid_to_idx = {txid: i for i, txid in enumerate(nodes_df[id_col].values)}

    # 4. Convert edges to node-index pairs, drop edges with unknown endpoints
    src = edges_df["txId1"].map(txid_to_idx)
    dst = edges_df["txId2"].map(txid_to_idx)
    valid = src.notna() & dst.notna()
    edge_index = torch.tensor(
        np.stack([src[valid].astype(int).values, dst[valid].astype(int).values]),
        dtype=torch.long,
    )

    # 5. Build time-step-boundary split on node rows
    unique_steps = sorted(nodes_df[time_col].unique())
    n_test_steps = max(1, int(round(len(unique_steps) * test_size)))
    train_steps = unique_steps[:-n_test_steps]
    test_steps  = unique_steps[-n_test_steps:]

    time_series = nodes_df[time_col].values
    labels      = nodes_df[target_col].values

    train_mask = torch.tensor(
        [(t in train_steps) and (c != unknown_label) for t, c in zip(time_series, labels)],
        dtype=torch.bool,
    )
    test_mask  = torch.tensor(
        [(t in test_steps)  and (c != unknown_label) for t, c in zip(time_series, labels)],
        dtype=torch.bool,
    )

    # 6. Node features: drop id, time, and class
    x = nodes_df.drop(columns=[id_col, time_col, target_col]).values.astype("float32")
    x = torch.tensor(x, dtype=torch.float32)

    # 7. Labels: replace class 3 with 0 just to have a valid int (masks handle exclusion)
    y_clean = pd.Series(labels).replace(unknown_label, 0).values
    y = torch.tensor(y_clean, dtype=torch.long)

    time_steps = torch.tensor(time_series, dtype=torch.long)

    # 8. Pack into a PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask  = test_mask
    data.time_steps = time_steps

    return data

def load_transactions_graph_temporal(
    target_col="class",
    time_col="Time step",
    id_col="txId",
    unknown_label=3,
    test_size=0.2,
):
    """
    Build a list of per-timestep graph snapshots for dynamic GNN training.

    Returns a dict with:
        snapshots: list of PyG Data objects, one per time step
        train_time_steps: list of ints
        test_time_steps: list of ints
        n_features: int
    """
    nodes_df = _load_preprocessed_elliptic_graph()
    edges_df = _load_transactions_edgelist()

    nodes_df = nodes_df.sort_values(time_col).reset_index(drop=True)

    # Time-step boundary split
    unique_steps = sorted(nodes_df[time_col].unique())
    n_test_steps = max(1, int(round(len(unique_steps) * test_size)))
    train_time_steps = unique_steps[:-n_test_steps]
    test_time_steps  = unique_steps[-n_test_steps:]

    # Feature columns = everything except id, time, class
    feature_cols = [c for c in nodes_df.columns if c not in [id_col, time_col, target_col]]

    snapshots = []
    for t in unique_steps:
        nodes_t = nodes_df[nodes_df[time_col] == t].reset_index(drop=True)

        local_ids = nodes_t[id_col].values
        local_to_new_idx = {txid: i for i, txid in enumerate(local_ids)}

        # Keep only edges fully inside this snapshot
        local_id_set = set(local_ids)
        edges_t = edges_df[
            edges_df["txId1"].isin(local_id_set) & edges_df["txId2"].isin(local_id_set)
        ]

        if len(edges_t) > 0:
            src = edges_t["txId1"].map(local_to_new_idx).astype(int).values
            dst = edges_t["txId2"].map(local_to_new_idx).astype(int).values
            edge_index = torch.tensor(
                np.stack([src, dst]),
                dtype=torch.long,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        x = torch.tensor(
            nodes_t[feature_cols].values.astype("float32"),
            dtype=torch.float32,
        )

        labels = nodes_t[target_col].values
        y = torch.tensor(
            pd.Series(labels).replace(unknown_label, 0).values,
            dtype=torch.long,
        )

        supervision_mask = torch.tensor(labels != unknown_label, dtype=torch.bool)

        snap = Data(x=x, edge_index=edge_index, y=y)
        snap.supervision_mask = supervision_mask
        snap.time_step = int(t)
        snapshots.append(snap)

    return {
        "snapshots": snapshots,
        "train_time_steps": train_time_steps,
        "test_time_steps": test_time_steps,
        "n_features": len(feature_cols),
    }

# ---------------------------------------------------
# XAI loaders for RQ3
# ---------------------------------------------------
def load_feature_xai_dataset(dataset="elliptic-transactions-combined"):
    """
    Loader for SHAP explanations in RQ3.

    Supported datasets:
        - ethereum
        - elliptic-wallets-combined
        - elliptic-transactions-combined

    Returns:
        X_train, X_test, y_train, y_test
    """

    dataset = dataset.lower()

    if dataset == "ethereum":
        return load_ethereum_simple()

    if dataset == "elliptic-wallets-combined":
        return load_wallet_combined_simple()

    if dataset == "elliptic-transactions-combined":
        return load_transactions_combined_simple()

    raise ValueError(f"Unknown feature XAI dataset: {dataset}")


def load_graph_xai_dataset(
    test_size=0.2,
    target_col="class",
    time_col="Time step",
    id_col="txId",
    unknown_label=3,
):
    """
    Loader for GNNExplainer in RQ3.

    Returns:
        data: PyTorch Geometric Data object
        metadata: dictionary with:
            - nodes_df
            - feature_names
            - test_illicit_nodes
    """

    data = load_transactions_graph_gcn(
        test_size=test_size,
        target_col=target_col,
        time_col=time_col,
        id_col=id_col,
        unknown_label=unknown_label,
    )

    nodes_df = _load_preprocessed_elliptic_graph()
    nodes_df = nodes_df.sort_values(time_col).reset_index(drop=True)

    feature_names = [
        col for col in nodes_df.columns
        if col not in [id_col, time_col, target_col]
    ]

    test_illicit_nodes = (
        (data.test_mask == True) &
        (data.y == 1)
    ).nonzero(as_tuple=True)[0].cpu().numpy().tolist()

    metadata = {
        "nodes_df": nodes_df,
        "feature_names": feature_names,
        "test_illicit_nodes": test_illicit_nodes,
    }

    return data, metadata