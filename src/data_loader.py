import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.preprocessing import preprocess_ethereum
from src.preprocessing import preprocess_elliptic_wallets_combined
from src.preprocessing import preprocess_elliptic_transactions_combined

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

def _time_based_split_features_target(
    df,
    target_col="class",
    time_col="Time step",
    test_size=0.2,
    drop_time_from_features=True,
):
    df = df.sort_values(time_col).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    if drop_time_from_features:
        X_train = X_train.drop(columns=[time_col], errors="ignore")
        X_test = X_test.drop(columns=[time_col], errors="ignore")

    return X_train, X_test, y_train, y_test

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

    X_train, X_test, y_train, y_test = _time_based_split_features_target(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
        drop_time_from_features=drop_time_from_features,
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

    X_train, X_test, y_train, y_test = _time_based_split_features_target(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
        drop_time_from_features=drop_time_from_features,
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

    X_train, X_test, y_train, y_test = _time_based_split_features_target(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
        drop_time_from_features=drop_time_from_features,
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

    X_train, X_test, y_train, y_test = _time_based_split_features_target(
        df,
        target_col=target_col,
        time_col=time_col,
        test_size=test_size,
        drop_time_from_features=drop_time_from_features,
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