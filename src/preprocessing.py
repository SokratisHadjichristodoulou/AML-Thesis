import pandas as pd

def preprocess_ethereum(df):
    """
    Preprocess Ethereum Fraud.

    Steps:
    1. Drop identifier columns
    2. Clean Column Names
    3. Remove duplicates
    4. Drop high-cardinality categorical ERC20 features
    5. Drop zero-variance features
    6. Drop weak + highly correlated features (manual selection - based on the testing)
    7. Handle missing values (ERC20 features)
    8. Rename target column

    Returns:
        Cleaned DataFrame
    """
    
    # -------------------------------
    # 1. Drop irrelevant identifiers
    # -------------------------------
    drop_cols = ["Unnamed: 0", "Index", "Address"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    
    # -------------------------------
    # 2. Clean column names
    # -------------------------------
    df.columns = df.columns.str.strip() # remove leading/trailing spaces
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    
    # -------------------------------
    # 3. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()
    
    # -------------------------------
    # 4. Drop high-cardinality categorical ERC20 features
    # -------------------------------
    erc20_cat_cols = [
        "ERC20_most_sent_token_type",
        "ERC20_most_rec_token_type"
    ]
    df = df.drop(columns=[col for col in erc20_cat_cols if col in df.columns], errors="ignore")
    
    # -------------------------------
    # 5. Drop zero-variance features
    # -------------------------------
    zero_var_cols = df.columns[df.nunique() <= 1]
    df = df.drop(columns=zero_var_cols)
    
    # -------------------------------
    # 6. Drop weak + highly correlated features (manual selection)
    # -------------------------------
    drop_corr_cols = [
        "total_transactions_(including_tnx_to_create_contract",
        "Sent_tnx",
        "Received_Tnx",
        "min_val_sent",
        "Unique_Received_From_Addresses",
        "Unique_Sent_To_Addresses",
        "ERC20_avg_val_sent",
        "ERC20_max_val_sent",
        "ERC20_uniq_sent_token_name",
        "ERC20_total_ether_sent",
        "ERC20_uniq_sent_addr",
        "ERC20_min_val_sent",
        "Number_of_Created_Contracts"
    ]
    
    df = df.drop(columns=[col for col in drop_corr_cols if col in df.columns], errors="ignore")

    # -------------------------------
    # 7. Handle missing values (ERC20 features)
    # -------------------------------
    erc20_cols = [col for col in df.columns if "ERC20" in col]

    df[erc20_cols] = df[erc20_cols].fillna(0)

    # -------------------------------
    # 8. Rename target column
    # -------------------------------
    if "FLAG" in df.columns:
        df = df.rename(columns={"FLAG": "class"})
    
    return df

def preprocess_elliptic_wallets_combined(df):
    """
    Preprocess Elliptic wallet-level dataset.

    Steps:
    1. Drop identifier columns
    2. Drop Time step
    3. Remove duplicates
    4. Remove unknown class (class = 3)
    5. Rename class 2 -> 0

    Returns:
        Cleaned DataFrame
    """

    # -------------------------------
    # 1. Drop identifier columns
    # -------------------------------
    drop_cols = ["address"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # -------------------------------
    # 2. Drop Time step column
    # -------------------------------
    if "Time step" in df.columns:
        df = df.drop(columns=["Time step"])

    # -------------------------------
    # 3. Remove duplicates
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # 4. Remove unknown class (class 3)
    # -------------------------------
    if "class" in df.columns:
        df = df[df["class"] != 3].copy()

    # -------------------------------
    # 5. Rename class 2 -> 0
    # -------------------------------
    df["class"] = df["class"].replace(2, 0)

    return df

def preprocess_elliptic_transactions_combined(df):
    """
    Preprocess Elliptic Transaction-level features + classes dataset.

    Steps:
    1. Drop identifier columns
    2. Drop NaN rows
    3. Remove unknown class (class = 3)
    4. Rename class 2 -> 0

    Returns:
        Cleaned DataFrame
    """

    # -------------------------------
    # 1. Drop identifier columns
    # -------------------------------
    drop_cols = ["txId"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # -------------------------------
    # 2. Drop NaN rows
    # -------------------------------
    df = df.dropna()

    # -------------------------------
    # 3. Remove unknown class (class = 3)
    # -------------------------------
    if "class" in df.columns:
        df = df[df["class"] != 3].copy()

    # -------------------------------
    # 4. Rename class 2 -> 0
    # -------------------------------
    df["class"] = df["class"].replace(2, 0)

    return df

def preprocess_elliptic_graph(df):
    """
    Preprocess Elliptic Transaction-level features + classes dataset.

    Steps:
    1. Drop NaN rows
    2. Rename class 2 -> 0

    Returns:
        Cleaned DataFrame
    """

    # -------------------------------
    # 1. Drop NaN rows
    # -------------------------------
    df = df.dropna()

    # -------------------------------
    # 2. Rename class 2 -> 0
    # -------------------------------
    df["class"] = df["class"].replace(2, 0)

    return df
