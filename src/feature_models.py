from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def get_random_forest(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
):
    """
    Returns a configured Random Forest model.

    Parameters:
        n_estimators (int): number of trees
        max_depth (int or None): max depth of trees
        min_samples_split (int): min samples to split
        min_samples_leaf (int): min samples in leaf
        random_state (int): reproducibility
        n_jobs (int): parallel processing

    Useful for datasets with SMOTE.

    Returns:
        RandomForestClassifier
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return model

def get_random_forest_balanced(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
):
    """
    Random Forest with class balancing.

    Useful for imbalanced datasets without SMOTE.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return model

def get_xgboost(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
):
    """
    Returns a configured XGBoost classifier.

    Parameters:
        n_estimators (int): number of boosting rounds
        learning_rate (float): step size shrinkage
        max_depth (int): tree depth
        subsample (float): row sampling
        colsample_bytree (float): feature sampling
        random_state (int): reproducibility
        n_jobs (int): parallel processing

    Useful for datasets with SMOTE.
        
    Returns:
        XGBClassifier
    """

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
        use_label_encoder=False
    )

    return model

def get_xgboost_balanced(
    scale_pos_weight=None,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
):
    """
    XGBoost with imbalance handling.

    scale_pos_weight = (#negative / #positive)

    Useful for imbalanced datasets without SMOTE.
    """

    if scale_pos_weight is None:
        scale_pos_weight = 1.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
        use_label_encoder=False
    )

    return model

def get_lightgbm(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
):
    """
    Returns a configured LightGBM classifier.

    Parameters:
        n_estimators (int): number of boosting rounds
        learning_rate (float): step size
        max_depth (int): max tree depth (-1 = no limit)
        num_leaves (int): tree complexity
        subsample (float): row sampling
        colsample_bytree (float): feature sampling
        random_state (int): reproducibility
        n_jobs (int): parallel processing

    Useful for datasets with SMOTE.

    Returns:
        LGBMClassifier
    """

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return model

def get_lightgbm_balanced(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
):
    """
    LightGBM with class imbalance handling.

    Useful for imbalanced datasets without SMOTE.
    """

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return model

def get_logistic_regression(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
):
    """
    Returns a configured Logistic Regression model.

    IMPORTANT TO USE WITH SCALING
    Useful for datasets with SMOTE.
    """

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )

    return model

def get_logistic_regression_balanced(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
):
    """
    Logistic Regression with class imbalance handling.

    IMPORTANT TO USE WITH SCALING
    Useful for imbalanced datasets without SMOTE.
    """

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
    )

    return model

def get_svm(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    probability=True,
    random_state=42,
):
    """
    Returns a configured Support Vector Machine (SVM) model.

    IMPORTANT TO USE WITH SCALING
    Useful for datasets with SMOTE.
    """

    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
    )

    return model

def get_svm_balanced(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    probability=True,
    random_state=42,
):
    """
    SVM with class imbalance handling.

    IMPORTANT TO USE WITH SCALING
    Useful for imbalanced datasets without SMOTE.
    """

    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        class_weight="balanced",
        random_state=random_state,
    )

    return model

def get_linear_svc(
    C=1.0,
    max_iter=10000,
    random_state=42,
):
    """
    Returns a configured Linear Support Vector Classifier.

    IMPORTANT TO USE WITH SCALING
    Much faster than SVC for large datasets.
    """

    model = LinearSVC(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
    )

    return model


def get_linear_svc_balanced(
    C=1.0,
    max_iter=10000,
    random_state=42,
):
    """
    LinearSVC with class imbalance handling.

    IMPORTANT TO USE WITH SCALING
    Useful for imbalanced datasets without SMOTE.
    """

    model = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
    )

    return model