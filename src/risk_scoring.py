from pathlib import Path

from IPython.display import display
import pandas as pd

from src.data_loader import load_feature_xai_dataset
from src.feature_models import get_random_forest_balanced
from src.training import _evaluate

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
def _get_project_root():
    return Path(__file__).resolve().parent.parent

def _make_risk_scoring_dir():
    path = _get_project_root() / "models" / "rq4_risk_scoring" / "wallet_risk_scoring"
    path.mkdir(parents=True, exist_ok=True)
    return path

# ---------------------------------------------------
# Risk score helpers
# ---------------------------------------------------
def probability_to_risk_score(probability):
    """
    Convert fraud probability into a 0-100 wallet risk score.
    """
    return probability * 100

def assign_risk_level(risk_score):
    """
    Convert numerical risk score into compliance-friendly risk category.
    """

    if risk_score < 25:
        return "Low"
    elif risk_score < 50:
        return "Medium"
    elif risk_score < 75:
        return "High"
    else:
        return "Critical"

def assign_recommended_action(risk_level):
    """
    Map risk category to a simple compliance action.
    """

    if risk_level == "Low":
        return "No immediate action required"
    elif risk_level == "Medium":
        return "Monitor wallet activity"
    elif risk_level == "High":
        return "Review wallet manually"
    elif risk_level == "Critical":
        return "Prioritize for compliance investigation"
    else:
        return "Unknown action"

# ---------------------------------------------------
# Main RQ4 risk scoring function
# ---------------------------------------------------
def run_wallet_risk_scoring(
    dataset="elliptic-wallets-combined",
):
    """
    Train the best wallet-level model and convert binary AML predictions
    into wallet-level risk scores.

    Recommended dataset:
        - elliptic-wallets-combined

    Saves:
        - metrics.csv
        - wallet_risk_scores.csv

    Returns:
        dict with model, metrics, risk_scores, and save_dir.
    """

    save_dir = _make_risk_scoring_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load wallet dataset
    X_train, X_test, y_train, y_test = load_feature_xai_dataset(dataset)

    # 2. Train best model from previous experiments
    model = get_random_forest_balanced()
    model.fit(X_train, y_train)

    # 3. Predict binary labels and fraud probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 4. Evaluate model
    metrics = _evaluate(y_test, y_pred, y_proba)

    metrics_df = pd.DataFrame([{
        "dataset": dataset,
        "model": "Random Forest balanced",
        **metrics,
    }])

    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

    # 5. Build risk scoring output
    risk_df = X_test.copy()

    risk_df["wallet_index"] = X_test.index
    risk_df["true_label"] = y_test.values
    risk_df["predicted_label"] = y_pred
    risk_df["fraud_probability"] = y_proba
    risk_df["risk_score"] = risk_df["fraud_probability"].apply(probability_to_risk_score)
    risk_df["risk_level"] = risk_df["risk_score"].apply(assign_risk_level)
    risk_df["recommended_action"] = risk_df["risk_level"].apply(assign_recommended_action)

    # 6. Put compliance columns first
    compliance_cols = [
        "wallet_index",
        "true_label",
        "predicted_label",
        "fraud_probability",
        "risk_score",
        "risk_level",
        "recommended_action",
    ]

    feature_cols = [col for col in risk_df.columns if col not in compliance_cols]

    risk_df = risk_df[compliance_cols + feature_cols]

    # 7. Sort highest-risk wallets first
    risk_df = risk_df.sort_values("risk_score", ascending=False)

    risk_df.to_csv(save_dir / "wallet_risk_scores.csv", index=False)

    print("=== Wallet Risk Scoring Metrics ===")
    display(metrics_df)

    print(f"\nWallet risk scores saved to: {save_dir}")

    return {
        "model": model,
        "metrics": metrics_df,
        "risk_scores": risk_df,
        "save_dir": save_dir,
    }

# ---------------------------------------------------
# Compliance output builder
# ---------------------------------------------------
def build_compliance_output(
    risk_scores_df,
    shap_df=None,
    lime_df=None,
    counterfactual_df=None,
    top_n_explanations=3,
):
    """
    Build a compact analyst-facing compliance table.

    Optional inputs:
        shap_df: SHAP explanations if available
        lime_df: LIME explanations if available
        counterfactual_df: counterfactual explanations if available

    Returns:
        compliance_df
    """

    compliance_df = risk_scores_df.copy()

    # Keep only main compliance columns
    keep_cols = [
        "wallet_index",
        "true_label",
        "predicted_label",
        "fraud_probability",
        "risk_score",
        "risk_level",
        "recommended_action",
    ]

    compliance_df = compliance_df[keep_cols]

    # Add placeholder columns
    if shap_df is not None and not shap_df.empty:

        top_shap_features = (
            shap_df
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n_explanations)["feature"]
            .tolist()
        )

        shap_summary = "; ".join(top_shap_features)

        compliance_df["top_shap_drivers"] = shap_summary

    else:
        compliance_df["top_shap_drivers"] = ""

    compliance_df["top_lime_drivers"] = ""
    compliance_df["counterfactual_recommendation"] = ""

    # Add LIME drivers if available
    if lime_df is not None and not lime_df.empty:
        lime_grouped = (
            lime_df
            .sort_values(["wallet_index", "lime_importance"], ascending=[True, False])
            .groupby("wallet_index")
            .head(top_n_explanations)
            .groupby("wallet_index")["feature_rule"]
            .apply(lambda x: "; ".join(x.astype(str)))
        )

        compliance_df["top_lime_drivers"] = (
            compliance_df["wallet_index"]
            .map(lime_grouped)
            .fillna("")
        )

    # Add counterfactual explanations if available
    if counterfactual_df is not None and not counterfactual_df.empty:
        cf_grouped = (
            counterfactual_df
            .sort_values(
                ["wallet_index", "risk_score_reduction"],
                ascending=[True, False]
            )
            .groupby("wallet_index")
            .head(top_n_explanations)
            .groupby("wallet_index")["explanation"]
            .apply(lambda x: "; ".join(x.astype(str)))
        )

        compliance_df["counterfactual_recommendation"] = (
            compliance_df["wallet_index"]
            .map(cf_grouped)
            .fillna("")
        )

    return compliance_df

# ---------------------------------------------------
# Save final compliance output
# ---------------------------------------------------
def save_compliance_output(
    compliance_df,
    filename="compliance_output.csv",
):
    """
    Save the final RQ4 compliance output table.
    """

    save_dir = _make_risk_scoring_dir()
    output_path = save_dir / filename

    compliance_df.to_csv(output_path, index=False)

    print(f"Compliance output saved to: {output_path}")

    return output_path