from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn.functional as F

from IPython.display import display
from sklearn.metrics import classification_report
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.metrics import classification_report

from src.data_loader import load_feature_xai_dataset
from src.data_loader import load_graph_xai_dataset
from src.feature_models import get_lightgbm_balanced
from src.feature_models import get_random_forest_balanced
from src.training import train_graphsage_xai
from src.training import _evaluate


# ---------------------------------------------------
# Paths
# ---------------------------------------------------
def _get_project_root():
    return Path(__file__).resolve().parent.parent


def _make_xai_feature_dir():
    path = _get_project_root() / "models" / "rq3_xai_models" / "feature_models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _make_xai_graph_dir():
    path = _get_project_root() / "models" / "rq3_xai_models" / "graph_models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _make_xai_feature_lvl_risk_dir():
    path = _get_project_root() / "models" / "rq4_risk_scoring" / "feature_models"
    path.mkdir(parents=True, exist_ok=True)
    return path

# ---------------------------------------------------
# Helper
# ---------------------------------------------------
def _explain_single_node_graphsage(
    model,
    data,
    metadata,
    node_index,
    save_root,
    epochs=200,
):
    """
    Run GNNExplainer for a single node and save results to a per-node folder.
    
    Folder naming: illicit_node_{idx} or licit_node_{idx} based on TRUE label.
    
    Returns a dict with feature_importance_df, edge_importance_df, save_dir,
    and prediction info.
    """
    node_index = int(node_index)

    # ----- Get prediction info for this node -----
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = F.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)

    true_label = int(data.y[node_index].detach().cpu())
    pred_label = int(predictions[node_index].detach().cpu())
    fraud_prob = float(probabilities[node_index].detach().cpu())

    # ----- Build the folder name based on the TRUE label -----
    label_str = "illicit" if true_label == 1 else "licit"
    time_step = int(metadata["time_steps"][node_index])
    folder_name = f"{label_str}_node_{node_index}_step{time_step}"
    save_dir = save_root / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----- Configure GNNExplainer -----
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    # ----- Run explanation -----
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        index=node_index,
    )

    # ----- Feature importance -----
    node_mask = explanation.node_mask.detach().cpu().numpy()
    feature_scores = node_mask.mean(axis=0) if node_mask.ndim == 2 else node_mask

    feature_importance_df = pd.DataFrame({
        "feature": metadata["feature_names"],
        "importance": feature_scores,
    }).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(
        save_dir / "feature_importance.csv",
        index=False,
    )

    # ----- Edge importance -----
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    edge_index_cpu = data.edge_index.detach().cpu().numpy()

    edge_importance_df = pd.DataFrame({
        "source_node": edge_index_cpu[0],
        "target_node": edge_index_cpu[1],
        "importance": edge_mask,
    }).sort_values("importance", ascending=False)

    edge_importance_df.to_csv(
        save_dir / "edge_importance.csv",
        index=False,
    )

    # ----- Save prediction metadata as a small text file -----
    with open(save_dir / "prediction_info.txt", "w") as f:
        f.write(f"node_index: {node_index}\n")
        f.write(f"time_step: {time_step}\n")
        f.write(f"true_label: {true_label} ({label_str})\n")
        f.write(f"predicted_label: {pred_label}\n")
        f.write(f"fraud_probability: {fraud_prob:.4f}\n")
        f.write(f"correctly_classified: {true_label == pred_label}\n")

    # ----- Plot top features -----
    top_features = feature_importance_df.head(20).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title(
        f"GNNExplainer Feature Importance — "
        f"{label_str.capitalize()} node {node_index} "
        f"(pred={pred_label}, p={fraud_prob:.2f})"
    )
    plt.tight_layout()
    plt.savefig(
        save_dir / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "node_index": node_index,
        "true_label": true_label,
        "predicted_label": pred_label,
        "fraud_probability": fraud_prob,
        "correctly_classified": true_label == pred_label,
        "feature_importance": feature_importance_df,
        "edge_importance": edge_importance_df,
        "save_dir": save_dir,
    }

# ---------------------------------------------------
# SHAP - LightGBM
# ---------------------------------------------------
def run_shap_lightgbm(
    dataset="elliptic-transactions-combined",
    max_explain=1000,
):
    """
    Train LightGBM balanced and explain it using SHAP.

    Recommended datasets:
        - ethereum
        - elliptic-wallets-combined
        - elliptic-transactions-combined

    Saves:
        - metrics.csv
        - shap_feature_importance.csv
        - shap_summary_bar.png
        - shap_beeswarm.png
    """

    save_dir = _make_xai_feature_dir() / f"lightgbm_shap_{dataset}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_feature_xai_dataset(dataset)

    # 2. Train LightGBM balanced
    model = get_lightgbm_balanced()
    model.fit(X_train, y_train)

    # 3. Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _evaluate(y_test, y_pred, y_proba)

    metrics_df = pd.DataFrame([{
        "dataset": dataset,
        "model": "LightGBM balanced",
        **metrics,
    }])

    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

    print("=== LightGBM Balanced Metrics ===")
    display(metrics_df)

    print("\n=== Classification Report ===")
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True
        )
    ).transpose()

    display(report_df)

    # 4. Sample test data for SHAP
    X_explain = X_test.sample(
        n=min(max_explain, len(X_test)),
        random_state=42
    )

    # 5. SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Binary classification handling
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values

    # 6. Save feature importance table
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X_explain.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(
        save_dir / "shap_feature_importance.csv",
        index=False
    )

    # 7. SHAP bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_class1,
        X_explain,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        save_dir / "shap_summary_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 8. SHAP beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_class1,
        X_explain,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        save_dir / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSHAP LightGBM results saved to: {save_dir}")

    return {
        "model": model,
        "metrics": metrics_df,
        "importance": importance_df,
        "X_explain": X_explain,
        "shap_values": shap_values_class1,
        "save_dir": save_dir,
    }

# ---------------------------------------------------
# SHAP - Random Forest
# ---------------------------------------------------
def run_shap_random_forest(
    dataset="elliptic-transactions-combined",
    max_explain=1000,
):
    """
    Train Random Forest balanced and explain it using SHAP.

    Recommended datasets:
        - ethereum
        - elliptic-wallets-combined
        - elliptic-transactions-combined

    Saves:
        - metrics.csv
        - shap_feature_importance.csv
        - shap_summary_bar.png
        - shap_beeswarm.png
    """

    save_dir = _make_xai_feature_dir() / f"random_forest_shap_{dataset}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_feature_xai_dataset(dataset)

    # 2. Train Random Forest balanced
    model = get_random_forest_balanced()
    model.fit(X_train, y_train)

    # 3. Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _evaluate(y_test, y_pred, y_proba)

    metrics_df = pd.DataFrame([{
        "dataset": dataset,
        "model": "Random Forest balanced",
        **metrics,
    }])

    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

    print("=== Random Forest Balanced Metrics ===")
    display(metrics_df)

    print("\n=== Classification Report ===")
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True
        )
    ).transpose()

    display(report_df.round(4))

    # 4. Sample test data for SHAP
    X_explain = X_test.sample(
        n=min(max_explain, len(X_test)),
        random_state=42
    )

    # 5. SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Binary classification handling
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Shape is usually: samples x features x classes
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    # 6. Save feature importance table
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0).ravel()

    importance_df = pd.DataFrame({
        "feature": X_explain.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(
        save_dir / "shap_feature_importance.csv",
        index=False
    )

    # 7. SHAP bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_class1,
        X_explain,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        save_dir / "shap_summary_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 8. SHAP beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_class1,
        X_explain,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        save_dir / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSHAP Random Forest results saved to: {save_dir}")

    return {
        "model": model,
        "metrics": metrics_df,
        "importance": importance_df,
        "X_explain": X_explain,
        "shap_values": shap_values_class1,
        "save_dir": save_dir,
    }

# ---------------------------------------------------
# GNNExplainer - GraphSAGE
# ---------------------------------------------------
def run_gnnexplainer_graphsage(
    node_indices=None,
    n_illicit_nodes=10,
    device="cpu",
    seed=42,
    class_balanced=True,
    epochs=200,
    only_correctly_classified=True,
):
    """
    Train GraphSAGE once and explain one or many test nodes with GNNExplainer.

    Parameters
    ----------
    node_indices : list[int] or None
        Specific node indices to explain. If None, the function picks the first
        `n_illicit_nodes` illicit test nodes.
    n_illicit_nodes : int
        How many illicit test nodes to explain when node_indices is None.
    only_correctly_classified : bool
        When auto-picking illicit nodes, restrict to those GraphSAGE
        actually predicts as illicit. Recommended for the XAI study,
        since explaining wrong predictions tells you about model errors,
        not about what graph signal looks like for true frauds.

    Returns
    -------
    list of per-node result dicts.
    """

    save_root = _make_xai_graph_dir() / "graphsage_gnnexplainer"
    save_root.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    data, metadata = load_graph_xai_dataset()
    data = data.to(device)

    # 2. Train GraphSAGE once
    model, logits = train_graphsage_xai(
        data=data,
        device=device,
        seed=seed,
        deterministic=True,
        class_balanced=class_balanced,
    )
    model = model.to(device)
    model.eval()

    # 3. Decide which nodes to explain
    if node_indices is None:
        test_illicit_nodes = metadata["test_illicit_nodes"]

        if len(test_illicit_nodes) == 0:
            raise ValueError("No illicit test nodes found for explanation.")

        if only_correctly_classified:
            with torch.no_grad():
                logits_eval = model(data.x, data.edge_index)
                predictions = logits_eval.argmax(dim=1).cpu().numpy()

            test_illicit_nodes = [
                idx for idx in test_illicit_nodes
                if predictions[idx] == 1
            ]

            if len(test_illicit_nodes) == 0:
                raise ValueError(
                    "No correctly classified illicit test nodes found."
                )

        node_indices = test_illicit_nodes[:n_illicit_nodes]

    # 4. Run the explainer on each node
    results = []
    for i, node_idx in enumerate(node_indices, start=1):
        print(f"[{i}/{len(node_indices)}] Explaining node {node_idx}...")
        result = _explain_single_node_graphsage(
            model=model,
            data=data,
            metadata=metadata,
            node_index=node_idx,
            save_root=save_root,
            epochs=epochs,
        )
        print(
            f"  -> {result['save_dir'].name} | "
            f"true={result['true_label']} pred={result['predicted_label']} "
            f"p={result['fraud_probability']:.3f}"
        )
        results.append(result)

    print(f"\nAll results saved under: {save_root}")

    return results

# ---------------------------------------------------
# Local SHAP - Random Forest
# ---------------------------------------------------
def run_local_shap_random_forest(
    dataset="elliptic-wallets-combined",
    num_wallets_to_explain=10,
    num_features=10,
    max_explain=1000,
):
    """
    Train Random Forest balanced and generate local SHAP explanations
    for the highest-risk wallets.

    Recommended dataset:
        - elliptic-wallets-combined

    Saves:
        - metrics.csv
        - local_shap_explanations.csv

    Returns:
        dict with model, metrics, local explanations, and save_dir.
    """

    save_dir = _make_xai_feature_lvl_risk_dir() / f"random_forest_local_shap_{dataset}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_feature_xai_dataset(dataset)

    # 2. Train Random Forest balanced
    model = get_random_forest_balanced()
    model.fit(X_train, y_train)

    # 3. Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _evaluate(y_test, y_pred, y_proba)

    metrics_df = pd.DataFrame([{
        "dataset": dataset,
        "model": "Random Forest balanced",
        **metrics,
    }])

    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

    print("=== Random Forest Balanced Metrics ===")
    display(metrics_df)

    # 4. Select highest-risk wallets
    risk_df = X_test.copy()
    risk_df["true_label"] = y_test.values
    risk_df["predicted_label"] = y_pred
    risk_df["fraud_probability"] = y_proba
    risk_df["risk_score"] = y_proba * 100

    risk_df = risk_df.sort_values("risk_score", ascending=False)
    selected_wallets = risk_df.head(num_wallets_to_explain)

    # 5. SHAP explanation
    explainer = shap.TreeExplainer(model)

    X_selected = selected_wallets[X_train.columns]

    shap_values = explainer.shap_values(X_selected)

    # Binary classification handling
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    local_rows = []

    # 6. Build local explanation table
    for row_position, (wallet_index, wallet_row) in enumerate(selected_wallets.iterrows(), start=0):

        wallet_shap_values = shap_values_class1[row_position]

        wallet_explanation = pd.DataFrame({
            "feature": X_train.columns,
            "feature_value": X_selected.iloc[row_position].values,
            "shap_value": wallet_shap_values,
            "abs_shap_value": np.abs(wallet_shap_values),
        }).sort_values("abs_shap_value", ascending=False)

        wallet_explanation = wallet_explanation.head(num_features)

        for _, exp_row in wallet_explanation.iterrows():
            shap_value = float(exp_row["shap_value"])

            if shap_value > 0:
                direction = "increases risk"
            elif shap_value < 0:
                direction = "decreases risk"
            else:
                direction = "neutral"

            local_rows.append({
                "wallet_index": wallet_index,
                "rank_by_risk": row_position + 1,
                "true_label": int(wallet_row["true_label"]),
                "predicted_label": int(wallet_row["predicted_label"]),
                "fraud_probability": float(wallet_row["fraud_probability"]),
                "risk_score": float(wallet_row["risk_score"]),
                "feature": exp_row["feature"],
                "feature_value": exp_row["feature_value"],
                "shap_value": shap_value,
                "abs_shap_value": float(exp_row["abs_shap_value"]),
                "direction": direction,
            })

    local_shap_df = pd.DataFrame(local_rows)

    local_shap_df.to_csv(
        save_dir / "local_shap_explanations.csv",
        index=False
    )

    print(f"\nLocal SHAP results saved to: {save_dir}")

    return {
        "model": model,
        "metrics": metrics_df,
        "local_shap_explanations": local_shap_df,
        "save_dir": save_dir,
    }

# ---------------------------------------------------
# LIME - Random Forest
# ---------------------------------------------------
def run_lime_random_forest(
    dataset="elliptic-wallets-combined",
    num_samples_to_explain=10,
    num_features=10,
    random_state=42,
):
    """
    Train Random Forest balanced and explain individual predictions using LIME.

    Recommended dataset:
        - elliptic-wallets-combined

    Saves:
        - metrics.csv
        - lime_explanations.csv
        - lime_wallet_*.html

    Returns:
        dict with model, metrics, explanations, and save_dir.
    """

    from lime.lime_tabular import LimeTabularExplainer

    save_dir = _make_xai_feature_lvl_risk_dir() / f"random_forest_lime_{dataset}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_feature_xai_dataset(dataset)

    # 2. Train Random Forest balanced
    model = get_random_forest_balanced()
    model.fit(X_train, y_train)

    # 3. Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _evaluate(y_test, y_pred, y_proba)

    metrics_df = pd.DataFrame([{
        "dataset": dataset,
        "model": "Random Forest balanced",
        **metrics,
    }])

    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

    print("=== Random Forest Balanced Metrics ===")
    display(metrics_df)

    # 4. Create LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["licit", "illicit"],
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    # 5. Select wallets to explain
    X_explain = X_test.copy()
    X_explain["true_label"] = y_test.values
    X_explain["predicted_label"] = y_pred
    X_explain["fraud_probability"] = y_proba
    X_explain["risk_score"] = y_proba * 100

    # Explain highest-risk wallets first
    X_explain = X_explain.sort_values("risk_score", ascending=False)

    selected_rows = X_explain.head(num_samples_to_explain)

    lime_rows = []

    feature_cols = X_train.columns.tolist()

    for rank, (idx, row) in enumerate(selected_rows.iterrows(), start=1):
        wallet_features = row[feature_cols].values

        explanation = explainer.explain_instance(
            data_row=wallet_features,
            predict_fn=model.predict_proba,
            num_features=num_features,
            labels=[1],
        )

        explanation_list = explanation.as_list(label=1)

        html_path = save_dir / f"lime_wallet_rank_{rank}_index_{idx}.html"
        explanation.save_to_file(html_path)

        for feature_rule, importance in explanation_list:
            lime_rows.append({
                "wallet_index": idx,
                "rank_by_risk": rank,
                "true_label": int(row["true_label"]),
                "predicted_label": int(row["predicted_label"]),
                "fraud_probability": float(row["fraud_probability"]),
                "risk_score": float(row["risk_score"]),
                "feature_rule": feature_rule,
                "lime_importance": importance,
                "html_file": str(html_path),
            })

    lime_df = pd.DataFrame(lime_rows)
    lime_df.to_csv(save_dir / "lime_explanations.csv", index=False)

    print(f"\nLIME results saved to: {save_dir}")

    return {
        "model": model,
        "metrics": metrics_df,
        "lime_explanations": lime_df,
        "save_dir": save_dir,
    }

# ---------------------------------------------------
# Counterfactual Reasoning - Random Forest
# ---------------------------------------------------
def run_counterfactual_random_forest(
    dataset="elliptic-wallets-combined",
    num_wallets_to_explain=10,
    max_features_changed=5,
    candidate_features=None,
):
    """
    Train Random Forest balanced and generate simple counterfactual explanations.

    The goal is to find small feature changes that reduce a wallet's predicted
    fraud probability.

    Saves:
        - metrics.csv
        - counterfactual_explanations.csv
    """

    save_dir = _make_xai_feature_lvl_risk_dir() / f"random_forest_counterfactual_{dataset}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_feature_xai_dataset(dataset)

    # 2. Train Random Forest balanced
    model = get_random_forest_balanced()
    model.fit(X_train, y_train)

    # 3. Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = _evaluate(y_test, y_pred, y_proba)

    metrics_df = pd.DataFrame([{
        "dataset": dataset,
        "model": "Random Forest balanced",
        **metrics,
    }])

    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

    print("=== Random Forest Balanced Metrics ===")
    display(metrics_df)

    # 4. Prepare risk table
    risk_df = X_test.copy()
    risk_df["true_label"] = y_test.values
    risk_df["predicted_label"] = y_pred
    risk_df["fraud_probability"] = y_proba
    risk_df["risk_score"] = y_proba * 100

    risk_df = risk_df.sort_values("risk_score", ascending=False)

    selected_wallets = risk_df.head(num_wallets_to_explain)

    # 5. Use licit wallets as safer reference behaviour
    licit_train = X_train[y_train == 0]

    if candidate_features is None:
        candidate_features = X_train.columns.tolist()

    licit_medians = licit_train[candidate_features].median()

    counterfactual_rows = []

    # 6. Generate counterfactuals
    for rank, (wallet_index, row) in enumerate(selected_wallets.iterrows(), start=1):

        original_features = row[X_train.columns].copy()
        modified_features = original_features.copy()

        original_probability = float(row["fraud_probability"])
        current_probability = original_probability

        changes = []

        for step in range(1, max_features_changed + 1):

            best_feature = None
            best_new_value = None
            best_new_probability = current_probability
            best_reduction = 0

            for feature in candidate_features:

                if feature in [change["feature"] for change in changes]:
                    continue

                trial_features = modified_features.copy()
                trial_features[feature] = licit_medians[feature]

                trial_df = pd.DataFrame(
                    [trial_features.values],
                    columns=X_train.columns
                )

                trial_probability = float(model.predict_proba(trial_df)[:, 1][0])
                reduction = current_probability - trial_probability

                if reduction > best_reduction:
                    best_feature = feature
                    best_new_value = licit_medians[feature]
                    best_new_probability = trial_probability
                    best_reduction = reduction

            if best_feature is None:
                break

            old_value = modified_features[best_feature]
            modified_features[best_feature] = best_new_value
            current_probability = best_new_probability

            changes.append({
                "feature": best_feature,
                "old_value": old_value,
                "new_value": best_new_value,
                "new_probability": current_probability,
                "risk_reduction": best_reduction,
            })

        # 7. Save one row per feature change
        for change in changes:
            counterfactual_rows.append({
                "wallet_index": wallet_index,
                "rank_by_risk": rank,
                "true_label": int(row["true_label"]),
                "predicted_label": int(row["predicted_label"]),
                "original_fraud_probability": original_probability,
                "original_risk_score": original_probability * 100,
                "counterfactual_fraud_probability": change["new_probability"],
                "counterfactual_risk_score": change["new_probability"] * 100,
                "risk_score_reduction": (original_probability - change["new_probability"]) * 100,
                "feature_changed": change["feature"],
                "old_value": change["old_value"],
                "new_value": change["new_value"],
                "explanation": (
                    f"Changing {change['feature']} from {change['old_value']} "
                    f"to {change['new_value']} reduces the fraud probability "
                    f"from {original_probability:.3f} to {change['new_probability']:.3f}."
                )
            })

    counterfactual_df = pd.DataFrame(counterfactual_rows)

    counterfactual_df.to_csv(
        save_dir / "counterfactual_explanations.csv",
        index=False
    )

    print(f"\nCounterfactual results saved to: {save_dir}")

    return {
        "model": model,
        "metrics": metrics_df,
        "counterfactuals": counterfactual_df,
        "save_dir": save_dir,
    }