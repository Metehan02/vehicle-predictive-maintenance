import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
import joblib
from pathlib import Path

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    classification_report
)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(config, root=Path(".")):
    test_df = pd.read_csv(root / config["paths"]["processed_test"])
    return test_df


def split_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def plot_roc_curve(model, X_test, y_test, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="XGBoost (tuned)")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title("ROC Curve — Tuned XGBoost")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved ROC curve to {output_path}")


def plot_precision_recall_curve(model, X_test, y_test, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name="XGBoost (tuned)")
    ax.set_title("Precision-Recall Curve — Tuned XGBoost")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved PR curve to {output_path}")


def plot_confusion_matrix(y_test, y_pred, output_path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg", "pos"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix at Optimal Threshold (0.68)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved confusion matrix to {output_path}")


def plot_feature_importance(model, feature_names, output_path, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices][::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_title(f"Top {top_n} Feature Importances — Tuned XGBoost")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved feature importance to {output_path}")


def run(root=Path(".")):
    config = load_config(root / "config.yaml")
    target_col = config["data"]["target_column"]
    model_output = root / config["paths"]["model_output"]
    reports_path = root / "reports"

    print("Loading test data...")
    test_df = load_processed_data(config, root)
    X_test, y_test = split_features_target(test_df, target_col)

    print("Loading tuned model and threshold...")
    model = joblib.load(model_output / "xgboost_tuned.pkl")
    threshold = joblib.load(model_output / "threshold.pkl")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nEvaluation at optimal threshold ({threshold:.2f}):")
    print(classification_report(y_test, y_pred, target_names=["neg", "pos"]))
    print(f"ROC-AUC:        {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision:  {average_precision_score(y_test, y_prob):.4f}")

    print("\nGenerating plots...")
    plot_roc_curve(model, X_test, y_test, reports_path / "roc_curve.png")
    plot_precision_recall_curve(model, X_test, y_test, reports_path / "pr_curve.png")
    plot_confusion_matrix(y_test, y_pred, reports_path / "confusion_matrix.png")
    plot_feature_importance(model, list(X_test.columns), reports_path / "feature_importance.png")

    print("\nAll evaluation plots saved to reports/")


if __name__ == "__main__":
    run(root=Path("."))