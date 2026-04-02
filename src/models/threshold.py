import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score


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


def find_optimal_threshold(model, X_test, y_test, output_path):
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        precision_scores.append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, pos_label=1, zero_division=0))

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"F1 at optimal threshold: {best_f1:.4f}")
    print(f"Precision at optimal threshold: {precision_scores[best_idx]:.4f}")
    print(f"Recall at optimal threshold: {recall_scores[best_idx]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1_scores, label="F1", color="steelblue")
    ax.plot(thresholds, precision_scores, label="Precision", color="tomato")
    ax.plot(thresholds, recall_scores, label="Recall", color="seagreen")
    ax.axvline(best_threshold, color="black", linestyle="--", label=f"Optimal threshold = {best_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Optimization — Precision, Recall, F1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved threshold plot to {output_path}")

    return float(best_threshold)


def run(root=Path(".")):
    config = load_config(root / "config.yaml")
    target_col = config["data"]["target_column"]
    model_output = root / config["paths"]["model_output"]
    reports_path = root / "reports"

    print("Loading test data...")
    test_df = load_processed_data(config, root)
    X_test, y_test = split_features_target(test_df, target_col)

    print("Loading tuned model...")
    model = joblib.load(model_output / "xgboost_tuned.pkl")

    print("Finding optimal threshold...")
    best_threshold = find_optimal_threshold(
        model, X_test, y_test,
        reports_path / "threshold_optimization.png"
    )

    threshold_path = model_output / "threshold.pkl"
    joblib.dump(best_threshold, threshold_path)
    print(f"Saved optimal threshold ({best_threshold:.2f}) to {threshold_path}")


if __name__ == "__main__":
    run(root=Path("."))