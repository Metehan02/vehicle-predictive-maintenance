import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(config, root=Path(".")):
    train_df = pd.read_csv(root / config["paths"]["processed_train"])
    test_df = pd.read_csv(root / config["paths"]["processed_test"])
    return train_df, test_df


def split_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def apply_smote(X_train, y_train, random_state):
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled


def get_models(random_state):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
    }


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["neg", "pos"]))
    print(f"ROC-AUC:          {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision:    {average_precision_score(y_test, y_prob):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return {
        "model_name": model_name,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }


def run(root=Path(".")):
    config = load_config(root / "config.yaml")
    target_col = config["data"]["target_column"]
    random_state = config["model"]["random_state"]
    model_output = root / config["paths"]["model_output"]

    print("Loading processed data...")
    train_df, test_df = load_processed_data(config, root)

    X_train, y_train = split_features_target(train_df, target_col)
    X_test, y_test = split_features_target(test_df, target_col)

    print("\nApplying SMOTE...")
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state)

    models = get_models(random_state)
    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_res, y_train_res)

        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)

        joblib.dump(model, model_output / f"{model_name}.pkl")
        print(f"Saved {model_name}.pkl")

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n\nModel Comparison:")
    print(results_df.to_string(index=False))
    results_df.to_csv(root / "reports/model_comparison.csv", index=False)
    print("\nResults saved to reports/model_comparison.csv")


if __name__ == "__main__":
    run(root=Path("."))