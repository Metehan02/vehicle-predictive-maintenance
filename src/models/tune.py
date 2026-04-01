import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
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
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def tune_xgboost(X_train, y_train, random_state, cv_folds):
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3],
    }

    scorer = make_scorer(f1_score, pos_label=1)

    xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=30,
        scoring=scorer,
        cv=cv_folds,
        verbose=2,
        random_state=random_state,
        n_jobs=-1
    )

    print("Starting RandomizedSearchCV — this will take several minutes...")
    search.fit(X_train, y_train)

    print(f"\nBest F1 score (CV): {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    return search.best_estimator_, search.best_params_


def run(root=Path(".")):
    config = load_config(root / "config.yaml")
    target_col = config["data"]["target_column"]
    random_state = config["model"]["random_state"]
    cv_folds = config["model"]["cv_folds"]
    model_output = root / config["paths"]["model_output"]

    print("Loading processed data...")
    train_df, test_df = load_processed_data(config, root)

    X_train, y_train = split_features_target(train_df, target_col)
    X_test, y_test = split_features_target(test_df, target_col)

    print("Applying SMOTE...")
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state)

    best_model, best_params = tune_xgboost(X_train_res, y_train_res, random_state, cv_folds)

    print("\nEvaluating best model on test set...")
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["neg", "pos"]))
    print(f"ROC-AUC:        {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision:  {average_precision_score(y_test, y_prob):.4f}")

    print("\nSaving tuned model...")
    joblib.dump(best_model, model_output / "xgboost_tuned.pkl")
    print("Saved xgboost_tuned.pkl")

    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(root / "reports/best_params.csv", index=False)
    print("Saved best_params.csv")


if __name__ == "__main__":
    run(root=Path("."))