import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_artifacts(model_output):
    model = joblib.load(model_output / "xgboost_tuned.pkl")
    scaler = joblib.load(model_output / "scaler.pkl")
    threshold = joblib.load(model_output / "threshold.pkl")
    feature_cols = joblib.load(model_output / "feature_cols.pkl")
    return model, scaler, threshold, feature_cols


def preprocess_input(df, scaler, feature_cols, na_placeholder):
    df = df.replace(na_placeholder, np.nan)
    df = df.reindex(columns=feature_cols)
    df = df.fillna(df.median())
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_cols)
    return df_scaled


def predict(input_df, config, root=Path(".")):
    model_output = root / config["paths"]["model_output"]
    na_placeholder = config["data"]["na_placeholder"]

    model, scaler, threshold, feature_cols = load_artifacts(model_output)
    X = preprocess_input(input_df, scaler, feature_cols, na_placeholder)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    results = pd.DataFrame({
        "failure_probability": y_prob.round(4),
        "prediction": y_pred,
        "prediction_label": ["pos" if p == 1 else "neg" for p in y_pred]
    })

    return results


if __name__ == "__main__":
    root = Path(".")
    config = load_config(root / "config.yaml")

    print("Loading test data as sample input...")
    sample = pd.read_csv(
        root / config["paths"]["raw_test"],
        na_values=config["data"]["na_placeholder"],
        skiprows=20
    ).head(10).drop(columns=[config["data"]["target_column"]], errors="ignore")

    results = predict(sample, config, root)
    print("\nPredictions on 10 sample rows:")
    print(results)