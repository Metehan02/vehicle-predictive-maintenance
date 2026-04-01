import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(config, root=Path(".")):
    train_df = pd.read_csv(
        root / config["paths"]["raw_train"],
        na_values=config["data"]["na_placeholder"],
        skiprows=20
    )
    test_df = pd.read_csv(
        root / config["paths"]["raw_test"],
        na_values=config["data"]["na_placeholder"],
        skiprows=20
    )
    return train_df, test_df

def drop_high_missing_columns(train_df, test_df, threshold):
    missing_pct = train_df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    print(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values")
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)
    return train_df, test_df

def impute_missing(train_df, test_df, target_col):
    feature_cols = [c for c in train_df.columns if c != target_col]
    medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(medians)
    return train_df, test_df

def encode_target(train_df, test_df, target_col, positive_class):
    train_df[target_col] = (train_df[target_col] == positive_class).astype(int)
    test_df[target_col] = (test_df[target_col] == positive_class).astype(int)
    return train_df, test_df

def scale_features(train_df, test_df, target_col, scaler_path):
    feature_cols = [c for c in train_df.columns if c != target_col]
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    return train_df, test_df

def run(root=Path(".")):
    config = load_config(root / "config.yaml")

    target_col = config["data"]["target_column"]
    positive_class = config["data"]["positive_class"]
    threshold = config["data"]["missing_threshold"]
    scaler_path = root / config["paths"]["model_output"] / "scaler.pkl"

    print("Loading data...")
    train_df, test_df = load_data(config, root)

    print("Dropping high-missing columns...")
    train_df, test_df = drop_high_missing_columns(train_df, test_df, threshold)

    print("Imputing missing values...")
    train_df, test_df = impute_missing(train_df, test_df, target_col)

    print("Encoding target...")
    train_df, test_df = encode_target(train_df, test_df, target_col, positive_class)

    print("Scaling features...")
    train_df, test_df = scale_features(train_df, test_df, target_col, scaler_path)

    print("Saving processed data...")
    train_df.to_csv(root / config["paths"]["processed_train"], index=False)
    test_df.to_csv(root / config["paths"]["processed_test"], index=False)

    print(f"Done. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

if __name__ == "__main__":
    run(root=Path("."))