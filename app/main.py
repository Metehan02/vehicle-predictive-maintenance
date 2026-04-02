import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import yaml
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import io

from src.models.predict import predict, load_config

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Vehicle Predictive Maintenance API",
    description="Predicts APS failure in heavy vehicles using a tuned XGBoost model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

config = load_config(ROOT / "config.yaml")
model_output = ROOT / config["paths"]["model_output"]
model = joblib.load(model_output / "xgboost_tuned.pkl")
scaler = joblib.load(model_output / "scaler.pkl")
threshold = joblib.load(model_output / "threshold.pkl")
feature_cols = joblib.load(model_output / "feature_cols.pkl")


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int
    prediction_label: str
    threshold_used: float


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total_samples: int
    failures_detected: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "xgboost_tuned",
        "threshold": threshold,
        "n_features": len(feature_cols)
    }


@app.post("/predict", response_model=BatchPredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")),
            na_values=config["data"]["na_placeholder"]
        )

        if config["data"]["target_column"] in input_df.columns:
            input_df = input_df.drop(columns=[config["data"]["target_column"]])

        results_df = predict(input_df, config, ROOT)

        results = [
            PredictionResponse(
                failure_probability=float(row["failure_probability"]),
                prediction=int(row["prediction"]),
                prediction_label=row["prediction_label"],
                threshold_used=float(threshold)
            )
            for _, row in results_df.iterrows()
        ]

        return BatchPredictionResponse(
            results=results,
            total_samples=len(results),
            failures_detected=int(results_df["prediction"].sum())
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory=str(ROOT / "app" / "static"), html=True), name="static")