"""Predict LLNA EC3 from a CSV containing the same feature columns used in training."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def y_to_ec3(y: np.ndarray, mw: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(mw, dtype=float) / (10 ** np.asarray(y, dtype=float))


def ecetoc_class(ec3: float) -> str:
    if pd.isna(ec3):
        return "Unknown"
    if ec3 <= 0.1:
        return "Extreme"
    if ec3 <= 1:
        return "Strong"
    if ec3 <= 10:
        return "Moderate"
    if ec3 <= 100:
        return "Weak"
    return "Non-sensitizer"


def ghs_class(ec3: float) -> str:
    if pd.isna(ec3):
        return "Unknown"
    if ec3 <= 2:
        return "GHS 1A"
    if ec3 <= 150:
        return "GHS 1B"
    return "Not classified"


def predict_csv(input_csv: str | Path, model_path: str | Path = "model/ec3_xgb_bundle.joblib") -> pd.DataFrame:
    bundle = joblib.load(model_path)
    df = pd.read_csv(input_csv)

    feature_columns = bundle["feature_columns"]
    mw_col = bundle["mw_col"]
    missing = [c for c in feature_columns + [mw_col] if c not in df.columns]
    if missing:
        raise ValueError(
            "入力CSVに必要な列が不足しています: " + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    X = df[feature_columns]
    pred_y = bundle["pipeline"].predict(X)
    pred_ec3 = y_to_ec3(pred_y, df[mw_col])

    imputed = bundle["pipeline"].named_steps["imputer"].transform(X)
    scaled = bundle["pipeline"].named_steps["scaler"].transform(imputed)
    distances = cdist(scaled, bundle["ad_training_matrix"])
    knn_mean_distance = np.sort(distances, axis=1)[:, : bundle["knn_k"]].mean(axis=1)
    in_ad = knn_mean_distance <= bundle["ad_threshold"]

    meta_cols = [c for c in bundle["meta_columns"] if c in df.columns]
    result = df[meta_cols].copy()
    result["pred_y_log10_MW_over_EC3"] = pred_y
    result["pred_LLNA_EC3"] = pred_ec3
    result["ECETOC_class"] = [ecetoc_class(v) for v in pred_ec3]
    result["GHS_class"] = [ghs_class(v) for v in pred_ec3]
    result["AD_mean_knn_distance"] = knn_mean_distance
    result["AD_threshold"] = bundle["ad_threshold"]
    result["in_applicability_domain"] = in_ad
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--model_path", default="model/ec3_xgb_bundle.joblib")
    parser.add_argument("--output_csv", default="prediction_results.csv")
    args = parser.parse_args()

    result = predict_csv(args.input_csv, args.model_path)
    result.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()
