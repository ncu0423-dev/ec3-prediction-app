"""Research-version EC3 model with fold storage and SHAP.

Usage:
    python train_model_research.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# =========================================================
# 基本設定
# =========================================================

TARGET_COL = "LLNA_EC3"
MW_COL = "MolWeight"

META_COLS = [
    "No.",
    "Name",
    "CAS Number",
    "SMILES",
    "CAS_Number_qsar",
    "SMILES_qsar"
]


# =========================================================
# 使用する23特徴量
# =========================================================

FEATURE_COLUMNS = [
    'OVERALL OH rate constant',
    'FM advection air',
    'VdW surface DPSA2',
    'LUMO Energy',
    'HOMO Energy',
    'MolWeight',
    'CombDipolPolariz',
    'LogHL_pred',
    'LogKM_pred',
    'alert_Protein_binding_by_OASIS',
    'potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS',
    'DPRA.percCysdep',
    'DPRA.percLysdep',
    'DPRA.score',
    'hCLAT.CD86.EC150..ug.ml.',
    'hCLAT.CD54.EC200..ug.ml.',
    'h.CLAT.CV75',
    'hCLAT.MIT',
    'hCLAT.score',
    'KS.EC1.5',
    'KS.EC3',
    'KS.IC50',
    'sum_invitro'
]


# =========================================================
# EC3変換
# =========================================================

def ec3_to_y(
    ec3: pd.Series | np.ndarray,
    mw: pd.Series | np.ndarray
) -> np.ndarray:

    return np.log10(
        np.asarray(mw, dtype=float) /
        np.asarray(ec3, dtype=float)
    )


def y_to_ec3(
    y: np.ndarray,
    mw: pd.Series | np.ndarray
) -> np.ndarray:

    return np.asarray(mw, dtype=float) / (
        10 ** np.asarray(y, dtype=float)
    )


# =========================================================
# Stratified CV
# =========================================================

def stratify_ec3(ec3: pd.Series) -> np.ndarray:

    labels = []

    for value in ec3:

        if value == 150:
            labels.append(0)

        elif value <= 2:
            labels.append(2)

        else:
            labels.append(1)

    return np.asarray(labels)


# =========================================================
# モデル構築
# =========================================================

def build_model(random_state: int) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "imputer",
                KNNImputer(n_neighbors=5)
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    n_estimators=150,
                    random_state=random_state,
                    n_jobs=-1,
                )
            ),
        ]
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_csv",
        default="data/DASS_join_data_opera_delete_LLNAnan.csv"
    )

    parser.add_argument(
        "--model_dir",
        default="model_research"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=0
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5
    )

    parser.add_argument(
        "--knn_k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--ad_coverage",
        type=float,
        default=0.95
    )

    args = parser.parse_args()


    # =====================================================
    # 読み込み
    # =====================================================

    train_csv = Path(args.train_csv)

    model_dir = Path(args.model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        train_csv,
        encoding="cp932"
    )

    df = df.dropna(
        subset=[TARGET_COL, MW_COL]
    ).reset_index(drop=True)


    # =====================================================
    # 特徴量
    # =====================================================

    X = df[FEATURE_COLUMNS]

    y = ec3_to_y(
        df[TARGET_COL],
        df[MW_COL]
    )

    strata = stratify_ec3(df[TARGET_COL])


    # =====================================================
    # CV
    # =====================================================

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    pred_y_cv = np.zeros(len(y))

    cv_fold_data = {}


    # =====================================================
    # Fold loop
    # =====================================================

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(X, strata),
        start=1
    ):

        print(f"========== Fold {fold_idx} ==========")

        fold_name = f"fold_{fold_idx}"

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline = build_model(args.random_state)

        pipeline.fit(
            X_train,
            y_train
        )

        pred_y = pipeline.predict(X_test)

        pred_y_cv[test_idx] = pred_y


        # =================================================
        # SHAP
        # =================================================

        transformed_X_train = pipeline.named_steps[
            "scaler"
        ].transform(

            pipeline.named_steps[
                "imputer"
            ].transform(X_train)

        )

        explainer = shap.TreeExplainer(
            pipeline.named_steps["model"]
        )

        shap_values = explainer.shap_values(
            transformed_X_train
        )

        plt.figure(figsize=(12, 8))

        shap.summary_plot(
            shap_values,
            transformed_X_train,
            feature_names=FEATURE_COLUMNS,
            plot_type="violin",
            show=False
        )

        plt.title(f"SHAP Summary Fold {fold_idx}")

        plt.savefig(
            model_dir / f"shap_summary_fold_{fold_idx}.png",
            bbox_inches="tight"
        )

        plt.close()


        # =================================================
        # fold保存
        # =================================================

        cv_fold_data[fold_name] = {

            "Studied_model": pipeline.named_steps["model"],

            "Pipeline": pipeline,

            "Train_data": {
                "X_data": X_train,
                "y_data": y_train,
            },

            "Test_data": {
                "X_data": X_test,
                "y_data": y_test,
            }
        }


    # =====================================================
    # CV Metrics
    # =====================================================

    pred_ec3_cv = y_to_ec3(
        pred_y_cv,
        df[MW_COL]
    )

    metrics = {

        "cv_r2_y":
            float(r2_score(y, pred_y_cv)),

        "cv_rmse_y":
            float(mean_squared_error(y, pred_y_cv) ** 0.5),

        "cv_rmse_ec3":
            float(
                mean_squared_error(
                    df[TARGET_COL],
                    pred_ec3_cv
                ) ** 0.5
            ),

        "n_train":
            int(len(df)),

        "n_features":
            int(len(FEATURE_COLUMNS)),

        "target_transform":
            "y = log10(MolWeight / LLNA_EC3)"
    }


    # =====================================================
    # Final model
    # =====================================================

    final_pipeline = build_model(args.random_state)

    final_pipeline.fit(X, y)

    transformed_X = final_pipeline.named_steps[
        "scaler"
    ].transform(

        final_pipeline.named_steps[
            "imputer"
        ].transform(X)

    )


    # =====================================================
    # AD
    # =====================================================

    d_train = cdist(
        transformed_X,
        transformed_X
    )

    np.fill_diagonal(
        d_train,
        np.inf
    )

    kth_mean = np.sort(
        d_train,
        axis=1
    )[:, : args.knn_k].mean(axis=1)

    ad_threshold = float(
        np.quantile(
            kth_mean,
            args.ad_coverage
        )
    )


    # =====================================================
    # 保存
    # =====================================================

    bundle = {

        "pipeline":
            final_pipeline,

        "cv_fold_data":
            cv_fold_data,

        "feature_columns":
            FEATURE_COLUMNS,

        "meta_columns":
            META_COLS,

        "target_col":
            TARGET_COL,

        "mw_col":
            MW_COL,

        "ad_training_matrix":
            transformed_X,

        "ad_threshold":
            ad_threshold,

        "metrics":
            metrics,
    }

    joblib.dump(
        bundle,
        model_dir / "ec3_xgb_bundle_research.joblib"
    )


    # =====================================================
    # metrics
    # =====================================================

    with open(
        model_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics | {"ad_threshold": ad_threshold},
            f,
            ensure_ascii=False,
            indent=2
        )


    # =====================================================
    # template
    # =====================================================

    template = df[
        META_COLS + FEATURE_COLUMNS
    ].head(3)

    template.to_csv(
        model_dir / "input_template.csv",
        index=False,
        encoding="utf-8-sig"
    )


    # =====================================================
    # print
    # =====================================================

    print(
        "Saved:",
        model_dir / "ec3_xgb_bundle_research.joblib"
    )

    print(
        json.dumps(
            metrics | {"ad_threshold": ad_threshold},
            ensure_ascii=False,
            indent=2
        )
    )


# =========================================================
# 実行
# =========================================================

if __name__ == "__main__":
    main()