from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from predict import predict_csv


# =========================
# Streamlit設定
# =========================

st.set_page_config(
    page_title="LLNA EC3 Prediction App",
    layout="wide"
)


# =========================
# パス設定
# =========================

MODEL_PATH = Path("model/ec3_xgb_bundle.joblib")
TEMPLATE_PATH = Path("model/input_template.csv")
SHAP_PATH = Path("model/shap_summary.png")


# =========================
# タイトル
# =========================

st.title("LLNA EC3 予測アプリ")

st.caption(
    "DASSデータで学習したXGBoostモデルを使い、CSV内の化学物質のLLNA EC3を予測します。"
)


# =========================
# モデル存在確認
# =========================

if not MODEL_PATH.exists():

    st.error(
        "モデルファイルが見つかりません。先に `python train_model.py` を実行してください。"
    )

    st.stop()


# =========================
# モデル読み込み
# =========================

bundle = joblib.load(MODEL_PATH)

metrics = bundle.get("metrics", {})


# =========================
# モデル情報
# =========================

with st.expander("モデル情報", expanded=False):

    st.write(f"特徴量数: {metrics.get('n_features', 'unknown')}")

    st.write(f"学習データ数: {metrics.get('n_train', 'unknown')}")

    st.write(f"CV R²: {metrics.get('cv_r2_y', 'unknown')}")

    st.write(f"CV RMSE: {metrics.get('cv_rmse_y', 'unknown')}")

    st.write("目的変数変換: y = log10(MolWeight / LLNA_EC3)")


# =========================
# テンプレートDL
# =========================

if TEMPLATE_PATH.exists():

    template_bytes = TEMPLATE_PATH.read_bytes()

    st.download_button(
        "入力CSVテンプレートをダウンロード",
        data=template_bytes,
        file_name="input_template.csv",
        mime="text/csv",
    )


# =========================
# CSVアップロード
# =========================

uploaded = st.file_uploader(
    "予測したいCSVをアップロードしてください",
    type=["csv"]
)


# =========================
# 予測
# =========================

if uploaded is not None:

    try:

        # -------------------------
        # 入力データ読み込み
        # -------------------------

        input_df = pd.read_csv(uploaded)

        st.subheader("入力データ")

        st.dataframe(
            input_df.head(20),
            use_container_width=True
        )


        # -------------------------
        # 一時CSV保存
        # -------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False
        ) as tmp:

            input_df.to_csv(
                tmp.name,
                index=False
            )

            result = predict_csv(
                tmp.name,
                MODEL_PATH
            )


        # -------------------------
        # 予測結果
        # -------------------------

        st.subheader("予測結果")

        st.dataframe(
            result,
            use_container_width=True
        )


        # -------------------------
        # CSVダウンロード
        # -------------------------

        output = BytesIO()

        result.to_csv(
            output,
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            "予測結果CSVをダウンロード",
            data=output.getvalue(),
            file_name="prediction_results.csv",
            mime="text/csv",
        )


        # =========================
        # SHAP Summary Plot
        # =========================

        st.subheader("SHAP Summary Plot")


        if SHAP_PATH.exists():

            image = Image.open(SHAP_PATH)

            st.image(
                image,
                caption="SHAP Summary Plot",
                use_container_width=True
            )

        else:

            st.warning(
                "SHAP画像が見つかりません。先に train_model.py を実行してください。"
            )


    except Exception as e:

        st.error(f"予測に失敗しました: {e}")

        st.info(
            "テンプレートと同じ特徴量列、特に MolWeight が入っているか確認してください。"
        )