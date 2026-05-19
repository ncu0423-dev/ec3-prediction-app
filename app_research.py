from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from predict import predict_csv


# =====================================================
# 設定
# =====================================================

st.set_page_config(
    page_title="LLNA EC3 Research App",
    layout="wide"
)


# =====================================================
# パス
# =====================================================

MODEL_PATH = Path(
    "model_research/ec3_xgb_bundle_research.joblib"
)

TEMPLATE_PATH = Path(
    "model_research/input_template.csv"
)


# =====================================================
# タイトル
# =====================================================

st.title("LLNA EC3 Research Model")

st.caption(
    "Fold保存型研究モデル"
)


# =====================================================
# モデル存在確認
# =====================================================

if not MODEL_PATH.exists():

    st.error(
        "research model が見つかりません"
    )

    st.stop()


# =====================================================
# load
# =====================================================

bundle = joblib.load(MODEL_PATH)

metrics = bundle["metrics"]

cv_fold_data = bundle["cv_fold_data"]


# =====================================================
# metrics
# =====================================================

with st.expander("Model Metrics"):

    st.write(metrics)


# =====================================================
# template
# =====================================================

if TEMPLATE_PATH.exists():

    template_bytes = TEMPLATE_PATH.read_bytes()

    st.download_button(
        "input template download",
        data=template_bytes,
        file_name="input_template.csv",
        mime="text/csv",
    )


# =====================================================
# upload
# =====================================================

uploaded = st.file_uploader(
    "CSV Upload",
    type=["csv"]
)


# =====================================================
# prediction
# =====================================================

if uploaded is not None:

    try:

        input_df = pd.read_csv(uploaded)

        st.subheader("Input Data")

        st.dataframe(
            input_df,
            use_container_width=True
        )

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

        st.subheader("Prediction Results")

        st.dataframe(
            result,
            use_container_width=True
        )

        output = BytesIO()

        result.to_csv(
            output,
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            "prediction csv download",
            data=output.getvalue(),
            file_name="prediction_results.csv",
            mime="text/csv",
        )

    except Exception as e:

        st.error(e)


# =====================================================
# Fold SHAP
# =====================================================

st.header("Fold SHAP Summary")

for i in range(1, 6):

    shap_path = Path(
        f"model_research/shap_summary_fold_{i}.png"
    )

    if shap_path.exists():

        st.subheader(f"Fold {i}")

        image = Image.open(shap_path)

        st.image(
            image,
            use_container_width=True
        )