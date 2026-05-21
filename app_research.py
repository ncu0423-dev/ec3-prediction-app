from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import joblib
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from predict import predict_csv


# =====================================================
# Page config
# =====================================================

st.set_page_config(
    page_title="LLNA EC3 Research Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================
# Minimal CSS
# HTML wrapper は使わず、見た目の微調整だけに限定
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stExpander"] details summary p {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Paths
# =====================================================

MODEL_PATH = Path("model_research/ec3_xgb_bundle_research.joblib")
TEMPLATE_PATH = Path("model_research/input_template.csv")
SHAP_DIR = Path("model_research")
N_FOLDS = 5


# =====================================================
# Fold-wise explanations
# 既存の Global SHAP 画像を読み取って説明を付与
# Local SHAP は今回は実装しない
# =====================================================

fold_explanations = {
    1: {
        "top_features": [
            "hCLAT.MIT",
            "sum_invitro",
            "hCLAT.CD86.EC150..ug.ml.",
            "LUMO Energy",
            "potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS",
        ],
        "summary": (
            "Fold 1 では hCLAT.MIT が最上位で、sum_invitro と "
            "hCLAT.CD86.EC150..ug.ml. が続いています。細胞応答系の特徴量が強く、"
            "in vitro の総合指標も高い比重で使われている Fold です。"
        ),
        "interpretation": (
            "この Fold は hCLAT 系を中心に予測を組み立てつつ、LUMO/HOMO Energy や "
            "LogKM_pred などの物性・電子状態の情報も補助的に利用していると読めます。"
        ),
    },
    2: {
        "top_features": [
            "h.CLAT.CV75",
            "KS.EC1.5",
            "LogHL_pred",
            "KS.EC3",
            "alert_Protein_binding_by_OASIS",
        ],
        "summary": (
            "Fold 2 では h.CLAT.CV75 が最上位で、KS.EC1.5、LogHL_pred、KS.EC3 が "
            "上位に来ています。h-CLAT の細胞毒性閾値と KeratinoSens 関連指標が前面に出た構図です。"
        ),
        "interpretation": (
            "Fold 2 は細胞応答に加えて LogHL_pred や OASIS の protein-binding alert が見えており、"
            "毒性閾値・感作関連活性・物性の三つを併せて評価している Fold と考えられます。"
        ),
    },
    3: {
        "top_features": [
            "hCLAT.MIT",
            "KS.EC1.5",
            "h.CLAT.CV75",
            "LogKM_pred",
            "LUMO Energy",
        ],
        "summary": (
            "Fold 3 では hCLAT.MIT が再び最上位で、KS.EC1.5、h.CLAT.CV75、LogKM_pred が続きます。"
            "Fold 1 よりも KeratinoSens と輸送・物性寄りの情報が前に出ている点が特徴です。"
        ),
        "interpretation": (
            "この Fold は hCLAT 系の寄与を軸にしつつ、LogKM_pred や LUMO Energy を通じて "
            "化学物性・電子的性質をやや強めに参照している、バランス型の Fold と読めます。"
        ),
    },
    4: {
        "top_features": [
            "hCLAT.MIT",
            "KS.EC1.5",
            "potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS",
            "DPRA.percCysdep",
            "h.CLAT.CV75",
        ],
        "summary": (
            "Fold 4 では hCLAT.MIT と KS.EC1.5 が強く、さらに "
            "potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS と "
            "DPRA.percCysdep が比較的上位にあります。"
        ),
        "interpretation": (
            "Fold 4 は hCLAT / KeratinoSens に加え、タンパク反応性の手掛かりが他 Fold より見えやすく、"
            "in vitro と in chemico の両面から予測している Fold と解釈しやすいです。"
        ),
    },
    5: {
        "top_features": [
            "hCLAT.MIT",
            "h.CLAT.CV75",
            "LogHL_pred",
            "potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS",
            "OVERALL OH rate constant",
        ],
        "summary": (
            "Fold 5 では hCLAT.MIT が最上位で、h.CLAT.CV75、LogHL_pred、"
            "potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS が続きます。"
        ),
        "interpretation": (
            "Fold 5 は hCLAT 系の影響が安定して強いうえで、LogHL_pred、CombDipolPolariz、"
            "DPRA.score などの物性・反応性特徴も入っており、比較的多面的な説明になっています。"
        ),
    },
}


# =====================================================
# Helpers
# =====================================================

@st.cache_resource(show_spinner=False)
def load_bundle(model_path: str):
    return joblib.load(model_path)


def format_metric_value(value, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def load_image_safe(image_path: Path):
    try:
        with Image.open(image_path) as img:
            return img.copy(), None
    except FileNotFoundError:
        return None, f"ファイルが見つかりません: {image_path}"
    except UnidentifiedImageError:
        return None, f"画像として読み込めません: {image_path.name}"
    except OSError as e:
        return None, f"画像の読み込みに失敗しました: {e}"


# =====================================================
# Header
# =====================================================

st.title("🧪 LLNA EC3 Research Platform")
st.caption("XGBoost / Global SHAP / Applicability Domain / Fold Analysis")
st.caption("既存の Fold ごとの Global SHAP 画像を表示します。Local SHAP は今回は実装していません。")


# =====================================================
# Model loading
# =====================================================

if not MODEL_PATH.exists():
    st.error(f"モデルファイルが見つかりません: {MODEL_PATH}")
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

if not isinstance(bundle, dict):
    st.error("モデルバンドルの形式が想定と異なります。dict 形式を想定しています。")
    st.stop()

metrics = bundle.get("metrics", {})


# =====================================================
# Metrics
# =====================================================

st.header("モデルのパフォーマンス")

m1, m2, m3, m4 = st.columns(4, gap="small")

with m1:
    st.metric(
        label="特徴量数",
        value=format_metric_value(metrics.get("n_features"), decimals=0),
        border=True,
    )

with m2:
    st.metric(
        label="学習データ数",
        value=format_metric_value(metrics.get("n_train"), decimals=0),
        border=True,
    )

with m3:
    st.metric(
        label="CV RMSE",
        value=format_metric_value(metrics.get("cv_rmse_y")),
        border=True,
    )

with m4:
    st.metric(
        label="CV R²",
        value=format_metric_value(metrics.get("cv_r2_y")),
        border=True,
    )


# =====================================================
# Template / Upload
# =====================================================

left_top, right_top = st.columns(2, gap="small")

with left_top:
    with st.container(border=True):
        st.subheader("📥 入力テンプレート")
        st.caption("予測用 CSV の列構成を確認するためのテンプレートです。")

        if TEMPLATE_PATH.exists():
            template_bytes = TEMPLATE_PATH.read_bytes()
            st.download_button(
                label="入力テンプレートをダウンロード",
                data=template_bytes,
                file_name="input_template.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning(f"テンプレートが見つかりません: {TEMPLATE_PATH}")

with right_top:
    with st.container(border=True):
        st.subheader("📂 CSVアップロード")
        st.caption("テンプレートと同じ列構成の descriptor CSV をアップロードしてください。")

        uploaded = st.file_uploader(
            label="予測したい descriptor CSV を選択してください。",
            type=["csv"],
            help="1ファイルの CSV をアップロードします。列名はテンプレートに揃えてください。",
        )


# =====================================================
# Prediction
# =====================================================

if uploaded is not None:
    input_df = None
    result_df = None

    try:
        input_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV の読み込みに失敗しました: {e}")

    if input_df is not None:
        with st.container(border=True):
            st.subheader("📋 入力データ")
            st.dataframe(
                input_df,
                use_container_width=True,
                height=320,
            )

        tmp_path = None

        try:
            with st.spinner("予測を実行しています..."):
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    input_df.to_csv(tmp_path, index=False)

                result_df = predict_csv(str(tmp_path), str(MODEL_PATH))

        except Exception as e:
            st.error(f"予測処理に失敗しました: {e}")

        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        if result_df is not None:
            with st.container(border=True):
                st.subheader("🧠 予測結果")
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    height=320,
                )

                output_buffer = BytesIO()
                result_df.to_csv(
                    output_buffer,
                    index=False,
                    encoding="utf-8-sig",
                )

                st.download_button(
                    label="⬇ 予測結果 CSV をダウンロード",
                    data=output_buffer.getvalue(),
                    file_name="prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# =====================================================
# SHAP overview
# =====================================================

st.header("Fold ごとの Global SHAP")

with st.container(border=True):
    st.subheader("SHAP 図の読み方")
    st.markdown(
        """
- この欄で表示しているのは、学習時に保存した **Fold ごとの Global SHAP 画像** です。
- 画像は既存 PNG をそのまま読み込んでいるため、**アップロードする CSV によってこの SHAP 画像自体は変わりません**。
- 上位に並ぶ特徴量ほど、その Fold のモデル全体で重要度が高いと読めます。
- 横軸の SHAP value は予測への寄与の大きさと方向を表します。
- 一般に、**赤は特徴量値が高い側、青は低い側** を示します。
- 右方向は EC3 予測を押し上げる寄与、左方向は押し下げる寄与として解釈します。
        """
    )


# =====================================================
# Fold-wise SHAP cards
# =====================================================

missing_shap_files = []

for fold in range(1, N_FOLDS + 1):
    shap_path = SHAP_DIR / f"shap_summary_fold_{fold}.png"
    explanation = fold_explanations.get(fold)

    with st.container(border=True):
        st.subheader(f"Fold {fold}")

        left, right = st.columns([1.15, 1.35], gap="large")

        with left:
            if explanation is not None:
                st.markdown("**上位に現れている特徴量**")
                st.markdown(
                    "\n".join([f"- {feature}" for feature in explanation["top_features"]])
                )

                st.markdown("**Fold ごとの読み取り**")
                st.write(explanation["summary"])

                st.markdown("**解釈**")
                st.write(explanation["interpretation"])
            else:
                st.info("この Fold の説明文はまだ登録されていません。")

        with right:
            image, image_error = load_image_safe(shap_path)

            if image_error is not None:
                st.warning(image_error)
                missing_shap_files.append(str(shap_path))
            else:
                st.image(
                    image,
                    caption=f"Fold {fold} Global SHAP プレビュー",
                    use_container_width=True,
                )

        if image_error is None:
            with st.expander(f"🔍 Fold {fold} の SHAP 画像を拡大表示", expanded=False):
                st.image(
                    image,
                    caption=f"Fold {fold} Global SHAP 拡大表示",
                    use_container_width=True,
                )
                st.caption("見出しをもう一度クリックすると折りたたまれ、縮小表示に戻ります。")


# =====================================================
# Missing SHAP files warning
# =====================================================

if missing_shap_files:
    with st.container(border=True):
        st.subheader("不足している SHAP 画像")
        st.warning("以下の SHAP 画像が見つからないため、該当 Fold の画像表示をスキップしました。")
        st.markdown("\n".join([f"- {path}" for path in missing_shap_files]))


# =====================================================
# Footer
# =====================================================

st.caption("LLNA EC3 Research Platform / XGBoost / Global SHAP / Applicability Domain")
