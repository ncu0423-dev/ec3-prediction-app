"""Streamlit interface. No model training or upload persistence occurs here."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from original_runtime import explain_prediction, load_artifact, predict_with_artifact, prepare_input_features
from utils_io import FIXED_FEATURE_SETS, MW_COL, read_csv_auto

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = Path(os.environ.get("EC3_ARTIFACT_PATH", os.environ.get("ARTIFACT_PATH", str(ROOT / "artifacts/original_model_artifact.joblib")))).expanduser()
if not ARTIFACT_PATH.is_absolute():
    ARTIFACT_PATH = ROOT / ARTIFACT_PATH
PAGES = ["ホーム", "単一物質の予測", "CSV一括予測", "モデルの性能評価", "SHAPによる説明"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_ROWS = 10_000


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read bytes in memory, retaining headers, identifiers, and row order."""
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    if not isinstance(raw, bytes):
        raise ValueError("CSVを読み込めません。ファイルを選び直してください。")
    if not raw:
        raise ValueError("CSVが空です。列名と1行以上のデータが必要です。")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("CSVの上限は20 MBです。ファイルを分割してください。")
    frame, _ = read_csv_auto(raw)
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError("上限は10,000行です。CSVを分割してください。")
    return frame


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


@st.cache_resource(show_spinner=False)
def _load_model(path: str, modified_ns: int) -> dict:
    return load_artifact(path)


def _folds(artifact: dict) -> list[dict]:
    raw = artifact["folds"]
    if isinstance(raw, dict):
        return [dict(value, fold_name=value.get("fold_name", name)) for name, value in raw.items()]
    return [dict(value, fold_name=value.get("fold_name", f"fold_{i}")) for i, value in enumerate(raw, 1)]


def _missing_model() -> None:
    st.error("利用できる学習済みモデルがありません。予測・評価・SHAPには、学習処理で保存したモデルが必要です。")
    st.write("管理者がREADMEの学習手順に従ってモデルを作成してください。学習条件は次のコマンドで確認できます。")
    st.code("python train.py --help", language="bash")
    st.caption("既定の保存先: artifacts/original_model_artifact.joblib。画面を開くだけで再学習は行いません。")


def _training_label(artifact: dict) -> str:
    mode = artifact.get("training_mode", artifact.get("metadata", {}).get("training_mode"))
    if artifact.get('metadata', {}).get('feature_mode') in FIXED_FEATURE_SETS:
        label = f"指定{len(artifact['full_feature_columns'])}項目モデル"
        return label + '（軽量動作確認用）' if mode == 'smoke' else label
    return {"notebook": "Notebook再現条件", "smoke": "軽量動作確認用"}.get(mode, "作成条件を要確認")


def _is_notebook_workflow(artifact: dict) -> bool:
    return artifact.get("metadata", {}).get("feature_mode", "boruta") not in FIXED_FEATURE_SETS


def _selected_features(artifact: dict) -> list[str]:
    return list(artifact["selected_feature_columns"])


def _notebook_shap_importance(values) -> np.ndarray:
    # Literal notebook operation: sum rows first (preserve dtype/order), then abs.
    return abs(sum(np.asarray(values)))


def _invitro_ad_note(artifact: dict) -> None:
    if _is_notebook_workflow(artifact) and not artifact.get("metadata", {}).get("invitro_ad_available", False):
        st.info("hCLAT.Callの元資料が未提供のため、in vitro試験のcallと組み合わせた追加ADは未計算です。kNNによるADは計算できます。hCLAT.scoreから推測した判定は使いません。")


def _home(artifact: dict | None) -> None:
    st.subheader("化学物質のデータから、皮膚感作性を調べる")
    st.write("in vitro試験や物性値を入力すると、XGBoostモデルがLLNA EC3を予測します。予測値と適用範囲（AD）を確認し、SHAPで各特徴量の寄与を調べられます。")
    left, right = st.columns(2)
    with left:
        st.subheader("LLNA EC3とは")
        st.write("局所リンパ節試験（LLNA）で、刺激指数が3に達する推定濃度です。単位は%で、値が小さいほど、より低い濃度で皮膚感作反応を起こす可能性を示します。")
    with right:
        st.subheader("XGBoostとNAMs")
        st.write("XGBoostは複数の決定木を組み合わせる機械学習手法です。本研究は、新たな動物実験代替手法（NAMs）の活用を目指し、既存の化学物質データから予測を行います。")
    st.subheader("使い方")
    st.markdown("1. **単一物質の予測**で分子量と特徴量を入力するか、**CSV一括予測**でテンプレートに沿ったCSVを選びます。\n2. 予測LLNA EC3とADを確認し、必要に応じて結果をダウンロードします。\n3. **モデルの性能評価**で検証結果を、**SHAPによる説明**で選択したFoldの寄与を確認します。")
    st.info("予測は研究を支援する推定値です。適用範囲や実験データを含めて解釈してください。アップロードしたCSVは、このアプリからディスクへ保存しません。")
    if artifact is None:
        _missing_model()
    else:
        with st.expander("使用するモデルの情報"):
            st.write("作成条件: " + _training_label(artifact))
            st.write(f"入力特徴量: {len(artifact['full_feature_columns'])}列 / Fold: {len(_folds(artifact))}")
            st.write(f"モデルに使う選択特徴量: {len(_selected_features(artifact))}項目")
            st.write("入力列は学習時と同じ順序に揃えて、保存済みの前処理を適用します。")
            st.write("各Foldの予測から、Notebookに従って最小のEC3を採用します。AD内予測は、AD内と判定されたFoldだけを対象にします。")
            metadata = artifact.get("metadata", {})
            if metadata.get('feature_mode') in FIXED_FEATURE_SETS:
                st.write(f"指定された{len(artifact['full_feature_columns'])}項目だけで欠損補完・標準化・学習を行います。Borutaで項目を増減しません。")
                st.caption('この固定項目版は、452項目からBorutaで選択するNotebook再現版とは異なります。')
                if MW_COL not in artifact['full_feature_columns']:
                    st.write('MolWeightはEC3への換算専用の別入力です。説明変数・補完距離・AD・SHAPには使いません。')
            else:
                st.write(f"前処理にはMolWeightを含む全{len(artifact['full_feature_columns'])}項目を使います。KNNImputerで補完し、StandardScalerで標準化してから、Borutaで選ばれた{len(_selected_features(artifact))}項目をモデルへ渡します。")
                st.write("Boruta選択用の基準は全学習データで作成し、その後の各Foldの前処理はそのFoldの学習行だけでfitします。外部検証・新規入力には保存した各Foldの前処理を適用し、再fitしません。")
                st.caption("選択されなかった入力列もKNNの近傍距離に使います。選択後の項目だけを渡して、残りの列を省略することはできません。")
                st.write("Borutaで選択された特徴量（元データの列順）")
                st.dataframe(pd.DataFrame({"特徴量": _selected_features(artifact)}), hide_index=True, width="stretch")
            if _is_notebook_workflow(artifact) and "notebook_reproduction_verified" in metadata:
                st.write("現行環境での再現性チェック: " + ("検証済み（比較範囲は下記参照）" if metadata["notebook_reproduction_verified"] else "未確認"))
                st.caption("元Notebookには比較用の学習済みモデル・保存済み予測がないため、過去の実行結果との数値一致は確認できません。")
            if metadata.get("source_parity_scope"):
                st.write("ソースコードとの比較範囲: " + str(metadata["source_parity_scope"]))
            if "knn_k" in metadata:
                st.write(f"ADの近傍数: {metadata['knn_k']} / 学習距離分布から除く割合: {metadata.get('ad_coverage', '不明')}")
            versions = artifact.get("versions", artifact.get("metadata", {}).get("versions", artifact.get("library_versions")))
            if versions:
                st.json(versions)
        _invitro_ad_note(artifact)


def _input_columns(artifact: dict) -> list[str]:
    return list(dict.fromkeys(list(artifact["full_feature_columns"]) + [MW_COL]))


def _signature(frame: pd.DataFrame) -> str:
    model_version = str(ARTIFACT_PATH.stat().st_mtime_ns) if ARTIFACT_PATH.is_file() else "missing"
    content = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(str(ARTIFACT_PATH).encode() + model_version.encode() + content).hexdigest()


def _validate_frame(frame: pd.DataFrame, artifact: dict) -> int:
    missing = [col for col in _input_columns(artifact) if col not in frame.columns]
    if missing:
        suffix = f" ほか{len(missing) - 12}列" if len(missing) > 12 else ""
        raise ValueError(f"入力列が{len(missing)}列不足しています: {', '.join(missing[:12])}{suffix}。テンプレートの全列を含めてください。")
    mw = pd.to_numeric(frame[MW_COL], errors="coerce").to_numpy(dtype=float)
    bad = np.flatnonzero(~np.isfinite(mw) | (mw <= 0))
    if len(bad):
        rows = ", ".join(str(i + 1) for i in bad[:10])
        raise ValueError(f"MolWeightには正の有限な数が必要です。データ行 {rows} を確認してください（ヘッダーを除く行番号）。")
    features = prepare_input_features(artifact, frame)
    return int(features.isna().sum().sum())


def _render_prediction(frame: pd.DataFrame) -> None:
    row = frame.iloc[0]
    left, right = st.columns(2)
    left.metric("予測LLNA EC3（%）", f"{float(row['Predict_LLNA EC3']):.6g}")
    left.caption(f"採用Fold: {row.get('selected_fold', '記録なし')}")
    inside = row.get("any_kNN_AD", row.get("kNN_AD", None))
    right.metric("全Foldの適用範囲（kNN AD）", ("範囲内" if bool(inside) else "範囲外") if pd.notna(inside) else "判定なし")
    right.caption("Notebookの判定: 1つ以上のFoldがAD内なら範囲内。")
    chosen_inside = row.get("kNN_AD", None)
    if pd.notna(chosen_inside):
        right.caption("最小EC3を採用したFold単独のAD: " + ("範囲内" if bool(chosen_inside) else "範囲外"))
    if "any_kNN_AD" in row and not bool(row["any_kNN_AD"]):
        st.warning("すべてのFoldで適用範囲外です。類似する学習データが少ない可能性があるため、予測値の解釈に注意してください。")
    inad = row.get("knn_inAD_Predict_LLNA EC3", np.nan)
    if pd.notna(inad):
        st.write(f"AD内Foldだけの予測: **{float(inad):.6g}%**（採用Fold: {row.get('selected_inAD_fold', '記録なし')}）")
    with st.expander("Foldごとの予測・ADと入力識別情報"):
        st.dataframe(frame, hide_index=True, width="stretch")


def _single(artifact: dict) -> None:
    st.subheader("単一化学物質の予測")
    st.write("分子量と特徴量を入力してください。特徴量の値は空欄にできます。空欄は欠損値として、学習済みの補完処理へ渡します。")
    name = st.text_input("化学物質名（任意）", key="single_name")
    mw_label = "MolWeight（分子量・説明変数およびEC3換算用）" if MW_COL in artifact['full_feature_columns'] else "MolWeight（分子量・EC3換算用）"
    mw = st.number_input(mw_label, value=180.16, min_value=0.0, format="%.6f", key="single_mw")
    features = [col for col in artifact["full_feature_columns"] if col != MW_COL]
    selected = set(artifact.get("selected_feature_columns", []))
    st.caption("下の「値」セルを選択すると数値を入力できます。特徴量名は学習時の名称です。表示順序はモデルの入力順序に対応します。")
    if _is_notebook_workflow(artifact):
        st.info(f"原本と同じ前処理には全{len(artifact['full_feature_columns'])}項目を使います。選択後の{len(_selected_features(artifact))}項目だけを入力し、残りを空欄にした場合は、全項目が既知のCSVと補完結果が異なることがあります。できるだけ全項目が入ったCSVを使ってください。")
    initial = pd.DataFrame({"特徴量": features, "値": pd.Series(np.nan, index=range(len(features))), "モデル選択": ["選択" if col in selected else "補完に使用" for col in features]})
    edited = st.data_editor(initial, key="single_feature_values", disabled=["特徴量", "モデル選択"], hide_index=True, num_rows="fixed", height=480, width="stretch", column_config={"値": st.column_config.NumberColumn("値", format="%.8g")})
    st.caption("モデルに選択されていない特徴量も、KNNによる欠損補完の距離計算に使われます。既知の値をできるだけ入力してください。")
    record = dict(zip(edited["特徴量"], edited["値"]))
    record.update({MW_COL: mw, "Name": name})
    frame = pd.DataFrame([record])
    signature = _signature(frame)
    if st.button("予測実行", type="primary", key="single_predict"):
        try:
            missing_count = _validate_frame(frame, artifact)
            if missing_count:
                st.warning(f"{missing_count}個の欠損値を学習済みKNNImputerで補完します。入力が少ない場合、予測の解釈に注意してください。")
                if MW_COL not in artifact['full_feature_columns'] and missing_count == len(artifact['full_feature_columns']):
                    st.warning('説明変数がすべて空欄です。近さを判定する情報がないため学習データの項目平均で補完されます。既知の値を入力してください。')
            with st.spinner("予測とAD判定を計算しています…"):
                result = predict_with_artifact(artifact, frame, compute_ad=True)
            st.session_state["single_result"] = result
            st.session_state["single_result_signature"] = signature
            st.session_state["latest_prediction_input"] = frame
        except (ValueError, KeyError, TypeError) as error:
            st.session_state.pop("single_result", None)
            st.error(str(error))
    if "single_result" in st.session_state and st.session_state.get("single_result_signature") == signature:
        st.success("予測が完了しました。")
        _render_prediction(st.session_state["single_result"])
        st.download_button("予測結果CSVをダウンロード", dataframe_to_csv_bytes(st.session_state["single_result"]), "ec3_single_prediction.csv", "text/csv", key="single_download")


def _batch(artifact: dict) -> None:
    if artifact.get('metadata', {}).get('feature_mode') == 'fixed21':
        _batch_fixed21(artifact)
        return
    st.subheader("CSVによる一括予測")
    st.write("テンプレートの全入力列を含むCSVを選んでください。Name、CAS Numberなどの識別列と、行の並び順は結果に保持されます。")
    columns = _input_columns(artifact)
    st.download_button("入力テンプレートCSVをダウンロード", dataframe_to_csv_bytes(pd.DataFrame(columns=columns)), "ec3_input_template.csv", "text/csv", key="input_template")
    input_description = (f"説明変数{len(artifact['full_feature_columns'])}列＋換算専用MolWeight 1列（合計{len(columns)}列）"
                         if MW_COL not in artifact['full_feature_columns'] else f"前処理用の全{len(columns)}列（MolWeightを含む）")
    st.caption(input_description + '。MolWeightは正の数を入力します。説明変数の欠損セルは学習済みKNNImputerで補完します。実測EC3の列は不要です。')
    with st.expander("CSVの形式と欠損値について"):
        st.write("1行につき1物質です。列名はテンプレートと完全に一致させてください。列の並び替えは可能です。すべての特徴量列が必要ですが、欠損セルは空欄にできます。MolWeightの欠損・0以下・無限大は使用できません。")
        if _is_notebook_workflow(artifact):
            st.write(f"Borutaで選択された{len(_selected_features(artifact))}項目だけのCSVは入力できません。残りを空欄の列として追加しても、元の全項目が既知の外部データと同じ変換値にはなりません。補完距離に使うため、全{len(artifact['full_feature_columns'])}項目のうち既知の値をすべて含めてください。")
        st.write("UTF-8 / CP932 / Shift-JIS、最大20 MB・10,000行。実測EC3は予測入力に使いません。アップロードしたデータと結果はセッションのメモリで扱います。")
    uploaded = st.file_uploader("予測用CSV", type=["csv"], key="batch_csv")
    if uploaded is None:
        return
    try:
        frame = read_uploaded_csv(uploaded)
        missing_count = _validate_frame(frame, artifact)
    except (ValueError, KeyError, TypeError) as error:
        st.error(str(error))
        return
    st.write(f"入力データ: {len(frame):,}行 × {len(frame.columns):,}列")
    st.dataframe(frame.head(10), hide_index=True, width="stretch")
    if missing_count:
        st.warning(f"入力特徴量の{missing_count:,}セルが欠損しています。学習済みKNNImputerで補完します。")
    fingerprint = _signature(frame)
    if st.button("一括予測を実行", type="primary", key="batch_predict"):
        try:
            with st.spinner("予測とAD判定を計算しています…"):
                result = predict_with_artifact(artifact, frame, compute_ad=True)
            st.session_state["batch_result"] = (fingerprint, result)
            st.session_state["latest_prediction_input"] = frame
        except (ValueError, KeyError, TypeError) as error:
            st.session_state.pop("batch_result", None)
            st.error(str(error))
    cached = st.session_state.get("batch_result")
    if cached is None or cached[0] != fingerprint:
        return
    result = cached[1]
    st.success(f"{len(result):,}件の予測が完了しました。")
    st.dataframe(result, hide_index=True, width="stretch")
    st.download_button("予測結果CSVをダウンロード", dataframe_to_csv_bytes(result), "ec3_batch_predictions.csv", "text/csv", key="batch_download")


def _batch_fixed21(artifact: dict) -> None:
    from csv_batch_inputs import attach_molecular_weights, mw_entry_frame
    from csv_input_schema import make_template_frame, normalize_csv_input

    st.subheader('CSVによる一括予測')
    st.write('CSVはNo.・Name・CAS Numberの3列に続けて、指定21項目を並べた24列です。分子量はアップロード後に物質ごとに入力します。')
    st.download_button('入力テンプレートCSVをダウンロード',
        dataframe_to_csv_bytes(make_template_frame()), 'ec3_input_template.csv', 'text/csv', key='input_template')
    st.caption('CSVの説明変数は21項目だけです。MolWeightは説明変数ではなく、予測yをEC3へ換算する別入力です。')
    uploaded = st.file_uploader('予測用CSV', type=['csv'], key='batch_csv')
    if uploaded is None:
        st.session_state.pop('batch_result', None)
        return
    try:
        frame = normalize_csv_input(read_uploaded_csv(uploaded))
    except (ValueError, KeyError, TypeError) as error:
        st.session_state.pop('batch_result', None)
        st.error(str(error))
        return
    st.write(f'入力データ: {len(frame):,}行 × {len(frame.columns):,}列')
    st.dataframe(frame.head(10), hide_index=True, width='stretch')
    missing = int(frame[artifact['full_feature_columns']].isna().sum().sum())
    if missing:
        st.warning(f'説明変数の{missing:,}セルが欠損しています。21項目で学習したKNNImputerで補完します。')
    st.subheader('物質ごとの分子量（MolWeight）')
    st.write('各行に有限で0より大きい分子量を入力してください。No.やCAS番号が重複していても、入力行番号で対応付けます。値の推測や固定値への置換は行いません。')
    content = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    upload_key = hashlib.sha256(content).hexdigest()
    edited = st.data_editor(mw_entry_frame(frame), key='batch_mw_' + upload_key,
        disabled=['入力行', 'No.', 'Name', 'CAS Number'], hide_index=True,
        num_rows='fixed', width='stretch',
        column_config={MW_COL: st.column_config.NumberColumn('MolWeight（分子量）', format='%.8g')})
    combined, fingerprint = None, None
    try:
        combined = attach_molecular_weights(frame, edited)
        fingerprint = _signature(combined)
    except (ValueError, KeyError, TypeError) as error:
        st.info('全物質のMolWeightを入力すると、一括予測を実行できます。')
        st.caption(str(error))
    cached = st.session_state.get('batch_result')
    if fingerprint is None or (cached is not None and cached[0] != fingerprint):
        st.session_state.pop('batch_result', None)
    clicked = st.button('一括予測を実行', type='primary', key='batch_predict', disabled=combined is None)
    if clicked and combined is not None:
        try:
            with st.spinner('予測とAD判定を計算しています…'):
                result = predict_with_artifact(artifact, combined, compute_ad=True)
            st.session_state['batch_result'] = (fingerprint, result)
            st.session_state['latest_prediction_input'] = combined
        except (ValueError, KeyError, TypeError) as error:
            st.session_state.pop('batch_result', None)
            st.error(str(error))
    cached = st.session_state.get('batch_result')
    if cached is None or cached[0] != fingerprint:
        return
    result = cached[1]
    st.success(f'{len(result):,}件の予測が完了しました。')
    st.dataframe(result, hide_index=True, width='stretch')
    st.download_button('予測結果CSVをダウンロード', dataframe_to_csv_bytes(result),
        'ec3_batch_predictions.csv', 'text/csv', key='batch_download')


def _show_metrics(metrics: dict | None) -> None:
    if not isinstance(metrics, dict):
        st.caption("この評価指標はモデルに記録されていません。")
        return
    for column, name in zip(st.columns(3), ("r2", "rmse", "mae")):
        value = next((metrics[key] for key in (name, name.upper(), "R²" if name == "r2" else name) if key in metrics), None)
        label = "R²" if name == "r2" else "MAE（追加指標）" if name == "mae" else "RMSE"
        column.metric(label, f"{float(value):.5g}" if value is not None and np.isfinite(float(value)) else "算出不可")
    if "n" in metrics:
        st.caption(f"評価データ数: {metrics['n']}")


def _scatter_report(report: pd.DataFrame | None, artifact: dict, space: str) -> None:
    if not isinstance(report, pd.DataFrame) or report.empty:
        st.info("行ごとの評価結果は保存されていません。学習処理で評価レポートを保存してください。")
        return
    if space == "y":
        actual_names = ["true_y", "y_true", "observed_y", "actual_y", "y"]
        pred_names = ["pred_y", "oof_pred_y", "prediction_y"]
        units = "変換目的変数 y"
    else:
        actual_names = ["true_ec3", "ec3_true", "observed_ec3", "actual_ec3", artifact.get("target", ""), "LLNA EC3(NS converted to 150%)", "LLNA_EC3"]
        pred_names = ["Predict_LLNA EC3", "pred_ec3", "oof_pred_ec3", "prediction_ec3"]
        units = "LLNA EC3（%）"
    actual = next((name for name in actual_names if name in report.columns), None)
    predicted = next((name for name in pred_names if name in report.columns), None)
    if actual is None or predicted is None:
        st.info("この単位で散布図を描くための実測・予測列がありません。保存済みレポートを確認してください。")
        st.dataframe(report, hide_index=True, width="stretch")
        return
    x = pd.to_numeric(report[actual], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(report[predicted], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    fig, axis = plt.subplots(figsize=(6.2, 4.5))
    if "fold" in report.columns:
        for fold in report["fold"].unique():
            selected = valid & report["fold"].eq(fold).to_numpy()
            axis.scatter(x[selected], y[selected], alpha=0.7, s=30, label=str(fold))
        axis.legend(fontsize="small")
    else:
        axis.scatter(x[valid], y[valid], alpha=0.7, color="#197e90", s=30)
    if valid.any():
        low, high = min(x[valid].min(), y[valid].min()), max(x[valid].max(), y[valid].max())
        axis.plot([low, high], [low, high], linestyle="--", color="#64748b", linewidth=1)
    axis.set(xlabel="Observed", ylabel="Predicted")
    axis.grid(alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"実測値（横軸）と予測値（縦軸）。単位: {units}。破線は実測と予測が一致する位置です。")
    with st.expander("評価データの予測一覧"):
        st.dataframe(report, hide_index=True, width="stretch")


def _notebook_evaluation_tables(report: pd.DataFrame | None, *, prediction_column="Predict_LLNA EC3") -> None:
    if not isinstance(report, pd.DataFrame) or report.empty:
        return
    truth_column = "LLNA EC3(NS converted to 150%)"
    if truth_column not in report or prediction_column not in report:
        return
    from notebook_metrics import evaluate_notebook_report
    values = evaluate_notebook_report(report, prediction_column=prediction_column, truth_column=truth_column)
    with st.expander("Notebookの分類・5倍範囲評価（EC3単位）"):
        st.caption("GHS: EC3≤2は1A、2<EC3≤100は1B、100超はNC。ECETOCは0.1 / 1 / 10 / 100を境界に原本の不等号で分類します。")
        st.markdown("**実測値の1/5倍より大きく、5倍より小さい予測の割合**")
        st.caption("境界のちょうど1/5倍・5倍は範囲内に含めません。overはEC3を低く予測する強めの予測、underはEC3を高く予測する弱めの予測です。")
        st.dataframe(values["range_rates"], width="stretch")
        for name, title in (("ghs", "GHS"), ("ecetoc", "ECETOC")):
            classification = values[name]
            st.markdown(f"**{title}分類のクロス表**")
            st.caption("行が予測、列が実測です。")
            st.dataframe(classification["matrix"], width="stretch")
            accuracy = classification['accuracy']
            accuracy_text = f"{accuracy:.3%}" if accuracy is not None else "算出不可"
            under = classification['underestimated']
            over = classification['overestimated']
            under_text = f"{under:.3%}" if under is not None else "算出不可"
            over_text = f"{over:.3%}" if over is not None else "算出不可"
            st.write(f"一致率: {accuracy_text} / 感作性の過小評価率: {under_text} / 感作性の過大評価率: {over_text} / 評価件数: {classification['n']}")
            st.caption("過小評価は実測より弱い感作性の区分、過大評価は実測より強い感作性の区分を予測した割合です。")
        if values.get("excluded_count", 0):
            st.caption(f"欠損・非有限等により評価から除外: {values['excluded_count']}件")


def _evaluation(artifact: dict) -> None:
    st.subheader("モデルの性能評価")
    st.write("学習に使ったデータ上の評価、交差検証の未学習Foldに対する予測（OOF）、独立した外部データの評価を区別して表示します。")
    if artifact.get('metadata', {}).get('feature_mode') in FIXED_FEATURE_SETS:
        st.info(f"説明変数は指定{len(artifact['full_feature_columns'])}項目に固定し、各Foldの学習部分で欠損補完・標準化します。外部検証データは学習へ混合しません。")
    else:
        st.info("Notebook再現条件では、特徴量選択をCV分割の前に全学習データへ適用します。また、内側CV前に外側学習データで補完・標準化します。この情報共有を保持しているため、OOFや内側CVの値だけで未知物質の性能を判断しないでください。外部検証データは学習へ混合しません。")
        st.caption(f"kNN ADは選択・標準化後の空間で計算します。近傍数k={artifact['metadata'].get('knn_k', '不明')}、coverage={artifact['metadata'].get('ad_coverage', '不明')}は学習距離分布から除く割合です。")
        _invitro_ad_note(artifact)
    metrics, reports = artifact.get("metrics", {}), artifact.get("reports", {})
    space_label = st.radio("表示する単位", ["変換目的変数 y", "LLNA EC3（%）"], horizontal=True, key="evaluation_space")
    space = "y" if space_label == "変換目的変数 y" else "ec3"
    st.caption("y = log10(MolWeight / EC3)。yのRMSE・MAEとEC3（%）のRMSE・MAEは異なる単位です。")
    oof, train, external = st.tabs(["交差検証（OOF）", "学習データ（Fold内）", "外部検証"])
    with oof:
        st.write("各行が属するテストFoldの予測を使用します。学習に使った行への当てはめとは区別します。")
        _show_metrics(metrics.get(f"oof_{space}"))
        _scatter_report(reports.get("oof"), artifact, space)
        _notebook_evaluation_tables(reports.get("oof"))
    with train:
        st.write("各Foldの学習部分への当てはめです。学習済みデータへの結果のため、未知データへの性能より良く見える可能性があります。")
        rows = metrics.get("train_folds", metrics.get("folds", []))
        if isinstance(rows, dict):
            display_rows = []
            for name, values in rows.items():
                for prefix, label in (("training", "学習部分"), ("holdout", "検証部分")):
                    scores = values.get(f"{prefix}_{space}", {}) if isinstance(values, dict) else {}
                    if scores:
                        display_rows.append({"Fold": name, "評価対象": label, "件数": scores.get("n"), "R²": scores.get("r2"), "RMSE": scores.get("rmse"), "MAE": scores.get("mae")})
            rows = display_rows
        if rows:
            st.dataframe(pd.json_normalize(rows), hide_index=True, width="stretch")
        else:
            st.info("Foldごとの学習指標は保存されていません。")
    with external:
        st.write("学習に混合していない外部検証用CSVに対する結果です。標準の最終予測は各Foldから得た最小EC3に対応します。平均はNotebook内で比較した別の集約です。")
        external_report = reports.get("external")
        alternatives = ["最小EC3（標準）"]
        if isinstance(external_report, pd.DataFrame) and {"pred_y_mean", "Predict_LLNA EC3_mean"}.issubset(external_report.columns):
            alternatives.append("平均（Notebookの比較用）")
        method = st.selectbox("外部予測の集約", alternatives, key="external_prediction_method")
        mean_method = method == "平均（Notebookの比較用）"
        key = f"external_mean_{space}" if mean_method else f"external_{space}"
        _show_metrics(metrics.get(key))
        display_report = external_report
        if mean_method:
            columns = ["true_y", "LLNA EC3(NS converted to 150%)", "pred_y_mean", "Predict_LLNA EC3_mean"]
            display_report = external_report[[col for col in columns if col in external_report]].rename(
                columns={"pred_y_mean": "pred_y", "Predict_LLNA EC3_mean": "Predict_LLNA EC3"})
            st.caption("yの平均と、各Foldで逆変換したEC3の平均をそれぞれ計算します。平均yを逆変換した値と、EC3の平均は同じではありません。")
        _scatter_report(display_report, artifact, space)
        _notebook_evaluation_tables(display_report)
        if metrics.get(f"external_inAD_{space}"):
            st.markdown("**AD内Foldだけから最小EC3を採用した外部評価**")
            _show_metrics(metrics[f"external_inAD_{space}"])
            if isinstance(external_report, pd.DataFrame):
                _notebook_evaluation_tables(external_report, prediction_column="knn_inAD_Predict_LLNA EC3")


def _training_explanation(fold: dict):
    import shap
    names = list(fold.get("selected_feature_columns", []))
    data = fold.get("X_shap", fold.get("X_train_selected"))
    if data is None:
        raise ValueError("SHAP用の参照データが保存されていません。予測を行ってから、直前の入力データを選んでください。")
    data = np.asarray(data, dtype=float)
    if not names:
        names = [f"feature_{i}" for i in range(data.shape[1])]
    if "shap_values" in fold and "shap_base_values" in fold:
        values = np.asarray(fold["shap_values"])[:len(data)]
        base = np.asarray(fold["shap_base_values"])
        if base.ndim:
            base = base[:len(data)]
        return shap.Explanation(values=values, base_values=base, data=data, feature_names=names)
    return shap.TreeExplainer(fold["model"])(pd.DataFrame(data, columns=names))


def _crossfold_notebook_importance(artifact: dict) -> pd.DataFrame:
    names = _selected_features(artifact)
    columns = {}
    for fold in _folds(artifact):
        if list(fold["selected_feature_columns"]) != names:
            raise ValueError("特徴量が異なるFoldを同一の重要度表に平均できません。")
        explanation = _training_explanation(fold)
        columns[fold["fold_name"]] = _notebook_shap_importance(explanation.values)
    table = pd.DataFrame(columns, index=pd.Index(names, name="特徴量"))
    table["Fold平均"] = table.mean(axis=1)
    return table.sort_values("Fold平均", ascending=False).reset_index()


def _shap(artifact: dict) -> None:
    import shap
    st.subheader("SHAPによる予測の説明")
    st.write("1つのFoldのXGBoostモデルについて、各特徴量が予測をどちらに動かすかを表示します。Foldごとに特徴量やモデルが異なるため、SHAP値を単純平均しません。")
    folds = _folds(artifact)
    names = [fold["fold_name"] for fold in folds]
    name = st.selectbox("説明するFold", names, key="shap_fold")
    fold = folds[names.index(name)]
    sources = ["Fold学習データ（全行）"]
    if "latest_prediction_input" in st.session_state:
        sources.append("直前に予測した入力データ")
    source = st.selectbox("説明するデータ", sources, key="shap_source")
    st.info("SHAPの出力単位は y = log10(MolWeight / EC3) です。正の寄与はyを増やし、同じ分子量ではEC3を小さくする方向です。EC3（%）への直接の加算値ではありません。")
    st.caption(f"選択モデル: {name}。図の特徴量値は、このFoldの前処理を適用したモデル入力値です。選択データの全行・全選択特徴量を使います。原本のSummaryはFold学習データが対象です。")
    signature = (str(ARTIFACT_PATH), str(ARTIFACT_PATH.stat().st_mtime_ns), name, source)
    if source == "直前に予測した入力データ":
        signature += (hashlib.sha256(pd.util.hash_pandas_object(st.session_state["latest_prediction_input"], index=True).values.tobytes()).hexdigest(),)
    if st.button("SHAPを計算・表示", type="primary", key="shap_compute"):
        try:
            with st.spinner("特徴量の寄与を計算しています…"):
                if source == "直前に予測した入力データ":
                    explanation = explain_prediction(artifact, st.session_state["latest_prediction_input"], fold_name=name)
                else:
                    explanation = _training_explanation(fold)
            st.session_state["shap_result"] = (signature, explanation)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            st.session_state.pop("shap_result", None)
            st.error(f"SHAPを表示できませんでした: {error}")
    cached = st.session_state.get("shap_result")
    if cached is None or cached[0] != signature:
        return
    explanation = cached[1]
    n_rows = len(explanation.values)
    st.success(f"{name} の {n_rows}行を説明しています。最終的な複数Foldの集約値そのものへの説明ではありません。")
    importance, summary, beeswarm, waterfall = st.tabs(["特徴量重要度", "SHAP Summary", "SHAP Beeswarm", "個別の寄与"])
    with importance:
        st.caption("原Notebookの集計式 abs(sum(SHAP値)): 行方向に符号付きで合計してから絶対値を取ります。平均絶対SHAP値とは異なり、正負の寄与は打ち消し合います。")
        values = _notebook_shap_importance(explanation.values)
        order = np.argsort(values)
        feature_names = np.asarray(explanation.feature_names)
        fig, axis = plt.subplots(figsize=(9, max(4.5, len(values) * .27)))
        axis.barh(feature_names[order], values[order], color="#197e90")
        axis.set_xlabel("abs(sum(SHAP values))")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if _is_notebook_workflow(artifact):
            with st.expander("全Foldの原Notebook集計表"):
                st.caption("各Foldの学習行全体でabs(sum(SHAP値))を計算し、その数値をFold間で平均します。元コードの文字列の特徴名列を平均へ含める処理は除外します。生のSHAP値をFold間で平均する処理ではありません。")
                st.dataframe(_crossfold_notebook_importance(artifact), hide_index=True, width="stretch")
    with summary:
        if n_rows < 2:
            st.info("Violin Plotで分布を表示するには2行以上が必要です。Fold学習データか、複数行を予測した入力データを選んでください。単一物質の寄与は「個別の寄与」で確認できます。")
        else:
            shap.summary_plot(explanation.values, features=explanation.data, feature_names=explanation.feature_names, plot_type="violin", max_display=len(explanation.feature_names), show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("各特徴量のSHAP値の分布をViolin Plotで表示します。横軸はyへの寄与、色は前処理後の特徴量値です。")
    with beeswarm:
        shap.plots.beeswarm(explanation, max_display=len(explanation.feature_names), show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("追加表示のBeeswarmです。各点が1行の寄与、横軸はyへの寄与、色は前処理後の特徴量値です。原本のSummaryは隣のViolin Plotです。")
    with waterfall:
        row = st.number_input("個別に説明する行（1から）", min_value=1, max_value=max(n_rows, 1), value=1, step=1, key="shap_row")
        shap.plots.waterfall(explanation[int(row) - 1], max_display=len(explanation.feature_names), show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("基準値から、このFoldの予測yに至る特徴量ごとの寄与を示します。")


def main() -> None:
    st.set_page_config(page_title="LLNA EC3 予測", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
    st.title("LLNA EC3 予測")
    st.caption("XGBoostによる皮膚感作性の推定と、モデルの検証・説明")
    artifact, load_error = None, None
    if ARTIFACT_PATH.is_file():
        try:
            artifact = _load_model(str(ARTIFACT_PATH), ARTIFACT_PATH.stat().st_mtime_ns)
        except Exception as error:
            load_error = error
    with st.sidebar:
        st.markdown("### EC3 研究アプリ")
        page = st.radio("画面を選択", PAGES, key="navigation")
        st.divider()
        if artifact is not None:
            st.caption("使用するモデル")
            st.write(_training_label(artifact))
            st.caption(f"{len(_folds(artifact))} Fold / 前処理{len(artifact['full_feature_columns'])}項目 → モデル{len(_selected_features(artifact))}項目")
        else:
            st.caption("モデル未準備")
    if artifact is not None:
        mode = artifact.get("training_mode", artifact.get("metadata", {}).get("training_mode"))
        if mode == "smoke":
            st.warning("軽量動作確認モデルを使用中です。本番のNotebook再現学習を完了したモデルではありません。予測・性能・SHAPは動作確認用です。")
        elif mode != "notebook":
            st.warning("モデルの作成条件が記録されていません。予測を利用する前に学習条件を確認してください。")
    if load_error is not None:
        st.error(f"学習済みモデルを読み込めませんでした: {load_error}")
        st.caption("信頼できる環境で作成したモデルと、READMEに記載のライブラリを使用してください。")
    if page == "ホーム":
        _home(artifact)
    elif artifact is None:
        _missing_model()
    else:
        {"単一物質の予測": _single, "CSV一括予測": _batch, "モデルの性能評価": _evaluation, "SHAPによる説明": _shap}[page](artifact)


if __name__ == "__main__":
    main()
