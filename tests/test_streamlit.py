"""Render actual Streamlit pages and exercise prediction/error paths with AppTest."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils_io import read_csv_auto


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp932"])
def test_upload_and_download_helpers_keep_japanese_ids(encoding):
    from streamlit_app import dataframe_to_csv_bytes, read_uploaded_csv

    text = "Name,MolWeight,x\n物質Ａ,100,\n物質Ｂ,200,1\n"
    frame = read_uploaded_csv(BytesIO(text.encode(encoding)))
    exported = dataframe_to_csv_bytes(frame)
    assert exported.startswith(b"\xef\xbb\xbf")
    decoded, _ = read_csv_auto(exported)
    pd.testing.assert_frame_equal(frame, decoded)
    assert decoded.Name.tolist() == ["物質Ａ", "物質Ｂ"]


@pytest.mark.parametrize("raw", [b"a,a\n1,2\n", b"a,b\n1,2,3\n", b"a,b\n1\n", b""])
def test_upload_rejects_malformed_rows_and_columns(raw):
    from streamlit_app import read_uploaded_csv

    with pytest.raises(ValueError):
        read_uploaded_csv(BytesIO(raw))


def test_upload_size_and_row_limits(monkeypatch):
    import streamlit_app as ui

    monkeypatch.setattr(ui, "MAX_UPLOAD_BYTES", 8)
    with pytest.raises(ValueError, match="20 MB"):
        ui.read_uploaded_csv(b"x\n1234567\n")
    monkeypatch.setattr(ui, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(ui, "MAX_UPLOAD_ROWS", 1)
    with pytest.raises(ValueError, match="10,000"):
        ui.read_uploaded_csv(b"x\n1\n2\n")


def test_all_pages_render_without_retraining(repo_root, smoke_artifact_path, monkeypatch):
    monkeypatch.setenv("EC3_ARTIFACT_PATH", str(smoke_artifact_path))
    modified = smoke_artifact_path.stat().st_mtime_ns
    app = AppTest.from_file(str(repo_root / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("NAMs" in block.value for block in app.markdown)
    for page in ["単一物質の予測", "CSV一括予測", "モデルの性能評価", "SHAPによる説明"]:
        app.sidebar.radio(key="navigation").set_value(page).run()
        assert not app.exception, page
        assert not app.error, page
    assert smoke_artifact_path.stat().st_mtime_ns == modified


def test_missing_artifact_is_explained_on_every_page(repo_root, tmp_path, monkeypatch):
    monkeypatch.setenv("EC3_ARTIFACT_PATH", str(tmp_path / "not_trained.joblib"))
    app = AppTest.from_file(str(repo_root / "streamlit_app.py"), default_timeout=30).run()
    for page in ["ホーム", "単一物質の予測", "CSV一括予測", "モデルの性能評価", "SHAPによる説明"]:
        app.sidebar.radio(key="navigation").set_value(page).run()
        assert not app.exception
        assert any("学習済みモデル" in message.value for message in app.error)
        assert any("train.py" in block.value for block in app.code)


def test_single_prediction_then_invalid_mw_clears_result(repo_root, smoke_artifact_path, monkeypatch):
    monkeypatch.setenv("EC3_ARTIFACT_PATH", str(smoke_artifact_path))
    app = AppTest.from_file(str(repo_root / "streamlit_app.py"), default_timeout=30).run()
    app.sidebar.radio(key="navigation").set_value("単一物質の予測").run()
    app.text_input(key="single_name").set_value("試験物質").run()
    app.number_input(key="single_mw").set_value(180.).run()
    app.button(key="single_predict").click().run()
    assert not app.exception
    assert not app.error
    assert app.success
    result = app.session_state["single_result"]
    assert result.Name.tolist() == ["試験物質"]
    assert np.isfinite(result["Predict_LLNA EC3"].iloc[0])
    assert app.metric[0].label == "予測LLNA EC3（%）"
    assert float(app.metric[0].value) > 0
    assert any("欠損値" in warning.value for warning in app.warning)

    app.number_input(key="single_mw").set_value(0.).run()
    assert not app.success  # editing an input must not leave an apparently current prediction
    app.button(key="single_predict").click().run()
    assert not app.exception
    assert any("MolWeight" in error.value for error in app.error)
    assert "single_result" not in app.session_state
    assert not app.success


BATCH_SCRIPT = """
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import streamlit as st
import streamlit_app as ui
ui.ARTIFACT_PATH = Path(st.session_state["_test_artifact_path"])
with patch.object(ui.st, "file_uploader", return_value=BytesIO(st.session_state["_test_uploaded_csv"])):
    ui.main()
"""


def test_batch_ui_prediction_download_and_column_error(repo_root, smoke_artifact_path):
    frame, _ = read_csv_auto(repo_root / "data" / "DASS_join_data_opera_delete_LLNAnan.csv")
    inputs = frame.head(3).drop(columns="LLNA_EC3")
    # AppTest does not expose a file-uploader setter; substitute its in-memory return value.
    app = AppTest.from_string(BATCH_SCRIPT, default_timeout=30)
    app.session_state["_test_artifact_path"] = str(smoke_artifact_path)
    app.session_state["_test_uploaded_csv"] = inputs.to_csv(index=False).encode("utf-8-sig")
    app.session_state["navigation"] = "CSV一括予測"
    app.run()
    assert not app.exception
    assert not app.error
    app.button(key="batch_predict").click().run()
    assert not app.exception
    assert not app.error
    _, result = app.session_state["batch_result"]
    assert result.Name.tolist() == inputs.Name.tolist()
    assert len(result) == 3
    assert app.success
    assert len(app.get("download_button")) == 2

    app.session_state["_test_uploaded_csv"] = b"Name,MolWeight\nmissing descriptors,100\n"
    app.run()
    assert not app.exception
    assert any("不足" in error.value for error in app.error)
    assert not app.success


def test_shap_ui_renders_all_four_plots(repo_root, smoke_artifact_path, monkeypatch):
    monkeypatch.setenv("EC3_ARTIFACT_PATH", str(smoke_artifact_path))
    app = AppTest.from_file(str(repo_root / "streamlit_app.py"), default_timeout=60).run()
    app.sidebar.radio(key="navigation").set_value("SHAPによる説明").run()
    app.selectbox(key="shap_fold").set_value("fold_2").run()
    app.button(key="shap_compute").click().run()
    assert not app.exception
    assert not app.error
    assert any("fold_2" in success.value for success in app.success)
    assert [tab.label for tab in app.tabs] == ["特徴量重要度", "SHAP Summary", "SHAP Beeswarm", "個別の寄与"]
    assert len(app.get("image")) == 4
    _, explanation = app.session_state["shap_result"]
    assert len(explanation.values) > 0
    app.selectbox(key="shap_fold").set_value("fold_3").run()
    assert not app.success
    assert not app.get("image")


def test_single_prediction_shap_keeps_individual_plots_and_explains_violin_limit(repo_root, smoke_artifact_path, monkeypatch):
    monkeypatch.setenv("EC3_ARTIFACT_PATH", str(smoke_artifact_path))
    app = AppTest.from_file(str(repo_root / "streamlit_app.py"), default_timeout=60).run()
    app.sidebar.radio(key="navigation").set_value("単一物質の予測").run()
    app.button(key="single_predict").click().run()
    assert not app.exception
    assert not app.error
    assert app.success

    app.sidebar.radio(key="navigation").set_value("SHAPによる説明").run()
    app.selectbox(key="shap_source").set_value("直前に予測した入力データ").run()
    app.button(key="shap_compute").click().run()
    assert not app.exception
    assert not app.error
    assert app.success
    _, explanation = app.session_state["shap_result"]
    assert explanation.values.shape[0] == 1
    assert len(app.get("image")) == 3
    assert any("Violin" in item.value and "2行以上" in item.value for item in app.info)
    assert [tab.label for tab in app.tabs] == ["特徴量重要度", "SHAP Summary", "SHAP Beeswarm", "個別の寄与"]
    for index in [0, 2, 3]:
        assert len(app.tabs[index].get("image")) == 1
    assert not app.tabs[1].get("image")


def test_compatibility_entrypoint_still_renders(repo_root, smoke_artifact_path, monkeypatch):
    monkeypatch.setenv("EC3_ARTIFACT_PATH", str(smoke_artifact_path))
    app = AppTest.from_file(str(repo_root / "app_original.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "LLNA EC3 予測"
