"""Both public model choices use their own pipeline, data and saved weights."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from original_runtime import load_artifact, predict_with_artifact
from utils_io import read_csv_auto


FIXED21 = '21項目版'
NOTEBOOK452 = '452項目版（元Notebook方式）'

DUAL_SCRIPT = """
from io import BytesIO
from unittest.mock import patch
import streamlit as st
import streamlit_app as ui

original_download = ui.st.download_button
original_editor = ui.st.data_editor
def capture_download(*args, **kwargs):
    if kwargs.get('key') == 'input_template':
        st.session_state['_test_template_bytes'] = kwargs.get('data', args[1] if len(args)>1 else None)
    if kwargs.get('key') == 'oof_download':
        st.session_state['_test_oof_bytes'] = kwargs.get('data', args[1] if len(args)>1 else None)
    return original_download(*args, **kwargs)
def uploaded_csv(*args, **kwargs):
    raw = st.session_state.get('_test_uploaded_csv')
    return BytesIO(raw) if raw is not None else None
def conversion_weights(data, *args, **kwargs):
    edited = original_editor(data, *args, **kwargs)
    if str(kwargs.get('key', '')).startswith('batch_mw_'):
        weights = st.session_state.get('_test_weights')
        if weights is not None:
            edited = edited.copy()
            edited['MolWeight'] = weights
    return edited
with patch.object(ui.st, 'download_button', side_effect=capture_download), \\
     patch.object(ui.st, 'file_uploader', side_effect=uploaded_csv), \\
     patch.object(ui.st, 'data_editor', side_effect=conversion_weights):
    ui.main()
"""


@pytest.fixture(autouse=True)
def clear_model_environment(monkeypatch):
    # These tests exercise the shipped two-model setup, not a legacy test override.
    for key in ('EC3_ARTIFACT_PATH', 'ARTIFACT_PATH',
                'EC3_FIXED21_ARTIFACT_PATH', 'EC3_NOTEBOOK452_ARTIFACT_PATH'):
        monkeypatch.delenv(key, raising=False)


def batch_app(profile=NOTEBOOK452, frame=None, weights=None):
    app = AppTest.from_string(DUAL_SCRIPT, default_timeout=60)
    app.session_state['navigation'] = 'CSV一括予測'
    app.session_state['model_profile'] = profile
    if frame is not None:
        app.session_state['_test_uploaded_csv'] = frame.to_csv(index=False).encode('utf-8-sig')
    if weights is not None:
        app.session_state['_test_weights'] = weights
    app.run()
    assert not app.exception
    return app


def test_default_and_both_downloaded_templates(repo_root):
    from csv_input_schema import requested_csv_columns

    app = AppTest.from_file(str(repo_root / 'streamlit_app.py'), default_timeout=60).run()
    assert not app.exception and not app.error
    assert app.sidebar.selectbox(key='model_profile').value == NOTEBOOK452

    app = batch_app()
    template452 = pd.read_csv(BytesIO(app.session_state['_test_template_bytes']))
    notebook = load_artifact(repo_root / 'artifacts/notebook_deployment_artifact.joblib')
    assert template452.empty
    assert template452.columns.tolist() == ['No.', 'Name', 'CAS Number', *notebook['full_feature_columns']]
    assert len(template452.columns) == 455
    assert template452.columns.tolist().count('MolWeight') == 1

    app.sidebar.selectbox(key='model_profile').set_value(FIXED21).run()
    assert not app.exception and not app.error
    template21 = pd.read_csv(BytesIO(app.session_state['_test_template_bytes']))
    assert template21.empty
    assert template21.columns.tolist() == requested_csv_columns()
    assert len(template21.columns) == 24
    assert 'MolWeight' not in template21 and 'VdW surface DPSA2' not in template21


@pytest.mark.parametrize('profile,source_name', [
    (FIXED21, 'original_model_artifact.joblib'),
    (NOTEBOOK452, 'notebook_deployment_artifact.joblib'),
])
def test_each_batch_prediction_matches_its_own_saved_source(repo_root, profile, source_name):
    from csv_input_schema import requested_csv_columns

    source = load_artifact(repo_root / 'artifacts' / source_name)
    data, _ = read_csv_auto(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
    data = data.head(3).copy()
    if profile == FIXED21:
        inputs = data[requested_csv_columns()]
        app = batch_app(profile, inputs, data.MolWeight.tolist())
    else:
        inputs = data[['No.', 'Name', 'CAS Number', *source['full_feature_columns']]]
        app = batch_app(profile, inputs)
    assert not app.error
    assert not app.button(key='batch_predict').disabled
    app.button(key='batch_predict').click().run()
    assert not app.exception and not app.error and app.success
    _, actual = app.session_state['batch_result']
    normalized, _ = read_csv_auto(app.session_state['_test_uploaded_csv'])
    if profile == FIXED21:
        normalized['MolWeight'] = data.MolWeight.to_numpy()
    expected = predict_with_artifact(source, normalized)
    assert actual.Name.tolist() == data.Name.tolist()
    np.testing.assert_array_equal(actual['pred_y'], expected['pred_y'])
    np.testing.assert_array_equal(actual['Predict_LLNA EC3'], expected['Predict_LLNA EC3'])
    assert actual['selected_fold'].tolist() == expected['selected_fold'].tolist()
    assert actual['any_kNN_AD'].tolist() == expected['any_kNN_AD'].tolist()


def test_switch_clears_model_bound_predictions_and_features(repo_root):
    app = batch_app(FIXED21)
    stale = {
        'single_result': pd.DataFrame({'Predict_LLNA EC3': [999.]}),
        'single_result_signature': 'old-model',
        'batch_result': ('old-model', pd.DataFrame({'Predict_LLNA EC3': [999.]})),
        'latest_prediction_input': pd.DataFrame({'old-feature': [1.]}),
        'shap_result': ('old-model', object()),
        'single_feature_values': {'edited_rows': {0: {'値': 987.}}},
        'batch_mw_stale_model': {'edited_rows': {0: {'MolWeight': 999.}}},
    }
    for key, value in stale.items():
        app.session_state[key] = value
    app.sidebar.selectbox(key='model_profile').set_value(NOTEBOOK452).run()
    assert not app.exception and not app.error
    for key in stale:
        assert key not in app.session_state, key
    assert not app.success
    assert '直前に予測した入力データ' not in '\n'.join(item.value for item in app.markdown)


def test_existing_session_cannot_reuse_predictions_from_before_model_selection():
    app = AppTest.from_string(DUAL_SCRIPT, default_timeout=60)
    app.session_state['navigation'] = 'CSV一括予測'
    # A session open before this update has old results but no active-model identity.
    app.session_state['latest_prediction_input'] = pd.DataFrame({'old-feature': [1.]})
    app.session_state['batch_result'] = ('old', pd.DataFrame({'Predict_LLNA EC3': [999.]}))
    app.session_state['shap_result'] = ('old', object())
    app.run()
    assert not app.exception and not app.error
    for key in ('latest_prediction_input', 'batch_result', 'shap_result'):
        assert key not in app.session_state
    assert not app.success


def test_21_column_upload_is_rejected_in_452_mode(repo_root):
    from csv_input_schema import requested_csv_columns

    data, _ = read_csv_auto(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
    app = batch_app(NOTEBOOK452, data[requested_csv_columns()].head(1))
    assert any('不足' in message.value for message in app.error)
    assert 'batch_result' not in app.session_state
    assert not app.success


def test_notebook_oof_download_keeps_saved_validation_predictions(repo_root):
    artifact = load_artifact(repo_root / 'artifacts/notebook_deployment_artifact.joblib')
    app = batch_app()
    app.sidebar.radio(key='navigation').set_value('モデルの性能評価').run()
    assert not app.exception and not app.error
    downloaded = pd.read_csv(BytesIO(app.session_state['_test_oof_bytes']))
    saved = artifact['reports']['oof'].reset_index(drop=True)
    pd.testing.assert_frame_equal(downloaded, saved, check_dtype=False, atol=1e-12)
    assert len(downloaded) == 154
    assert any('DASS_predict_result' in item.value for item in app.caption)
    assert any('最小EC3とは比較条件が異なります' in item.value for item in app.caption)

    data, _ = read_csv_auto(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
    batch = predict_with_artifact(artifact, data.head(5))
    # An all-fold minimum must never overwrite the held-out-fold report.
    assert not np.array_equal(downloaded['Predict_LLNA EC3'].head(5), batch['Predict_LLNA EC3'])


@pytest.mark.parametrize('wrong_kind', ['missing', 'fixed21'])
def test_missing_or_wrong_452_model_does_not_fall_back(repo_root, tmp_path, monkeypatch, wrong_kind):
    path = (tmp_path / 'missing452.joblib' if wrong_kind == 'missing'
            else repo_root / 'artifacts/original_model_artifact.joblib')
    monkeypatch.setenv('EC3_NOTEBOOK452_ARTIFACT_PATH', str(path))
    app = AppTest.from_file(str(repo_root / 'streamlit_app.py'), default_timeout=60).run()
    assert not app.exception
    assert app.sidebar.selectbox(key='model_profile').value == NOTEBOOK452
    assert app.error
    app.sidebar.radio(key='navigation').set_value('単一物質の予測').run()
    assert not app.exception and app.error
    assert not app.button
    assert 'single_result' not in app.session_state
