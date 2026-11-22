"""Requested 24-column CSV and explicitly entered per-row molecular weights."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from test_streamlit import BATCH_SCRIPT
from utils_io import read_csv_auto


REQUESTED_COLUMNS = [
    'No.', 'Name', 'CAS Number',
    'OVERALL OH rate constant', 'FM advection sediment', 'LUMO Energy',
    'HOMO Energy', 'CombDipolPolariz', 'LogHL_pred', 'LogKM_pred',
    'alert_Protein_binding_by_OASIS',
    'judge_Protein_binding_alerts_for_skin_sensitization_according_to_GHS',
    'potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS',
    'DPRA.percCysdep', 'DPRA.percLysdep', 'DPRA.score',
    'hCLAT.CD86.EC150..ug.ml.', 'h.CLAT.CV75', 'hCLAT.MIT', 'hCLAT.score',
    'KS.EC1.5', 'KS.EC3', 'KS.IC50', 'sum_invitro',
]


@pytest.fixture(scope='module')
def fixed21_model(repo_root):
    from original_runtime import load_artifact
    for name in ('fixed21_model_artifact.joblib', 'original_model_artifact.joblib'):
        path = repo_root / 'artifacts' / name
        if path.is_file() and load_artifact(path)['metadata'].get('feature_mode') == 'fixed21':
            return path
    pytest.skip('A trusted fixed21 artifact is needed for the per-row MW UI.')


@pytest.fixture
def input_rows(repo_root):
    frame, _ = read_csv_auto(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
    return frame[REQUESTED_COLUMNS].head(3).copy(), frame.MolWeight.head(3).tolist()


def make_app(model_path, frame):
    app = AppTest.from_string(BATCH_SCRIPT, default_timeout=60)
    app.session_state['_test_artifact_path'] = str(model_path)
    app.session_state['_test_uploaded_csv'] = frame.to_csv(index=False).encode('utf-8-sig')
    app.session_state['navigation'] = 'CSV一括予測'
    app.run()
    assert not app.exception
    return app


def enter_weights(app, weights):
    key = app.session_state['_test_mw_editor_key']
    app.session_state['_test_mw_values_by_key'] = {key: weights}
    app.run()
    assert not app.exception
    return key


def test_downloaded_template_has_exact_requested_24_columns(fixed21_model, input_rows):
    frame, _ = input_rows
    app = make_app(fixed21_model, frame)
    template = pd.read_csv(BytesIO(app.session_state['_test_template_bytes']))
    assert template.empty
    assert template.columns.tolist() == REQUESTED_COLUMNS
    assert len(template.columns) == 24
    assert not {'MolWeight', 'SMILES', 'VdW surface DPSA2'} & set(template)


def test_no_mw_does_not_predict_until_every_row_has_valid_value(fixed21_model, input_rows):
    frame, weights = input_rows
    app = make_app(fixed21_model, frame)
    assert app.button(key='batch_predict').disabled
    assert 'batch_result' not in app.session_state
    for invalid in (np.nan, 0., -1., np.inf):
        values = weights.copy()
        values[1] = invalid
        enter_weights(app, values)
        assert app.button(key='batch_predict').disabled
        assert not app.success
    enter_weights(app, weights)
    assert not app.error
    assert not app.button(key='batch_predict').disabled
    app.button(key='batch_predict').click().run()
    assert not app.exception and not app.error
    assert app.success
    _, result = app.session_state['batch_result']
    assert len(result) == len(frame)
    assert result.Name.tolist() == frame.Name.tolist()
    assert np.isfinite(result['Predict_LLNA EC3']).all()
    invalid = weights.copy()
    invalid[0] = 0.
    enter_weights(app, invalid)
    assert app.button(key='batch_predict').disabled
    assert not app.success  # stale successful results must disappear after an invalid edit


def test_new_upload_cannot_reuse_old_weights_or_prediction(fixed21_model, input_rows):
    frame, weights = input_rows
    app = make_app(fixed21_model, frame)
    old_key = enter_weights(app, weights)
    app.button(key='batch_predict').click().run()
    assert app.success
    changed = frame.copy()
    changed.loc[changed.index[0], 'Name'] = '別の物質'
    app.session_state['_test_uploaded_csv'] = changed.to_csv(index=False).encode('utf-8-sig')
    app.run()
    assert not app.exception
    assert app.session_state['_test_mw_editor_key'] != old_key
    assert app.button(key='batch_predict').disabled
    assert not app.success


def test_duplicate_identifiers_keep_per_row_weights_and_order(fixed21_model, input_rows):
    from original_runtime import load_artifact, predict_with_artifact
    frame, weights = input_rows
    frame['No.'] = 1
    frame['Name'] = '同名物質'
    frame['CAS Number'] = '123-45-6'
    app = make_app(fixed21_model, frame)
    enter_weights(app, weights)
    app.button(key='batch_predict').click().run()
    assert not app.exception and not app.error
    _, result = app.session_state['batch_result']
    assert len(result) == 3
    assert result.Name.tolist() == frame.Name.tolist()
    np.testing.assert_array_equal(result.MolWeight, weights)
    expected = predict_with_artifact(load_artifact(fixed21_model), frame.assign(MolWeight=weights))
    pd.testing.assert_frame_equal(result, expected)


def test_mw_attachment_uses_immutable_row_position_not_duplicate_ids(input_rows):
    from csv_batch_inputs import attach_molecular_weights, mw_entry_frame
    frame, weights = input_rows
    frame['No.'] = 1
    frame['Name'] = '同名'
    frame['CAS Number'] = '123-45-6'
    frame.index = [9, 9, 1]
    editor = mw_entry_frame(frame)
    assert editor.columns.tolist() == ['入力行', 'No.', 'Name', 'CAS Number', 'MolWeight']
    assert editor.MolWeight.isna().all()
    editor['MolWeight'] = weights
    editor = editor.iloc[::-1]
    result = attach_molecular_weights(frame, editor)
    np.testing.assert_array_equal(result.MolWeight, weights)
    assert result.index.tolist() == [9, 9, 1]
    pd.testing.assert_frame_equal(result.drop(columns='MolWeight'), frame)


def test_tampered_mw_row_mapping_is_rejected(input_rows):
    from csv_batch_inputs import attach_molecular_weights, mw_entry_frame
    frame, weights = input_rows
    editor = mw_entry_frame(frame)
    editor['MolWeight'] = weights
    editor.loc[editor.index[1], '入力行'] = editor.loc[editor.index[0], '入力行']
    with pytest.raises(ValueError):
        attach_molecular_weights(frame, editor)
