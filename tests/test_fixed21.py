"""The current 21 descriptors exclude MW from fitting, imputation, AD and SHAP."""
import numpy as np
import pandas as pd
import pytest

from original_runtime import explain_prediction, load_artifact, make_prediction_template, predict_with_artifact
from utils_io import FIXED_FEATURE_COLUMNS, FIXED21_FEATURE_COLUMNS, load_dataset, read_csv_auto

EXPECTED = [
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
def fixed21(repo_root):
    for name in ('fixed21_smoke_model_artifact.joblib', 'original_model_artifact.joblib'):
        path = repo_root / 'artifacts' / name
        if path.exists():
            artifact = load_artifact(path)
            if artifact['metadata'].get('feature_mode') == 'fixed21':
                return artifact
    pytest.skip('A trusted fixed21 model is needed.')


@pytest.fixture
def inputs(repo_root):
    frame, _ = read_csv_auto(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
    return frame[EXPECTED + ['MolWeight']].head(3).copy()


def test_exact_current_features_and_mw_are_separate(repo_root):
    assert FIXED_FEATURE_COLUMNS == FIXED21_FEATURE_COLUMNS == EXPECTED
    assert len(EXPECTED) == 21 and 'MolWeight' not in EXPECTED
    data = load_dataset(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv', feature_columns=EXPECTED)
    assert data['X'].shape == (154, 21)
    assert data['X'].columns.tolist() == EXPECTED
    np.testing.assert_array_equal(data['mw'], data['frame'].MolWeight)


def test_fixed21_training_uses_no_boruta_or_mw_predictor(repo_root, tmp_path, monkeypatch):
    import train
    def forbidden(*args, **kwargs):
        raise AssertionError('Fixed21 must not run Boruta')
    monkeypatch.setattr(train, 'fit_global_boruta', forbidden)
    model = train.train_model(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv',
        mode='smoke', feature_mode='fixed21', n_trials=1, max_estimators=16,
        artifact_path=tmp_path / 'model.joblib', output_dir=tmp_path / 'reports')
    assert model['full_feature_columns'] == model['selected_feature_columns'] == EXPECTED
    assert not model['metadata']['boruta']['enabled']
    assert model['metadata']['mw_role'] == 'conversion_only'
    assert model['metadata']['required_input_columns'] == EXPECTED + ['MolWeight']
    for fold in model['folds']:
        assert fold['pipeline'].feature_names_in_.tolist() == EXPECTED
        assert fold['pipeline'].named_steps['inputer']._fit_X.shape[1] == 21
        assert fold['model'].n_features_in_ == 21


def test_mw_changes_only_inverse_conversion_even_when_features_are_missing(fixed21, inputs):
    inputs.loc[inputs.index[0], 'LUMO Energy'] = np.nan
    doubled = inputs.copy()
    doubled['MolWeight'] *= 2
    original = predict_with_artifact(fixed21, inputs)
    changed = predict_with_artifact(fixed21, doubled)
    for name in original:
        if name.startswith(('pred_y', 'knn_AD_', 'knn_distance_')) or name in ('selected_fold', 'kNN_AD', 'any_kNN_AD', 'selected_inAD_fold'):
            pd.testing.assert_series_equal(original[name], changed[name])
    np.testing.assert_array_equal(original['Predict_LLNA EC3'] * 2, changed['Predict_LLNA EC3'])
    for fold in fixed21['folds']:
        a = explain_prediction(fixed21, inputs, fold_name=fold['fold_name'])
        b = explain_prediction(fixed21, doubled, fold_name=fold['fold_name'])
        assert a.feature_names == EXPECTED
        np.testing.assert_array_equal(a.values, b.values)
        np.testing.assert_array_equal(a.base_values, b.base_values)


@pytest.mark.parametrize('value', [None, 0, -1, np.inf, 'invalid'])
def test_conversion_mw_still_required_and_validated(fixed21, inputs, value):
    inputs['MolWeight'] = inputs['MolWeight'].astype(object)
    inputs.loc[inputs.index[0], 'MolWeight'] = value
    with pytest.raises(ValueError, match='MolWeight'):
        predict_with_artifact(fixed21, inputs)


def test_missing_mw_and_missing_descriptor_are_both_rejected(fixed21, inputs):
    for name in ('MolWeight', 'FM advection sediment'):
        with pytest.raises(ValueError, match='MolWeight|不足'):
            predict_with_artifact(fixed21, inputs.drop(columns=name))


def test_templates_and_label_distinguish_descriptors_from_conversion(fixed21):
    from streamlit_app import _input_columns, _training_label
    assert _input_columns(fixed21) == EXPECTED + ['MolWeight']
    assert len(_input_columns(fixed21)) == 22
    assert make_prediction_template(fixed21).columns.tolist()[-22:] == EXPECTED + ['MolWeight']
    assert _training_label(fixed21).startswith('指定21項目モデル')
