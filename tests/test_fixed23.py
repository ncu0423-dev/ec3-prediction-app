"""User-specified feature order and 23-dimensional preprocessing contract."""
import numpy as np
import pandas as pd
import pytest

from original_runtime import load_artifact, predict_with_artifact
from utils_io import FIXED_FEATURE_COLUMNS, load_dataset, read_csv_auto

EXPECTED = [
    'OVERALL OH rate constant', 'FM advection air', 'VdW surface DPSA2',
    'LUMO Energy', 'HOMO Energy', 'MolWeight', 'CombDipolPolariz',
    'LogHL_pred', 'LogKM_pred', 'alert_Protein_binding_by_OASIS',
    'potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS',
    'DPRA.percCysdep', 'DPRA.percLysdep', 'DPRA.score',
    'hCLAT.CD86.EC150..ug.ml.', 'hCLAT.CD54.EC200..ug.ml.', 'h.CLAT.CV75',
    'hCLAT.MIT', 'hCLAT.score', 'KS.EC1.5', 'KS.EC3', 'KS.IC50', 'sum_invitro',
]


def test_user_features_and_training_input_order(repo_root):
    assert FIXED_FEATURE_COLUMNS == EXPECTED
    data = load_dataset(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv',
                        feature_columns=FIXED_FEATURE_COLUMNS)
    assert data['X'].shape == (154, 23)
    assert data['X'].columns.tolist() == EXPECTED


def test_fixed23_training_never_runs_boruta(repo_root, tmp_path, monkeypatch):
    import train
    def forbidden(*args, **kwargs):
        raise AssertionError('Fixed23 must not run Boruta')
    monkeypatch.setattr(train, 'fit_global_boruta', forbidden)
    artifact = train.train_model(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv',
        mode='smoke', feature_mode='fixed23', n_trials=1, max_estimators=10,
        artifact_path=tmp_path / 'fixed23.joblib', output_dir=tmp_path / 'reports')
    assert artifact['full_feature_columns'] == EXPECTED
    assert artifact['selected_feature_columns'] == EXPECTED
    assert artifact['metadata']['boruta']['enabled'] is False
    assert artifact['metadata']['notebook_reproduction_verified'] is False
    for fold in artifact['folds']:
        assert fold['pipeline'].feature_names_in_.tolist() == EXPECTED
        assert fold['X_train_selected'].shape[1] == 23
        assert len(fold['optimization']['trials']) == 1


def test_fixed23_only_input_predicts_and_ignores_unselected_columns(repo_root):
    path = repo_root / 'artifacts/fixed23_smoke_model_artifact.joblib'
    if not path.exists():
        pytest.skip('Generate the fixed23 smoke model for integration verification.')
    artifact = load_artifact(path)
    frame, _ = read_csv_auto(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
    inputs = frame[EXPECTED].head(3).copy()
    inputs.loc[inputs.index[0], 'LUMO Energy'] = np.nan
    result = predict_with_artifact(artifact, inputs)
    extra = inputs.copy()
    extra['irrelevant_descriptor'] = ['unused'] * len(extra)
    with_extra = predict_with_artifact(artifact, extra)
    pd.testing.assert_frame_equal(result, with_extra.drop(columns='irrelevant_descriptor'))
    assert np.isfinite(result['Predict_LLNA EC3']).all()
    with pytest.raises(ValueError, match='不足'):
        predict_with_artifact(artifact, inputs.drop(columns='KS.EC3'))


def test_fixed23_template_and_model_label(repo_root):
    from streamlit_app import _input_columns, _training_label
    artifact = {'full_feature_columns': EXPECTED,
                'metadata': {'feature_mode': 'fixed23', 'training_mode': 'notebook'}}
    assert _input_columns(artifact) == EXPECTED
    assert _training_label(artifact) == '指定23項目モデル'


def test_notebook_comparison_rejects_fixed23_before_running_reference(repo_root):
    import subprocess
    import sys
    path = repo_root / 'artifacts/fixed23_smoke_model_artifact.joblib'
    if not path.exists():
        pytest.skip('Fixed23 smoke model is needed for this guard check.')
    result = subprocess.run([sys.executable, 'scripts/verify_notebook.py',
        '--use-existing-artifact', '--artifact-path', str(path),
        '--notebook', '/nonexistent/notebook.ipynb'], cwd=repo_root,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert '23項目版' in result.stderr
    assert 'FileNotFoundError' not in result.stderr


def test_parallel_training_keeps_adaptive_optuna_and_predictions_identical(repo_root, tmp_path):
    from train import train_model
    source = repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv'
    runs = [train_model(source, mode='smoke', feature_mode='fixed23',
        n_trials=12, max_estimators=20, fold_workers=workers,
        artifact_path=tmp_path / f'model-{workers}.joblib',
        output_dir=tmp_path / f'reports-{workers}') for workers in (1, 2)]
    pd.testing.assert_frame_equal(runs[0]['reports']['oof'], runs[1]['reports']['oof'])
    for sequential, parallel in zip(runs[0]['folds'], runs[1]['folds']):
        assert sequential['optimization']['trials'] == parallel['optimization']['trials']
        assert sequential['best_params'].keys() == parallel['best_params'].keys()
        for name in ('X_train_selected', 'shap_values', 'shap_base_values'):
            np.testing.assert_array_equal(sequential[name], parallel[name])
