"""Regression contracts for the original notebook preprocessing and final AD branch.

The immutable full-cost source-validated artifact supplies the learned weights.
AD expectations below use the notebook's literal distance/order-statistic math.
"""
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from original_runtime import _fold_inputs, load_artifact, predict_with_artifact
from utils_io import load_dataset, read_csv_auto
from utils_ml import library_versions, make_preprocess_pipeline


@pytest.fixture(scope='module')
def canonical(repo_root):
    path = repo_root / 'artifacts/notebook_model_artifact.joblib'
    if not path.is_file():
        pytest.skip('The immutable, locally validated original notebook artifact is unavailable.')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '0d1dc76ecb32cb3d6b3d00974a51bcf78161d7be323917d38179e679052bb4ad'
    return load_artifact(path)


@pytest.fixture(scope='module')
def restored(repo_root):
    path = repo_root / 'artifacts/notebook_restored_model_artifact.joblib'
    if not path.is_file():
        pytest.skip('Generate the restored original notebook artifact before these integration checks.')
    return load_artifact(path)


@pytest.fixture(scope='module')
def original_training(repo_root):
    frame, _ = read_csv_auto(repo_root / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
    return frame


def test_original_schema_boruta_and_full_cost_history(canonical, restored, original_training):
    full = original_training.columns[6:-1].tolist()
    assert len(full) == 452 and 'MolWeight' in full
    assert restored['full_feature_columns'] == canonical['full_feature_columns'] == full
    mask = np.asarray(canonical['global_boruta'].support_, dtype=bool)
    np.testing.assert_array_equal(restored['selected_feature_mask'], mask)
    assert restored['selected_feature_columns'] == np.asarray(full)[mask].tolist()
    assert restored['metadata']['feature_mode'] == 'boruta'
    assert restored['metadata']['n_trials'] == 1000
    assert restored['metadata']['max_estimators'] == 100000
    assert restored['metadata']['random_state'] == 1037
    assert restored['metadata']['ad_coverage'] == .2
    assert restored['versions'] == library_versions()
    for fold in restored['folds']:
        assert fold['pipeline'].feature_names_in_.tolist() == full
        assert fold['pipeline'].n_features_in_ == 452
        assert fold['pipeline'].named_steps['inputer']._fit_X.shape[1] == 452
        assert fold['pipeline'].named_steps['inputer'].n_neighbors == 5
        assert fold['pipeline'].named_steps['inputer'].weights == 'uniform'
        assert fold['model'].n_features_in_ == int(mask.sum())
        assert len(fold['optimization']['trials']) == 1000


def test_selected_columns_alone_cannot_replace_full_preprocessing(restored, original_training):
    minimal = list(dict.fromkeys(restored['selected_feature_columns'] + ['MolWeight']))
    assert len(minimal) < 452
    with pytest.raises(ValueError, match='不足'):
        predict_with_artifact(restored, original_training[minimal].head(1))


def test_unselected_descriptor_really_changes_selected_missing_value_imputation():
    # Five donors with selected=0 vs five with selected=100. Only the UNSELECTED
    # descriptor distinguishes donor groups; MW is fixed. Dropping it loses the
    # information needed to impute the selected column as the original does.
    training = pd.DataFrame({
        'selected': [0.] * 5 + [100.] * 5,
        'unselected': [0.] * 5 + [100.] * 5,
        'MolWeight': [100.] * 10,
    })
    query = pd.DataFrame({'selected': [np.nan, np.nan],
                          'unselected': [0., 100.], 'MolWeight': [100., 100.]})
    pipeline = make_preprocess_pipeline().fit(training)
    imputed = pipeline.named_steps['inputer'].transform(query)
    np.testing.assert_array_equal(imputed[:, 0], [0., 100.])
    artifact = {'full_feature_columns': training.columns.tolist(),
                'selected_feature_mask': np.array([True, False, False])}
    selected = _fold_inputs(artifact, query, {'pipeline': pipeline})
    np.testing.assert_array_equal(selected[:, 0], [-1., 1.])


def test_restoration_does_not_retrain_or_modify_learned_models(canonical, restored, original_training):
    inputs = original_training.drop(columns='LLNA_EC3')
    old = predict_with_artifact(canonical, inputs)
    new = predict_with_artifact(restored, inputs)
    prediction_columns = [c for c in old if c.startswith(('pred_y', 'Predict_LLNA EC3'))]
    pd.testing.assert_frame_equal(old[prediction_columns], new[prediction_columns])
    for old_fold, new_fold in zip(canonical['folds'], restored['folds']):
        assert old_fold['model'].get_booster().save_raw() == new_fold['model'].get_booster().save_raw()
        np.testing.assert_array_equal(old_fold['pipeline'].named_steps['inputer']._fit_X,
                                      new_fold['pipeline'].named_steps['inputer']._fit_X)
        for name in ('mean_', 'scale_'):
            np.testing.assert_array_equal(getattr(old_fold['pipeline'].named_steps['standarization'], name),
                                          getattr(new_fold['pipeline'].named_steps['standarization'], name))
        np.testing.assert_array_equal(old_fold['shap_values'], new_fold['shap_values'])
        assert old_fold['optimization']['trials'] == new_fold['optimization']['trials']


def _literal_threshold(training):
    distances = np.sort(cdist(training, training), axis=1)
    sorted_means = np.sort(distances[:, 1:6].mean(axis=1))
    return sorted_means[round(len(training) * (1 - .2)) - 1]


def test_final_notebook_ad_order_statistic_and_oof_flags(restored, original_training):
    saved = restored['reports']['oof']
    for fold in restored['folds']:
        training = fold['X_train_selected']
        threshold = _literal_threshold(training)
        assert fold['ad_threshold'] == threshold
        rows = fold['test_indices']
        transformed = fold['pipeline'].transform(original_training[restored['full_feature_columns']].iloc[rows])
        query = transformed[:, restored['selected_feature_mask']]
        means = np.sort(cdist(query, training), axis=1)[:, :5].mean(axis=1)
        np.testing.assert_array_equal(saved.iloc[rows]['kNN_AD'], means <= threshold)
        np.testing.assert_array_equal(saved.iloc[rows]['knn_distance'], means)


def test_external_ad_eligible_aggregation_and_saved_report(restored, repo_root):
    path = repo_root / 'data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv'
    if not path.is_file():
        pytest.skip('Private external validation input is not distributed.')
    external = load_dataset(path, feature_columns=restored['full_feature_columns'], external=True)
    frame = external['frame'].drop(columns='LLNA_EC3')
    result = predict_with_artifact(restored, frame)
    inside, ec3 = [], []
    for fold in restored['folds']:
        selected = fold['pipeline'].transform(frame[restored['full_feature_columns']])[:, restored['selected_feature_mask']]
        distances = np.sort(cdist(selected, fold['X_train_selected']), axis=1)[:, :5].mean(axis=1)
        expected = distances <= _literal_threshold(fold['X_train_selected'])
        np.testing.assert_array_equal(result['knn_AD_' + fold['fold_name']], expected)
        inside.append(expected)
        ec3.append(result['Predict_LLNA EC3_' + fold['fold_name']].to_numpy())
    eligible = np.asarray(inside).T
    values = np.asarray(ec3).T
    np.testing.assert_array_equal(result['any_kNN_AD'], eligible.any(axis=1))
    np.testing.assert_array_equal(result['Predict_LLNA EC3'], values.min(axis=1))
    best_eligible = np.where(eligible, values, np.inf).min(axis=1)
    best_eligible[~eligible.any(axis=1)] = np.nan
    np.testing.assert_array_equal(result['knn_inAD_Predict_LLNA EC3'], best_eligible)
    saved = restored['reports']['external']
    for name in ('Predict_LLNA EC3', 'pred_y', 'any_kNN_AD', 'knn_inAD_Predict_LLNA EC3'):
        pd.testing.assert_series_equal(result[name], saved[name])
    # Match the original CSV report's ratio using its round-tripped EC3 truth,
    # including NaN when no eligible fold exists; do not substitute raw labels.
    truth_y = np.log10(external['mw'] / external['ec3'])
    truth_ec3 = external['mw'] / (10 ** truth_y)
    np.testing.assert_array_equal(saved['Fold_misprediction'], values.min(axis=1) / truth_ec3)
    np.testing.assert_array_equal(saved['knn_inAD_Fold_misprediction'], best_eligible / truth_ec3)


def test_default_training_cli_uses_boruta_and_last_notebook_ad_branch(monkeypatch):
    import train
    captured = {}
    monkeypatch.setattr(sys, 'argv', ['train.py'])
    monkeypatch.setattr(train, 'train_model', lambda *args, **kwargs: captured.update(kwargs))
    train.main()
    assert captured['feature_mode'] == 'boruta'
    assert captured['ad_coverage'] == .2
    assert train.resolve_ad_coverage('boruta', None) == .2
    assert train.resolve_ad_coverage('boruta', .1) == .1
    assert train.resolve_ad_coverage('fixed21', None) == .1


def test_current_ui_exposes_full_input_and_full_original_shap_reference(restored):
    from streamlit_app import _input_columns, _training_explanation, _training_label
    assert _input_columns(restored) == restored['full_feature_columns']
    assert len(_input_columns(restored)) == 452
    assert _training_label(restored) == 'Notebook再現条件'
    explanation = _training_explanation(restored['folds'][0])
    assert len(explanation.values) == len(restored['folds'][0]['X_train_selected'])
    assert explanation.feature_names == restored['selected_feature_columns']
