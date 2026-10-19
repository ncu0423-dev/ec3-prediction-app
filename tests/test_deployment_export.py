import pandas as pd
import numpy as np

from original_runtime import load_artifact, predict_with_artifact
from scripts.export_deployment_model import build_deployment_artifact


def test_export_removes_identifiers_without_mutating_source(synthetic_artifact, synthetic_input):
    source = synthetic_artifact
    source['global_selection_transformed'] = np.ones((2, 3))
    source['train_target_y'] = np.ones(2)
    source['reports'] = {'external': pd.DataFrame({
        'Name': ['private substance'], 'CAS Number': ['private identifier'],
        'SMILES': ['private structure'], 'source_row': [12],
        'true_y': [1.], 'pred_y': [1.2], 'Predict_LLNA EC3': [10.],
        'selected_fold': ['fold_1'], 'unexpected_text': ['private note'],
    })}
    exported = build_deployment_artifact(source)
    assert 'global_selection_transformed' not in exported
    assert 'train_target_y' not in exported
    assert list(exported['reports']['external']) == [
        'selected_fold', 'true_y', 'pred_y', 'Predict_LLNA EC3']
    assert 'Name' in source['reports']['external']
    assert 'deployment_export' not in source['metadata']
    pd.testing.assert_frame_equal(predict_with_artifact(source, synthetic_input),
                                  predict_with_artifact(exported, synthetic_input))


def test_deployed_real_model_matches_canonical_predictions(repo_root):
    deployed = load_artifact(repo_root / 'artifacts' / 'original_model_artifact.joblib')
    reference_name = ('fixed23_model_artifact.joblib' if deployed['metadata'].get('feature_mode') == 'fixed23'
                      else 'notebook_model_artifact.joblib')
    canonical_path = repo_root / 'artifacts' / reference_name
    if not canonical_path.is_file():
        import pytest
        pytest.skip('Canonical local model is needed for export equivalence verification.')
    from utils_io import read_csv_auto
    canonical = load_artifact(canonical_path)
    frame, _ = read_csv_auto(repo_root / 'data' / 'DASS_join_data_opera_delete_LLNAnan.csv')
    inputs = frame.drop(columns='LLNA_EC3')
    pd.testing.assert_frame_equal(predict_with_artifact(canonical, inputs),
                                  predict_with_artifact(deployed, inputs))
    assert deployed['metadata']['deployment_export']['report_identifiers_removed']
    for report in deployed['reports'].values():
        assert not {'Name', 'CAS Number', 'SMILES', 'No.', 'source_row'} & set(report)
