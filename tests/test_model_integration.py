"""Integration against a separately trained artifact; never trigger training on test/import."""
import joblib
import numpy as np
import pandas as pd
import pytest

from original_runtime import explain_prediction, load_artifact, predict_with_artifact
from utils_io import load_dataset, read_csv_auto


@pytest.fixture(scope="module")
def trained_artifact(smoke_artifact_path):
    return load_artifact(smoke_artifact_path)


@pytest.fixture(scope="module")
def real_inputs(repo_root):
    frame, _ = read_csv_auto(repo_root / "data" / "DASS_join_data_opera_delete_LLNAnan.csv")
    return frame.head(3).drop(columns="LLNA_EC3")


def test_saved_artifact_captures_real_training_conditions(trained_artifact, repo_root):
    frame, _ = read_csv_auto(repo_root / "data" / "DASS_join_data_opera_delete_LLNAnan.csv")
    if trained_artifact['metadata'].get('feature_mode') == 'fixed23':
        from utils_io import FIXED_FEATURE_COLUMNS
        assert trained_artifact['full_feature_columns'] == FIXED_FEATURE_COLUMNS
    else:
        assert trained_artifact["full_feature_columns"] == frame.columns[6:-1].tolist()
    assert len(trained_artifact["folds"]) == 5
    assert trained_artifact["metadata"]["optuna_enabled"] is True
    assert trained_artifact["metadata"]["n_trials"] >= 1
    assert trained_artifact["metadata"]["random_state"] == 1037
    assert trained_artifact["versions"]
    observed_holdouts = []
    for fold in trained_artifact["folds"]:
        train, holdout = set(fold["train_indices"]), set(fold["test_indices"])
        assert not train.intersection(holdout)
        assert len(train | holdout) == len(frame)
        assert fold["optimization"]["trials"]
        assert fold["selected_feature_columns"] == trained_artifact["selected_feature_columns"]
        observed_holdouts.extend(holdout)
    assert sorted(observed_holdouts) == list(range(len(frame)))


def test_real_single_batch_missing_cells_and_archive_roundtrip(trained_artifact, real_inputs, tmp_path):
    inputs = real_inputs.copy()
    imputed_descriptor = next(c for c in trained_artifact["full_feature_columns"] if c != "MolWeight")
    inputs.loc[inputs.index[1], imputed_descriptor] = np.nan
    batch = predict_with_artifact(trained_artifact, inputs)
    singles = pd.concat([predict_with_artifact(trained_artifact, inputs.iloc[[i]]) for i in range(len(inputs))],
                        ignore_index=True)
    pd.testing.assert_frame_equal(batch, singles)
    assert np.isfinite(batch["Predict_LLNA EC3"]).all()
    assert (batch["Predict_LLNA EC3"] > 0).all()
    assert batch.Name.tolist() == inputs.Name.tolist()
    path = tmp_path / "roundtrip.joblib"
    joblib.dump(trained_artifact, path, compress=3)
    pd.testing.assert_frame_equal(batch, predict_with_artifact(load_artifact(path), inputs))


def test_real_fold_predictions_match_direct_saved_pipelines(trained_artifact, real_inputs):
    frame = real_inputs[trained_artifact["full_feature_columns"]]
    result = predict_with_artifact(trained_artifact, real_inputs)
    ec3_by_fold = []
    for fold in trained_artifact["folds"]:
        matrix = fold["pipeline"].transform(frame)[:, fold["selected_feature_mask"]]
        predicted = fold["model"].predict(matrix)
        # Evaluate the notebook arithmetic directly, preserving float32 exponentiation.
        ec3 = frame.MolWeight.to_numpy() / (10 ** predicted)
        np.testing.assert_allclose(result["pred_y_" + fold["fold_name"]], predicted, rtol=0, atol=0)
        np.testing.assert_allclose(result["Predict_LLNA EC3_" + fold["fold_name"]], ec3, rtol=1e-12, atol=1e-12)
        ec3_by_fold.append(ec3)
    np.testing.assert_allclose(result["Predict_LLNA EC3"], np.min(ec3_by_fold, axis=0), rtol=1e-12)


def test_external_report_can_be_reproduced_from_original_csv(trained_artifact, repo_root):
    path = repo_root / "data" / "newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv"
    if not path.is_file():
        pytest.skip("Private external-validation CSV is not distributed with the repository.")
    external = load_dataset(path, feature_columns=trained_artifact["full_feature_columns"], external=True)
    result = predict_with_artifact(trained_artifact, external["frame"])
    saved = trained_artifact["reports"]["external"]
    assert len(result) == 72
    if trained_artifact['metadata'].get('deployment_export'):
        assert not {'Name', 'CAS Number', 'SMILES', 'No.', 'source_row'} & set(saved)
    else:
        assert result.Name.tolist() == saved.Name.tolist()
        np.testing.assert_array_equal(external["source_rows"], saved.source_row)
    for name in ["Predict_LLNA EC3", "pred_y", "kNN_AD", "any_kNN_AD", "selected_fold"]:
        pd.testing.assert_series_equal(result[name], saved[name])
    assert result.attrs["metrics"]["ec3"]["n"] == len(result)


def test_real_fold_shap_explains_that_fold_without_averaging(trained_artifact, real_inputs):
    for fold in trained_artifact["folds"]:
        explanation = explain_prediction(trained_artifact, real_inputs, fold_name=fold["fold_name"])
        assert explanation.feature_names == fold["selected_feature_columns"]
        matrix = fold["pipeline"].transform(real_inputs[trained_artifact["full_feature_columns"]])
        expected = fold["model"].predict(matrix[:, fold["selected_feature_mask"]])
        np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1), expected,
                                   rtol=1e-5, atol=1e-5)
