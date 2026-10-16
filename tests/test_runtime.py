"""Prediction invariants tested with transparent fitted fold estimators."""
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest

from original_runtime import load_artifact, make_prediction_template, predict_with_artifact, prepare_input_features


def test_inference_feature_order_and_target_not_required(synthetic_artifact, synthetic_input):
    prepared = prepare_input_features(synthetic_artifact, synthetic_input)
    assert prepared.columns.tolist() == ["x", "MolWeight", "z"]
    np.testing.assert_allclose(prepared.iloc[1], [2., 250., 2.])
    assert "LLNA_EC3" not in synthetic_input
    result = predict_with_artifact(synthetic_artifact, synthetic_input)
    assert len(result) == len(synthetic_input)
    assert result.Name.tolist() == synthetic_input.Name.tolist()
    assert result["CAS Number"].tolist() == ["a", "b", "c"]
    assert result.index.is_unique  # duplicate source indices must not multiply or lose rows


def test_minimum_ec3_uses_y_from_same_fold(synthetic_artifact, synthetic_input):
    result = predict_with_artifact(synthetic_artifact, synthetic_input)
    assert result.selected_fold.tolist() == ["fold_2"] * 3
    np.testing.assert_allclose(result.pred_y, [2.] * 3)
    np.testing.assert_allclose(result["Predict_LLNA EC3"], [.01 * mw for mw in [100, 250, 400]])
    np.testing.assert_allclose(result["pred_y_fold_1"], [1.] * 3)
    np.testing.assert_allclose(result["pred_y_fold_2"], [2.] * 3)
    # Averages remain diagnostic columns and cannot replace the selected result.
    assert np.all(result.pred_y != result.pred_y_mean)


def test_ad_candidate_selection_and_no_eligible_folds(synthetic_artifact, synthetic_input):
    result = predict_with_artifact(synthetic_artifact, synthetic_input)
    assert result.kNN_AD.tolist() == [False] * 3
    assert result.any_kNN_AD.tolist() == [True] * 3
    assert result.selected_inAD_fold.tolist() == ["fold_1"] * 3
    np.testing.assert_allclose(result.knn_inAD_pred_y, [1.] * 3)
    np.testing.assert_allclose(result["knn_inAD_Predict_LLNA EC3"], [10., 25., 40.])

    synthetic_artifact["folds"][0]["ad_threshold"] = 0.
    result = predict_with_artifact(synthetic_artifact, synthetic_input)
    assert not result.any_kNN_AD.any()
    assert result.selected_inAD_fold.isna().all()
    assert result.knn_inAD_pred_y.isna().all()
    assert result["knn_inAD_Predict_LLNA EC3"].isna().all()


def test_uncomputed_ad_is_not_reported_as_inside(synthetic_artifact, synthetic_input):
    result = predict_with_artifact(synthetic_artifact, synthetic_input, compute_ad=False)
    assert result.kNN_AD.isna().all()
    assert result.any_kNN_AD.isna().all()


def test_single_batch_and_missing_cells_agree(synthetic_artifact, synthetic_input):
    synthetic_input.iloc[1, synthetic_input.columns.get_loc("x")] = np.nan
    synthetic_input.iloc[2, synthetic_input.columns.get_loc("z")] = np.nan
    batch = predict_with_artifact(synthetic_artifact, synthetic_input)
    single_results = [predict_with_artifact(synthetic_artifact, synthetic_input.iloc[[i]]) for i in range(3)]
    singles = pd.concat(single_results, ignore_index=True)
    pd.testing.assert_frame_equal(batch, singles)


@pytest.mark.parametrize("column", ["x", "z", "MolWeight"])
def test_missing_full_descriptor_columns_rejected(synthetic_artifact, synthetic_input, column):
    with pytest.raises(ValueError, match=column):
        predict_with_artifact(synthetic_artifact, synthetic_input.drop(columns=column))


def test_real_fitted_objects_survive_joblib_roundtrip(tmp_path, synthetic_artifact, synthetic_input):
    path = tmp_path / "trusted_test_artifact.joblib"
    joblib.dump(synthetic_artifact, path)
    loaded = load_artifact(path)
    pd.testing.assert_frame_equal(predict_with_artifact(synthetic_artifact, synthetic_input),
                                  predict_with_artifact(loaded, synthetic_input))
    template = make_prediction_template(loaded)
    assert "MolWeight" in template.columns
    assert "LLNA_EC3" not in template.columns
    assert template.empty


def test_wrong_feature_order_in_artifact_is_rejected(tmp_path, synthetic_artifact):
    corrupt = deepcopy(synthetic_artifact)
    corrupt["selected_feature_columns"] = ["z", "x"]
    path = tmp_path / "corrupt.joblib"
    joblib.dump(corrupt, path)
    with pytest.raises(ValueError):
        load_artifact(path)


def test_legacy_bundle_is_not_silently_accepted(tmp_path):
    path = tmp_path / "legacy.joblib"
    joblib.dump({"pipeline": "old", "feature_columns": ["x"]}, path)
    with pytest.raises(ValueError):
        load_artifact(path)


def test_truth_is_only_used_for_evaluation(synthetic_artifact, synthetic_input):
    baseline = predict_with_artifact(synthetic_artifact, synthetic_input)
    labelled = synthetic_input.assign(LLNA_EC3=[2., 10., 150.])
    evaluated = predict_with_artifact(synthetic_artifact, labelled)
    np.testing.assert_array_equal(baseline["Predict_LLNA EC3"], evaluated["Predict_LLNA EC3"])
    assert evaluated.attrs["metrics"]["ec3"]["n"] == 3
    assert "true_y" in evaluated
