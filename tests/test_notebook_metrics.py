"""Literal notebook-boundary and row-orientation checks."""
import numpy as np
import pandas as pd
import pytest

from notebook_metrics import classify_ecetoc, classify_ghs, evaluate_notebook_report


def report(observed, predicted, **extra):
    return pd.DataFrame({'LLNA EC3(NS converted to 150%)': observed,
                         'Predict_LLNA EC3': predicted, **extra})


def test_original_ghs_thresholds_use_100_not_150():
    values = [0.01, 2., np.nextafter(2., np.inf), 100., np.nextafter(100., np.inf), 150.]
    np.testing.assert_array_equal(classify_ghs(values), [0, 0, 1, 1, 2, 2])


def test_original_ecetoc_exact_boundaries_go_to_weaker_class():
    values = [.01, np.nextafter(.1, 0), .1, np.nextafter(1., 0), 1.,
              np.nextafter(10., 0), 10., np.nextafter(100., 0), 100., 150.]
    np.testing.assert_array_equal(classify_ecetoc(values), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])


def test_fivefold_boundaries_are_not_in_any_original_mode():
    result = evaluate_notebook_report(report([10.] * 7, [1., 2., 3., 10., 49., 50., 51.]))
    row = result['range_rates'].loc['All']
    assert row['within_count'] == 3
    assert row['over_count'] == 1  # lower EC3 = overestimated sensitization
    assert row['under_count'] == 1
    assert row['boundary_count'] == 2
    assert row['within_rate'] == 3 / 7
    assert row['over_rate'] == row['under_rate'] == 1 / 7
    assert row['boundary_rate'] == 2 / 7


def test_ghs_groups_come_from_observed_values_and_absent_group_is_undefined():
    result = evaluate_notebook_report(report([1., 2., 100., 101.], [1., 20., 1., 101.]))
    ranges = result['range_rates']
    assert ranges.loc['1A', 'n'] == 2
    assert ranges.loc['1A', 'within_rate'] == .5
    assert ranges.loc['1A', 'under_rate'] == .5
    assert ranges.loc['1B', 'n'] == 1
    assert ranges.loc['1B', 'over_rate'] == 1
    assert ranges.loc['NC', 'within_rate'] == 1
    absent = evaluate_notebook_report(report([1.], [1.]))['range_rates']
    assert absent.loc['1B', 'n'] == 0
    assert pd.isna(absent.loc['1B', 'within_rate'])


def test_confusion_rows_are_predicted_and_direction_is_sensitization():
    result = evaluate_notebook_report(report([1., 50., 150.], [50., 50., 1.]))
    matrix = result['ghs']['matrix']
    assert matrix.loc['1B', '1A'] == 1  # underestimated sensitization
    assert matrix.loc['1B', '1B'] == 1
    assert matrix.loc['1A', 'NC'] == 1  # overestimated sensitization
    assert matrix.to_numpy().sum() == 3
    assert result['ghs']['accuracy'] == 1 / 3
    assert result['ghs']['underestimated'] == 1 / 3
    assert result['ghs']['overestimated'] == 1 / 3
    assert matrix.index.name == 'Predicted'
    assert matrix.columns.name == 'Observed'


def test_ecetoc_confusion_counts_and_ordinal_errors():
    result = evaluate_notebook_report(report([.05, .5, 5., 50., 150.], [.5, .05, 5., 150., 50.]))
    matrix = result['ecetoc']['matrix']
    assert matrix.loc['Strong', 'Extreme'] == 1
    assert matrix.loc['Extreme', 'Strong'] == 1
    assert matrix.loc['Moderate', 'Moderate'] == 1
    assert matrix.loc['ns', 'Weak'] == 1
    assert matrix.loc['Weak', 'ns'] == 1
    assert result['ecetoc']['accuracy'] == .2
    assert result['ecetoc']['underestimated'] == .4
    assert result['ecetoc']['overestimated'] == .4


def test_nonfinite_mask_is_positional_with_duplicate_indices():
    frame = report([1., np.nan, 100., 150., 2.], [1., 1., np.inf, 1., 2.])
    frame.index = [8, 8, 9, 9, 8]
    result = evaluate_notebook_report(frame)
    assert result['input_count'] == 5
    assert result['excluded_count'] == 2
    assert result['regression']['n'] == 3
    assert result['ghs']['matrix'].to_numpy().sum() == 3
    assert result['range_rates'].loc['All', 'n'] == 3
    assert frame.index.tolist() == [8, 8, 9, 9, 8]


def test_all_outside_ad_can_be_evaluated_as_zero_rows_without_passing_score():
    frame = report([1., 10.], [1., 10.], knn_inAD_Predict_LLNA_EC3=[np.nan, np.nan])
    result = evaluate_notebook_report(frame, prediction_column='knn_inAD_Predict_LLNA_EC3')
    assert result['excluded_count'] == 2
    assert result['regression'] == {'n': 0, 'r2': None, 'rmse': None, 'mae': None}
    assert result['ghs']['accuracy'] is None
    assert result['ghs']['matrix'].to_numpy().sum() == 0
    assert result['range_rates'].loc['All', 'n'] == 0


def test_mean_and_minimum_are_distinct_evaluation_columns():
    frame = report([1., 100.], [1., 1.], **{'Predict_LLNA EC3_mean': [1., 100.]})
    minimum = evaluate_notebook_report(frame)
    mean = evaluate_notebook_report(frame, prediction_column='Predict_LLNA EC3_mean')
    assert minimum['regression']['rmse'] > 0
    assert mean['regression']['rmse'] == 0
    assert mean['regression']['r2'] == 1
    assert mean['additional_metrics'] == ['mae']


@pytest.mark.parametrize('observed,predicted', [([0.], [1.]), ([1.], [-1.])])
def test_finite_nonpositive_ec3_is_rejected(observed, predicted):
    with pytest.raises(ValueError, match='EC3'):
        evaluate_notebook_report(report(observed, predicted))


def test_metric_boundaries_match_independently_extracted_notebook_functions(repo_root):
    """Execute only the three original pure function definitions, never plots."""
    import ast
    import json
    path = repo_root / 'reference_private' / '★XGBoost_model_refact_tokunaga.ipynb'
    if not path.is_file():
        pytest.skip('Original private notebook is needed for independent source verification.')
    notebook = json.loads(path.read_text())
    namespace = {'np': np}
    parsed = ast.parse(''.join(notebook['cells'][66]['source']))
    original_range = next(node for node in parsed.body
                          if isinstance(node, ast.FunctionDef) and node.name == 'calculate_range_rate')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original_range], type_ignores=[])),
                 str(path) + ':cell66', 'exec'), namespace)
    values = np.asarray([.01, .1, 1., 2., 10., 100., 150.])
    predictions = np.asarray([.002, .5, 1., 11., 1., 100., 150.])
    actual = evaluate_notebook_report(report(values, predictions))['range_rates']
    for group, label in [(None, 'All'), (0, '1A'), (1, '1B'), (2, 'NC')]:
        for mode, key in [(None, 'within'), ('over', 'over'), ('under', 'under')]:
            original = namespace['calculate_range_rate'](values, predictions, cl=group, mode=mode)
            assert actual.loc[label, key + '_rate'] == original[1]
            assert actual.loc[label, key + '_count'] == original[2]
            assert actual.loc[label, 'n'] == original[3]
    for cell, name, current in [(71, 'ecetoc_classification', classify_ecetoc),
                                (72, 'GHS_classification', classify_ghs)]:
        parsed = ast.parse(''.join(notebook['cells'][cell]['source']))
        original = next(node for node in ast.walk(parsed)
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])),
                     str(path) + f':cell{cell}', 'exec'), namespace)
        np.testing.assert_array_equal(current(values), [namespace[name](v) for v in values])
