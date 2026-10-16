#!/usr/bin/env python3
"""Compare the app with the original notebook class, independently of app formulas.

Only the class definition is executed. Cell paths, searches, plots, file writes and
10,000 resampling runs are never executed. Default validation bounds are explicitly
NOT the notebook's 1,000-trial production fit. User-supplied notebooks must be trusted
because Python class bodies are executable code.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from boruta import BorutaPy
import optuna
from optuna.samplers import TPESampler
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from utils_io import save_json
from utils_ml import library_versions

# A repeated DataFrame-insertion warning in the original source does not affect values.
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class ValidationBounds(ast.NodeTransformer):
    """Change explicit training cost controls, never mathematical expressions."""
    def __init__(self, trials, estimators):
        self.trials, self.estimators = trials, estimators
        self.replacements = []

    def visit_Dict(self, node):
        self.generic_visit(node)
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == 'n_estimators'
                    and isinstance(value, ast.Constant) and value.value == 100000):
                value.value = self.estimators
                self.replacements.append('base n_estimators: 100000 -> ' + str(self.estimators))
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'optimize':
            for item in node.keywords:
                if item.arg == 'n_trials' and isinstance(item.value, ast.Constant) and item.value.value == 1000:
                    item.value.value = self.trials
                    self.replacements.append('Optuna n_trials: 1000 -> ' + str(self.trials))
        return node


def notebook_class(path, trials, estimators):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    found = []
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] != 'code' or 'class train_test_XGBR_cv_plus_exval' not in source:
            continue
        parsed = ast.parse(source)
        for node in parsed.body:
            if isinstance(node, ast.ClassDef) and node.name == 'train_test_XGBR_cv_plus_exval':
                found.append((index, node))
    if len(found) != 1:
        raise ValueError('Expected exactly one original model class in the trusted notebook.')
    index, class_definition = found[0]
    bounded = ValidationBounds(trials, estimators)
    class_definition = bounded.visit(class_definition)
    module = ast.fix_missing_locations(ast.Module(body=[class_definition], type_ignores=[]))
    if len(bounded.replacements) != 2:
        raise ValueError('Notebook cost controls differ from the audited source; review before running.')
    namespace = dict(globals())
    namespace['copy'] = __import__('copy')
    exec(compile(module, str(path) + f':cell{index}', 'exec'), namespace)
    return namespace['train_test_XGBR_cv_plus_exval'], notebook, bounded.replacements


def assert_numeric(checks, name, expected, actual, *, atol=0, rtol=0):
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape:
        raise AssertionError(f'{name}: shape {a.shape} != {b.shape}')
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True, err_msg=name)
    finite = np.isfinite(a) & np.isfinite(b)
    difference = float(np.max(np.abs(a[finite].astype(float)-b[finite].astype(float)))) if finite.any() else 0.
    checks[name] = {'status': 'passed', 'shape': list(a.shape), 'max_absolute_difference': difference,
                    'atol': atol, 'rtol': rtol}


def assert_params(checks, name, expected, actual):
    if set(expected) != set(actual):
        raise AssertionError(f'{name}: parameter keys differ: {set(expected)^set(actual)}')
    for key in expected:
        a, b = expected[key], actual[key]
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
            continue
        if a != b:
            raise AssertionError(f'{name}: {key}: {a!r} != {b!r}')
    checks[name] = {'status': 'passed', 'parameter_count': len(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT/'reference_private/★XGBoost_model_refact_tokunaga.ipynb')
    parser.add_argument('--train-csv', type=Path, default=ROOT/'data/DASS_join_data_opera_delete_LLNAnan.csv')
    parser.add_argument('--exval-csv', type=Path, default=ROOT/'data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv')
    parser.add_argument('--artifact-path', type=Path, default=ROOT/'artifacts/notebook_validation.joblib')
    parser.add_argument('--use-existing-artifact', action='store_true')
    parser.add_argument('--wait-for-artifact-seconds', type=int, default=0)
    parser.add_argument('--ad-coverage-only', action='store_true', help='Use saved matrices/models to compare all original AD branches without fitting.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--n-trials', type=int, default=2)
    parser.add_argument('--max-estimators', type=int, default=200)
    parser.add_argument('--ad-coverage', type=float, default=.1)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT/('outputs/notebook_ad_coverage_comparison.json' if args.ad_coverage_only else 'outputs/notebook_comparison.json')
    if args.n_trials < 1 or args.max_estimators < 16:
        parser.error('Validation needs positive trials and at least 16 trees for early stopping.')
    started = time.monotonic()
    canonical_cost = args.n_trials == 1000 and args.max_estimators == 100000
    report = {'scope': 'full_notebook_source_parity' if canonical_cost else 'bounded_optuna_source_parity', 'historical_notebook_output_parity': 'unavailable',
              'production_1000_trials_verified': False, 'versions': library_versions(), 'checks': {},
              'conditions': {'random_state': 1037, 'cv_folds': 5, 'boruta_perc': 100,
                             'n_trials_per_outer_fold': args.n_trials, 'max_estimators': args.max_estimators,
                             'knn_k': 5, 'ad_coverage': args.ad_coverage},
              'files': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (args.notebook, args.train_csv, args.exval_csv)}}
    checks = report['checks']
    try:
        cls, notebook, adjustments = notebook_class(args.notebook, args.n_trials, args.max_estimators)
        report['source_adjustments_for_cost_only'] = adjustments
        report['stored_execution'] = {
            'final_model_execution_count': notebook['cells'][42].get('execution_count'),
            'final_model_output_count': len(notebook['cells'][42].get('outputs', [])),
            'last_error': [{'cell': i, 'name': o.get('ename'), 'message': o.get('evalue')}
                           for i,c in enumerate(notebook['cells']) for o in c.get('outputs', [])
                           if o.get('output_type') == 'error']}
        # This data extraction intentionally uses only literal notebook statements.
        dass = pd.read_csv(args.train_csv, encoding='cp932')
        newce_raw = pd.read_csv(args.exval_csv, encoding='cp932')
        newce = newce_raw.dropna(subset=['MolWeight'], axis=0).reset_index(drop=True)
        X, ec3 = dass.iloc[:, :-1].iloc[:, 6:], dass.iloc[:, -1]
        external_X, external_ec3 = newce.iloc[:, :-1].iloc[:, 6:], newce.iloc[:, -1]
        reference = cls()
        reference.SET_config(random_state=1037, feature_selection=True, parameter_tuning=True, cv_folds=5)
        reference.SET_train_test_data(df_X=X, Name_df=dass.iloc[:, :-1], y_series=ec3)
        reference.SET_exval_data(df_X=external_X, y_series=external_ec3)
        if args.ad_coverage_only:
            from original_runtime import load_artifact
            artifact = load_artifact(args.artifact_path)
            initialize_reference_ad_from_artifact(reference, artifact)
            compare_ad_coverage_branches(reference, artifact, checks)
            report['scope'] = 'original_class_AD_branches_from_saved_model'
            report['conditions']['n_trials_per_outer_fold'] = artifact['metadata']['n_trials']
            report['conditions']['max_estimators'] = artifact['metadata']['max_estimators']
            report['model_sha256'] = hashlib.sha256(args.artifact_path.read_bytes()).hexdigest()
            report['artifact'] = {'path': str(args.artifact_path.resolve()), 'sha256': report['model_sha256']}
            report['training_executed'] = False
            report['conditions']['AD_coverage_branches'] = [0., .1, .2]
            report['status'] = 'passed'
            return
        reference.SET_preprocess_pipeline()
        reference.SET_feature_selection_by_Boruta(rf_depth=5, perc=100)
        print(f'Independent notebook class: full Boruta + Optuna {args.n_trials} trials / {args.max_estimators} rounds, 5 folds.', flush=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        reference.CONDUCT_pipeline_preprocessed_boruta()
        reference.CONSTRUCT_train_test_xgb_cv_model()
        reference.PREDICT_xgb_cv_model()
        reference.SET_knn_ad_config(knn_k=5, AD_coverage=float(args.ad_coverage))
        reference.CALCULATE_KNN_AD_cv()
        reference.SET_newpred_data(df_X=external_X, y_series=external_ec3)
        reference.PREDICT_newpred_xgb_cv_model()
        print('Independent reference fitted. Comparing app pipeline.', flush=True)
        if args.use_existing_artifact:
            from original_runtime import load_artifact
            deadline = time.monotonic() + args.wait_for_artifact_seconds
            while not args.artifact_path.is_file() and time.monotonic() < deadline:
                print('Independent reference is ready; waiting for backend artifact.', flush=True)
                time.sleep(min(10, max(0, deadline-time.monotonic())))
            artifact = load_artifact(args.artifact_path)
        else:
            from train import train_model
            artifact = train_model(args.train_csv, args.exval_csv, artifact_path=args.artifact_path,
                                   output_dir=args.output.parent/'notebook_validation', mode='smoke',
                                   n_trials=args.n_trials, max_estimators=args.max_estimators,
                                   ad_coverage=args.ad_coverage)
        report['artifact'] = {'path': str(args.artifact_path.resolve()),
                              'sha256': hashlib.sha256(args.artifact_path.read_bytes()).hexdigest()}
        compare_artifact(reference, artifact, dass, newce, args, checks)
        compare_ad_coverage_branches(reference, artifact, checks)
        report['conditions']['AD_coverage_branches'] = [0., .1, .2]
        report['status'] = 'passed'
        report['production_1000_trials_verified'] = canonical_cost
        report['selected_features'] = X.columns[reference._feature_selected_fold_flag].tolist()
        report['n_train'], report['n_exval'] = len(dass), len(newce)
        report['limitations'] = [
            ('This validates full-cost executable notebook source in the current environment.' if canonical_cost
             else 'This validates executable notebook source in the current environment with reduced Optuna cost.'),
            'No saved final-model predictions, masks or tuned parameters are present in the uploaded notebook.',
            ('Historical predictions from the original PC remain unavailable for comparison.' if canonical_cost
             else 'The original 1,000-trial-per-fold/100,000-tree bound production run is not certified by this report.'),
            'Missing external hCLAT.Call source files prevent faithful combined in vitro AD comparison.']
    except Exception as error:
        report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic()-started, 3)
        save_json(report, args.output)
        print(f'Comparison report: {args.output}', flush=True)


def compare_artifact(reference, artifact, dass, newce, args, checks):
    """Backend field contracts are inspected strictly so missing checks cannot pass."""
    from original_runtime import predict_with_artifact
    from utils_io import load_dataset
    from utils_ml import compute_knn_ad, convert_ec3_to_y, convert_y_to_ec3, make_llna_labels
    source_X = reference._traintest_df_org['X_data']
    ds = load_dataset(args.train_csv)
    ex = load_dataset(args.exval_csv, feature_columns=source_X.columns, external=True)
    if list(ds['X'].columns) != source_X.columns.tolist():
        raise AssertionError('Input feature names/order differ from the notebook.')
    checks['input_feature_names_order'] = {'status': 'passed', 'count': len(source_X.columns)}
    assert_numeric(checks, 'raw_X_including_missing', source_X, ds['X'])
    assert_numeric(checks, 'external_MW_filter_and_order', reference._exval_df_org['X_data'], ex['X'])
    assert_numeric(checks, 'target_transform', reference._traintest_df_org['y_data'],
                   convert_ec3_to_y(ds['ec3'], ds['mw']))
    roundtrip = convert_y_to_ec3(reference._traintest_df_org['y_data'], ds['mw'])
    assert_numeric(checks, 'stratification_labels_after_roundtrip', reference._traintest_df_org['LLNA_label'],
                   make_llna_labels(roundtrip))
    assert_numeric(checks, 'global_preprocessing', reference._config['Preprocess_pipeline'].fit_transform(source_X), artifact['global_selection_transformed'])
    assert_numeric(checks, 'Boruta_support', reference._feature_selected_fold_flag,
                   artifact['selected_feature_mask'])
    if artifact['metadata']['n_trials'] != args.n_trials or artifact['metadata']['max_estimators'] != args.max_estimators:
        raise AssertionError('Existing artifact uses different Optuna validation limits.')
    folds = artifact['folds']
    if isinstance(folds, dict):
        folds = [dict(value, fold_name=name) for name,value in folds.items()]
    if len(folds) != 5:
        raise AssertionError('Expected all 5 outer models.')
    for number, fold in enumerate(folds, 1):
        name = f'fold_{number}'
        original = reference._cv_fold_data[name]
        assert_numeric(checks, name+'.train_indices', original['Train_test_index']['Train'], fold['train_indices'])
        assert_numeric(checks, name+'.test_indices', original['Train_test_index']['Test'], fold['test_indices'])
        assert_params(checks, name+'.xgb_parameters', original['Best_params'], fold['best_params'])
        for scope, frame in [('train', source_X.iloc[fold['train_indices']]),
                             ('test', source_X.iloc[fold['test_indices']]),
                             ('external', reference._exval_df_org['X_data'])]:
            source_scope = {'train': 'Train_data', 'test': 'Test_data', 'external': 'Exval_data'}[scope]
            transformed = fold['pipeline'].transform(frame)[:, artifact['selected_feature_mask']]
            assert_numeric(checks, name+'.'+scope+'_preprocessed', original[source_scope]['X_data'], transformed)
            predicted = fold['model'].predict(transformed)
            assert_numeric(checks, name+'.'+scope+'_prediction_y', original[source_scope]['y_pred_data'], predicted)
            # Preserve XGBoost float32 exponentiation before division, as the notebook does.
            predicted_ec3 = convert_y_to_ec3(predicted, frame['MolWeight'].to_numpy())
            assert_numeric(checks, name+'.'+scope+'_prediction_EC3',
                           original[source_scope]['y_pred_reconverted'], predicted_ec3)
            actual_ad, means, threshold = compute_knn_ad(fold['X_train_selected'], transformed,
                                                        knn_k=5, ad_coverage=args.ad_coverage)
            if scope == 'external':
                assert_numeric(checks, name+'.external_AD', reference._exval_pred_result['AD'][name], actual_ad)
            elif scope == 'test':
                idx = fold['test_indices']
                assert_numeric(checks, name+'.holdout_AD', np.asarray(reference._test_pred_result['AD'])[idx], actual_ad)
            train_distances = cdist(original['Train_data']['X_data'], original['Train_data']['X_data'])
            train_distances.sort()
            training_means = np.mean(train_distances[:, 1:6], axis=1)
            training_means.sort()
            expected_threshold = training_means[round(len(training_means)*(1-args.ad_coverage))-1]
            assert_numeric(checks, name+'.'+scope+'_AD_threshold', expected_threshold, threshold)
        import shap
        original_shap = shap.TreeExplainer(original['Studied_model'])(original['Train_data']['X_data'])
        assert_numeric(checks, name+'.training_SHAP', original_shap.values, fold['shap_values'])
        assert_numeric(checks, name+'.training_SHAP_base', original_shap.base_values, fold['shap_base_values'])
    assert_numeric(checks, 'saved_OOF_y', reference._test_pred_result['y_pred_data'], artifact['reports']['oof']['pred_y'])
    assert_numeric(checks, 'saved_OOF_EC3', reference._test_pred_result['y_pred_reconverted'], artifact['reports']['oof']['Predict_LLNA EC3'])
    assert_numeric(checks, 'saved_OOF_AD', reference._test_pred_result['AD'], artifact['reports']['oof']['kNN_AD'])
    assert_numeric(checks, 'last_fold_Optuna_trial_scores', [t.value for t in reference._optuna.trials], [t['value'] for t in folds[-1]['optimization']['trials']])
    assert_numeric(checks, 'last_fold_Optuna_inner_best_iterations', reference._best_iterators_list, [t['user_attrs']['best_iterations'] for t in folds[-1]['optimization']['trials']])
    output = predict_with_artifact(artifact, newce.drop(columns=['LLNA_EC3']))
    if isinstance(output, tuple):
        output = output[0]
    pred_y = reference._exval_pred_result['y_pred_data'].drop(columns=['y_data'])
    pred_ec3 = reference._exval_pred_result['y_pred_reconverted'].drop(columns=['y_reconverted'])
    ad = reference._exval_pred_result['AD']
    assert_numeric(checks, 'final_max_y', pred_y.max(axis=1), output['pred_y'])
    assert_numeric(checks, 'final_min_EC3', pred_ec3.min(axis=1), output['Predict_LLNA EC3'])
    assert_numeric(checks, 'AD_eligible_max_y', pred_y[ad].max(axis=1), output['knn_inAD_pred_y'])
    assert_numeric(checks, 'AD_eligible_min_EC3', pred_ec3[ad].min(axis=1), output['knn_inAD_Predict_LLNA EC3'])
    for column in ('No.', 'Name', 'CAS Number', 'SMILES'):
        if output[column].tolist() != newce[column].tolist():
            raise AssertionError('Prediction metadata/order differs: '+column)
    checks['external_row_IDs_and_order'] = {'status': 'passed', 'row_count': len(newce)}


def initialize_reference_ad_from_artifact(reference, artifact):
    """Populate only the original AD method's matrix inputs; never fit a model."""
    original_X = reference._traintest_df_org['X_data']
    external_X = reference._exval_df_org['X_data']
    mask = np.asarray(artifact['selected_feature_mask'], dtype=bool)
    reference._cv_fold_data = {}
    reference._test_pred_result, reference._exval_pred_result = {}, {}
    for fold in artifact['folds']:
        indices = {'Train': fold['train_indices'], 'Test': fold['test_indices']}
        reference._cv_fold_data[fold['fold_name']] = {
            'Train_test_index': indices,
            'Train_data': {'X_data': fold['X_train_selected']},
            'Test_data': {'X_data': fold['pipeline'].transform(original_X.iloc[indices['Test']])[:, mask]},
            'Exval_data': {'X_data': fold['pipeline'].transform(external_X)[:, mask]},
        }


def compare_ad_coverage_branches(reference, artifact, checks):
    """Use the unmodified original CALCULATE_KNN_AD_cv for 0/.1/.2 branches."""
    from original_runtime import predict_with_artifact
    from utils_ml import fit_knn_ad, compute_knn_ad
    external_X = reference._exval_df_org['X_data']
    for coverage in (0., .1, .2):
        prefix = 'AD_coverage_' + str(coverage)
        reference.SET_knn_ad_config(knn_k=5, AD_coverage=coverage)
        reference.CALCULATE_KNN_AD_cv()
        branch = dict(artifact)
        branch['metadata'] = dict(artifact['metadata'], ad_coverage=coverage)
        branch['folds'] = []
        for fold in artifact['folds']:
            name = fold['fold_name']
            source_fold = reference._cv_fold_data[name]
            train_X = source_fold['Train_data']['X_data']
            distance = cdist(train_X, train_X)
            distance.sort()
            means = np.mean(distance[:, 1:6], axis=1)
            means.sort()
            threshold = means[round(len(train_X)*(1-coverage))-1]
            actual_threshold = fit_knn_ad(fold['X_train_selected'], knn_k=5, ad_coverage=coverage)
            assert_numeric(checks, prefix+'.'+name+'.threshold', threshold, actual_threshold)
            for scope in ('Test', 'Exval'):
                query = source_fold[scope+'_data']['X_data']
                inside, _, _ = compute_knn_ad(fold['X_train_selected'], query, knn_k=5, ad_coverage=coverage)
                expected = (np.asarray(reference._test_pred_result['AD'])[fold['test_indices']]
                            if scope == 'Test' else reference._exval_pred_result['AD'][name])
                assert_numeric(checks, prefix+'.'+name+'.'+scope+'_AD', expected, inside)
            branch['folds'].append(dict(fold, ad_threshold=actual_threshold))
        result = predict_with_artifact(branch, external_X)
        per_fold_y = result[[f'pred_y_{fold["fold_name"]}' for fold in artifact['folds']]].copy()
        per_fold_ec3 = result[[f'Predict_LLNA EC3_{fold["fold_name"]}' for fold in artifact['folds']]].copy()
        per_fold_y.columns = per_fold_ec3.columns = reference._exval_pred_result['AD'].columns
        source_ad = reference._exval_pred_result['AD']
        assert_numeric(checks, prefix+'.any_inside', source_ad.any(axis=1), result['any_kNN_AD'])
        assert_numeric(checks, prefix+'.eligible_max_y', per_fold_y[source_ad].max(axis=1), result['knn_inAD_pred_y'])
        assert_numeric(checks, prefix+'.eligible_min_EC3', per_fold_ec3[source_ad].min(axis=1), result['knn_inAD_Predict_LLNA EC3'])


if __name__ == '__main__':
    main()
