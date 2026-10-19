"""Train a fold ensemble, with notebook Boruta or the user-specified 23 features.

Notebook mode: original 5 x 1000-trial nested optimization. Smoke mode: 5 x 2
trials, 200 rounds maximum per inner fit. Smoke artifacts are visibly distinct.
--feature-mode fixed23 uses only those 23 columns for preprocessing and fitting;
it does not claim original-notebook feature-selection parity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from original_runtime import ARTIFACT_FORMAT_VERSION, make_prediction_template, predict_with_artifact
from utils_io import FIXED_FEATURE_COLUMNS, META_COLUMNS, TARGET_COLUMN_DEFAULT, ensure_dir, load_dataset, save_json
from utils_ml import (NOTEBOOK_XGB_BASE_PARAMS, _fit_model, clone_and_transform,
                      compute_knn_ad, convert_ec3_to_y, convert_y_to_ec3,
                      extract_invitro_calls, fit_global_boruta, fit_knn_ad,
                      library_versions, make_llna_labels, optimize_xgb_params,
                      save_shap_artifacts_for_fold, valid_regression_metrics)


def _fit_fold(X, y, mw, reconverted, labels, mask, selected_features, number, train_indices,
              test_indices, n_trials, max_estimators, knn_k, ad_coverage, mode):
    name = f'fold_{number}'
    print(f'[{mode}] {name}: fit preprocessing and tune', flush=True)
    train_X, test_X, pipeline = clone_and_transform(X.iloc[train_indices], X.iloc[test_indices])
    train_selected, test_selected = train_X[:, mask], test_X[:, mask]
    params, optimization = optimize_xgb_params(
        train_selected, y[train_indices], labels[train_indices],
        n_trials=n_trials, max_estimators=max_estimators, n_jobs=1)
    model = _fit_model(params, train_selected, y[train_indices])
    train_prediction = model.predict(train_selected)
    test_prediction = model.predict(test_selected)
    train_ec3_prediction = convert_y_to_ec3(train_prediction, mw[train_indices])
    test_ec3_prediction = convert_y_to_ec3(test_prediction, mw[test_indices])
    inside, distance, threshold = compute_knn_ad(
        train_selected, test_selected, knn_k=knn_k, ad_coverage=ad_coverage)
    metrics = {'training_y': valid_regression_metrics(y[train_indices],train_prediction),
               'training_ec3': valid_regression_metrics(reconverted[train_indices],train_ec3_prediction),
               'holdout_y': valid_regression_metrics(y[test_indices],test_prediction),
               'holdout_ec3': valid_regression_metrics(reconverted[test_indices],test_ec3_prediction)}
    fold = {'fold_name': name, 'pipeline': pipeline, 'model': model,
            'selected_feature_mask': mask.copy(), 'selected_feature_columns': selected_features,
            'X_train_selected': train_selected, 'train_indices': train_indices,
            'test_indices': test_indices, 'best_params': params, 'optimization': optimization,
            'ad_threshold': threshold, 'metrics': metrics,
            'feature_importance': dict(zip(selected_features,model.feature_importances_.tolist()))}
    # The notebook's summary SHAP is on each outer fold's training matrix.
    save_shap_artifacts_for_fold(fold, train_selected, selected_features)
    return fold, test_prediction, test_ec3_prediction, inside, distance


def train_model(train_csv, exval_csv=None, *, artifact_path=None, output_dir=None,
                mode='notebook', n_trials=None, max_estimators=None,
                boruta_n_jobs=2, ad_coverage=.1, knn_k=5, feature_mode='boruta', fold_workers=1):
    if mode not in ('notebook','smoke'):
        raise ValueError('modeはnotebookまたはsmokeです。')
    if not 1 <= fold_workers <= 4:
        raise ValueError('fold_workersは1から4です。')
    if feature_mode not in ('boruta', 'fixed23'):
        raise ValueError('feature_modeはborutaまたはfixed23です。')
    n_trials = (1000 if mode == 'notebook' else 2) if n_trials is None else n_trials
    max_estimators = (100000 if mode == 'notebook' else 200) if max_estimators is None else max_estimators
    if n_trials < 1 or max_estimators < 1:
        raise ValueError('n_trialsおよびmax_estimatorsは1以上です。Optunaを無効化できません。')
    if mode == 'notebook' and (n_trials != 1000 or max_estimators != 100000):
        raise ValueError('Notebook再現モードは1000 trials / 100000 roundsです。軽量検証には--mode smokeを指定してください。')
    if not 1 <= boruta_n_jobs <= 2:
        raise ValueError('Boruta並列数は1または2に制限しています。')
    output_dir = ensure_dir(output_dir or (f'outputs/fixed23_{mode}' if feature_mode == 'fixed23' else f'outputs/{mode}'))
    default_artifact = (f'artifacts/fixed23_{"model" if mode == "notebook" else "smoke_model"}_artifact.joblib'
                        if feature_mode == 'fixed23' else
                        ('artifacts/original_model_artifact.joblib' if mode=='notebook' else 'artifacts/smoke_model_artifact.joblib'))
    artifact_path = Path(artifact_path or default_artifact)
    ensure_dir(artifact_path.parent)
    dataset = load_dataset(train_csv, feature_columns=FIXED_FEATURE_COLUMNS if feature_mode == 'fixed23' else None)
    X, ec3, mw = dataset['X'], dataset['ec3'], dataset['mw']
    y = convert_ec3_to_y(ec3, mw)
    reconverted = convert_y_to_ec3(y, mw)
    # Exact ==150 after roundtrip is intentional: matching raw labels is different.
    labels = make_llna_labels(reconverted)
    if min(np.unique(labels, return_counts=True)[1]) < 7:
        raise ValueError('5-fold外側/内側の層化分割には各クラスに十分なデータが必要です。')
    if feature_mode == 'fixed23':
        print(f'[{mode}/fixed23] User-specified {X.shape[1]} features; no Boruta.', flush=True)
        selector, global_pipeline, global_X = None, None, None
        mask = np.ones(len(FIXED_FEATURE_COLUMNS), dtype=bool)
    else:
        print(f'[{mode}] Global Boruta: {len(X)} rows, {X.shape[1]} features', flush=True)
        selector, global_pipeline, global_X = fit_global_boruta(X, y, n_jobs=boruta_n_jobs)
        mask = selector.support_
    selected_features = X.columns[mask].tolist()
    print(f'[{mode}] Selected {int(mask.sum())} features. Optuna {n_trials} trials x 5 folds.', flush=True)
    folds, fold_metrics = [], {}
    oof_y, oof_ec3 = np.empty(len(X)), np.empty(len(X))
    oof_ad, oof_distance = np.zeros(len(X), dtype=bool), np.empty(len(X))
    oof_fold = np.empty(len(X), dtype=object)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1037)
    splits = list(cv.split(X, labels))
    arguments = [(X, y, mw, reconverted, labels, mask, selected_features, number, train_indices,
                  test_indices, n_trials, max_estimators, knn_k, ad_coverage, mode)
                 for number, (train_indices, test_indices) in enumerate(splits, start=1)]
    if fold_workers == 1:
        results = [_fit_fold(*args) for args in arguments]
    else:
        # Each study/model has an explicit seed; processes share no RNG state.
        with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=fold_workers)(
                joblib.delayed(_fit_fold)(*args) for args in arguments)
    for fold, test_prediction, test_ec3_prediction, inside, distance in results:
        test_indices = fold['test_indices']
        oof_y[test_indices], oof_ec3[test_indices] = test_prediction, test_ec3_prediction
        oof_fold[test_indices] = fold['fold_name']
        oof_ad[test_indices], oof_distance[test_indices] = inside, distance
        fold_metrics[fold['fold_name']] = fold['metrics']
        folds.append(fold)
    metadata = {
        'training_mode': mode, 'created_at': datetime.now(timezone.utc).isoformat(),
        'feature_mode': feature_mode, 'fold_workers': fold_workers,
        'random_state': 1037, 'cv_folds': 5, 'inner_folds': 5, 'inner_random_state': 0,
        'n_trials': n_trials, 'max_estimators': max_estimators,
        'optuna_enabled': True, 'optuna_sampler': 'TPESampler(seed=0)',
        'optuna_objective': 'mean inner StratifiedKFold R2 on log10(MW/EC3)',
        'boruta': {'enabled': True, 'scope': 'all_training_before_outer_cv', 'rf_max_depth': 5,
                   'perc': 100, 'random_state': 0, 'n_estimators': 'auto',
                   'max_iter': 100, 'n_jobs': boruta_n_jobs, 'confirmed_only': True}
                   if feature_mode == 'boruta' else {'enabled': False, 'scope': 'user_specified_fixed23'},
        'target_transform': 'log10(MolWeight / LLNA_EC3)',
        'target_inverse': 'MolWeight / (10 ** y), preserving model prediction dtype',
        'fold_aggregation': 'minimum_predicted_ec3_per_row',
        'final_n_estimators': 'max winning-trial inner best_iteration, WITHOUT +1 (notebook behavior)',
        'stratification': 'exact ==150 and <=2 AFTER target roundtrip (notebook behavior)',
        'knn_k': knn_k, 'ad_coverage': ad_coverage,
        'ad_coverage_meaning': 'fraction excluded from training distance distribution',
        'ad_threshold_method': 'sorted_mean_distances[round(n_train*(1-coverage))-1]',
        'notebook_reproduction_verified': False,
        'smoke_warning': '軽量動作検証モデルです。本番Notebookの1000 trialsでの性能・一致は未検証。' if mode=='smoke' else None,
        'limitations': [
            'Global supervised Boruta sees outer holdouts: OOF scores contain feature-selection leakage.',
            'Each outer preprocessing pipeline is fitted before inner tuning CV: inner selection scores contain preprocessing leakage.',
            'Original library versions are not pinned in notebook; historical predictions may differ.',
            'Combined in-vitro AD is unavailable without original hCLAT.Call source; hCLAT.score is not substituted.',
        ],
        'train_source': {'sha256': dataset['sha256'], 'rows': len(X), 'encoding': dataset['encoding']},
        'input_missing_policy': 'All saved feature columns required; missing numeric cells KNN-imputed, MW positive finite.',
    }
    if feature_mode == 'fixed23':
        metadata['limitations'].remove('Global supervised Boruta sees outer holdouts: OOF scores contain feature-selection leakage.')
        metadata['limitations'].append('User-specified 23-feature workflow differs from the original 452-feature Boruta workflow; historical parity does not apply.')
    identity = dataset['frame'][[c for c in META_COLUMNS if c in dataset['frame']]].reset_index(drop=True)
    oof = identity.copy()
    oof['source_row'] = dataset['source_rows']
    oof['fold'] = oof_fold
    oof['true_y'], oof['LLNA EC3(NS converted to 150%)'] = y, reconverted
    oof['pred_y'], oof['Predict_LLNA EC3'] = oof_y, oof_ec3
    oof['kNN_AD'], oof['knn_distance'] = oof_ad, oof_distance
    oof['Fold_misprediction'] = oof_ec3 / reconverted
    invitro = extract_invitro_calls(dataset['frame'])
    oof['invitro_AD'], oof['vitro_call_sum'] = invitro['invitro_AD'], invitro['vitro_call_sum']
    metrics = {'oof_y': valid_regression_metrics(y,oof_y),
               'oof_ec3': valid_regression_metrics(reconverted,oof_ec3),
               'oof_inAD_y': valid_regression_metrics(y[oof_ad],oof_y[oof_ad]),
               'oof_inAD_ec3': valid_regression_metrics(reconverted[oof_ad],oof_ec3[oof_ad]),
               'train_folds': fold_metrics}
    artifact = {'format_version': ARTIFACT_FORMAT_VERSION,
                'full_feature_columns': X.columns.tolist(),
                'selected_feature_mask': mask, 'selected_feature_columns': selected_features,
                'target_col': TARGET_COLUMN_DEFAULT, 'folds': folds, 'metadata': metadata,
                'metrics': metrics, 'versions': library_versions(), 'reports': {'oof': oof},
                'global_boruta': selector, 'global_selection_pipeline': global_pipeline,
                'global_selection_transformed': global_X, 'train_target_y': y,
                'train_target_reconverted': reconverted, 'train_labels': labels}
    if exval_csv is not None:
        external = load_dataset(exval_csv, feature_columns=X.columns.tolist(), external=True)
        prediction = predict_with_artifact(artifact, external['frame'])
        prediction['source_row'] = external['source_rows']
        prediction.attrs = {}
        artifact['reports']['external'] = prediction
        truth_y = convert_ec3_to_y(external['ec3'], external['mw'])
        truth_ec3 = convert_y_to_ec3(truth_y, external['mw'])
        for key, true_values, predicted in [
            ('external_y',truth_y,prediction['pred_y']),
            ('external_ec3',truth_ec3,prediction['Predict_LLNA EC3']),
            ('external_inAD_y',truth_y,prediction['knn_inAD_pred_y']),
            ('external_inAD_ec3',truth_ec3,prediction['knn_inAD_Predict_LLNA EC3']),
            ('external_mean_y',truth_y,prediction['pred_y_mean']),
            ('external_mean_ec3',truth_ec3,prediction['Predict_LLNA EC3_mean']),
        ]:
            metrics[key] = valid_regression_metrics(true_values,predicted)
        metadata['external_source'] = {'sha256': external['sha256'], 'rows': len(prediction),
            'encoding': external['encoding'], 'dropped_missing_mw_rows': external['dropped_missing_mw_rows'],
            'role': 'external_validation_only_never_used_for_fit'}
    joblib.dump(artifact, artifact_path, compress=3)
    for name, report in artifact['reports'].items():
        report.to_csv(output_dir/f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
    save_json(metrics, output_dir/'metrics.json')
    save_json(metadata | {'versions': artifact['versions'], 'selected_feature_columns': selected_features}, output_dir/'training_conditions.json')
    make_prediction_template(artifact).to_csv(output_dir/'input_template.csv', index=False, encoding='utf-8-sig')
    save_json({'feature_columns': X.columns.tolist(), 'selected_mask': mask,
               'selected_feature_columns': selected_features,
               'folds': [{'fold': f['fold_name'], 'train_indices': f['train_indices'],
                          'test_indices': f['test_indices'], 'params': f['best_params'],
                          'ad_threshold': f['ad_threshold'], 'optimization': f['optimization']} for f in folds]},
              output_dir/'fold_training_details.json')
    print(f'[{mode}] Saved: {artifact_path}', flush=True)
    return artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-csv', default='data/DASS_join_data_opera_delete_LLNAnan.csv')
    p.add_argument('--exval-csv', default='data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv')
    p.add_argument('--artifact-path')
    p.add_argument('--output-dir')
    p.add_argument('--mode', choices=('notebook','smoke'), default='notebook')
    p.add_argument('--feature-mode', choices=('boruta','fixed23'), default='boruta')
    p.add_argument('--n-trials', type=int)
    p.add_argument('--max-estimators', type=int)
    p.add_argument('--ad-coverage', type=float, default=.1)
    p.add_argument('--knn-k', type=int, default=5)
    p.add_argument('--fold-workers', type=int, default=1, help='Independent outer folds, 1-4 processes')
    p.add_argument('--boruta-n-jobs', type=int, default=2)
    args = p.parse_args()
    train_model(args.train_csv, args.exval_csv, artifact_path=args.artifact_path,
                output_dir=args.output_dir, mode=args.mode, n_trials=args.n_trials,
                max_estimators=args.max_estimators, ad_coverage=args.ad_coverage,
                knn_k=args.knn_k, boruta_n_jobs=args.boruta_n_jobs, feature_mode=args.feature_mode, fold_workers=args.fold_workers)


if __name__ == '__main__':
    main()
