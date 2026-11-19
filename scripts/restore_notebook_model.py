"""Restore verified Notebook weights and rebuild the final (.2) AD reports.

No fit/tuning is executed. The original verified artifact remains unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from original_runtime import load_artifact, make_prediction_template, predict_with_artifact
from utils_io import ensure_dir, load_dataset, save_json
from utils_ml import compute_knn_ad, convert_ec3_to_y, convert_y_to_ec3, fit_knn_ad, library_versions, valid_regression_metrics


def restore_verified_notebook(source, verification_report, train_csv, external_csv,
                              *, output, output_dir, ad_coverage=.2):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Keep the verified canonical model unchanged; use a separate output.')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = json.loads(Path(verification_report).read_text())
    artifact = load_artifact(source)
    checks = evidence.get('checks', {})
    if not (evidence.get('status') == 'passed' and evidence.get('production_1000_trials_verified')
            and len(checks) >= 112 and all(c.get('status') == 'passed' for c in checks.values())
            and evidence.get('artifact', {}).get('sha256') == digest):
        raise ValueError('Full-cost independent Notebook source verification does not match the supplied model.')
    if not artifact['versions'] == evidence['versions'] == library_versions():
        raise ValueError('Library versions differ from the verified run; independently revalidate before reuse.')
    if artifact['metadata'].get('feature_mode', 'boruta') != 'boruta':
        raise ValueError('A fixed-feature model cannot restore Notebook preprocessing.')
    if not np.array_equal(artifact['selected_feature_mask'], artifact['global_boruta'].support_):
        raise ValueError('Boruta mask differs from the verified artifact.')
    training = load_dataset(train_csv)
    external = load_dataset(external_csv, feature_columns=artifact['full_feature_columns'], external=True)
    if (training['sha256'] != artifact['metadata']['train_source']['sha256']
            or external['sha256'] != artifact['metadata']['external_source']['sha256']
            or training['X'].columns.tolist() != artifact['full_feature_columns']):
        raise ValueError('Source data/feature order differ from the verified model.')
    if not 0 <= ad_coverage <= 1:
        raise ValueError('AD coverage must be between0 and1 (excluded fraction).')
    artifact['metadata'].update(
        feature_mode='boruta', ad_coverage=ad_coverage,
        mw_role='full_preprocessing_and_inverse_conversion',
        required_input_columns=artifact['full_feature_columns'],
        notebook_reproduction_verified=True,
        notebook_reproduction_scope='verified_prediction_workflow_current_pinned_environment',
        notebook_all_specs_verified=False,
        source_parity_scope=evidence['scope'], source_parity_check_count=len(checks),
        source_artifact_sha256=digest,
        source_full_verification_sha256=hashlib.sha256(Path(verification_report).read_bytes()).hexdigest(),
        historical_notebook_output_parity='unavailable',
        weights_reused_without_retraining=True,
        notebook_output_branch='final_CSV_cells_265_270' if ad_coverage == .2 else 'explicit_AD_alternative',
        invitro_ad_available=False,
        missing_invitro_sources=['Annex-2-in-vitro-in-silico-in-vivo-defined-approaches.csv', 'newCE_data.csv'],
    )
    oof = artifact['reports']['oof'].copy()
    for fold in artifact['folds']:
        if len(fold['optimization']['trials']) != 1000:
            raise ValueError('Source model lacks a complete1000-trial study perFold.')
        fold['ad_threshold'] = fit_knn_ad(fold['X_train_selected'],
            knn_k=artifact['metadata']['knn_k'], ad_coverage=ad_coverage)
        idx = fold['test_indices']
        selected = fold['pipeline'].transform(training['X'].iloc[idx])[:, fold['selected_feature_mask']]
        pred_y = fold['model'].predict(selected)
        pred_ec3 = convert_y_to_ec3(pred_y, training['mw'][idx])
        np.testing.assert_array_equal(pred_y, oof.iloc[idx]['pred_y'])
        np.testing.assert_array_equal(pred_ec3, oof.iloc[idx]['Predict_LLNA EC3'])
        inside, distance, _ = compute_knn_ad(fold['X_train_selected'], selected,
            knn_k=artifact['metadata']['knn_k'], threshold=fold['ad_threshold'])
        oof.loc[idx, 'kNN_AD'], oof.loc[idx, 'knn_distance'] = inside, distance
    artifact['reports']['oof'] = oof
    inside = oof['kNN_AD'].to_numpy(dtype=bool)
    metrics = artifact['metrics']
    metrics['oof_inAD_y'] = valid_regression_metrics(oof.loc[inside, 'true_y'], oof.loc[inside, 'pred_y'])
    metrics['oof_inAD_ec3'] = valid_regression_metrics(oof.loc[inside, 'LLNA EC3(NS converted to 150%)'], oof.loc[inside, 'Predict_LLNA EC3'])
    prediction = predict_with_artifact(artifact, external['frame'])
    prediction['source_row'] = external['source_rows']
    prediction.attrs = {}
    artifact['reports']['external'] = prediction
    true_y = convert_ec3_to_y(external['ec3'], external['mw'])
    true_ec3 = convert_y_to_ec3(true_y, external['mw'])
    for key, truth, values in [
        ('external_y', true_y, prediction['pred_y']),
        ('external_ec3', true_ec3, prediction['Predict_LLNA EC3']),
        ('external_inAD_y', true_y, prediction['knn_inAD_pred_y']),
        ('external_inAD_ec3', true_ec3, prediction['knn_inAD_Predict_LLNA EC3']),
        ('external_mean_y', true_y, prediction['pred_y_mean']),
        ('external_mean_ec3', true_ec3, prediction['Predict_LLNA EC3_mean']),
    ]:
        metrics[key] = valid_regression_metrics(truth, values)
    ensure_dir(output.parent)
    temporary = output.with_name(output.name + '.tmp')
    try:
        joblib.dump(artifact, temporary, compress=3)
        load_artifact(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    directory = ensure_dir(output_dir)
    save_json(artifact['metadata'] | {'versions': artifact['versions'],
        'selected_feature_columns': artifact['selected_feature_columns']}, directory / 'restoration_conditions.json')
    save_json(metrics, directory / 'metrics.json')
    for name, report in artifact['reports'].items():
        report.to_csv(directory / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
    make_prediction_template(artifact).to_csv(directory / 'input_template.csv', index=False, encoding='utf-8-sig')
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Canonical source unexpectedly changed.')
    print(f'Restored verified weights without fitting: {output}; full inputs={len(artifact["full_feature_columns"])}, selected={len(artifact["selected_feature_columns"])}, ADexcluded={ad_coverage}')
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/notebook_model_artifact.joblib')
    parser.add_argument('--verification-report', type=Path, default=ROOT / 'outputs/notebook_comparison_full.json')
    parser.add_argument('--train-csv', type=Path, default=ROOT / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
    parser.add_argument('--external-csv', type=Path, default=ROOT / 'data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/notebook_restored_model_artifact.joblib')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/notebook_restored')
    parser.add_argument('--ad-coverage', type=float, default=.2)
    args = parser.parse_args()
    restore_verified_notebook(args.source, args.verification_report, args.train_csv, args.external_csv,
        output=args.output, output_dir=args.output_dir, ad_coverage=args.ad_coverage)


if __name__ == '__main__':
    main()
