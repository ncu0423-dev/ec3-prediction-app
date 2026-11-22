"""Reuse the validated fixed21 weights for the requested CSV24 + screen-MW workflow."""
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
from csv_input_schema import requested_csv_columns
from original_runtime import load_artifact, predict_with_artifact
from utils_io import FIXED21_FEATURE_COLUMNS, ensure_dir, load_dataset, save_json
from utils_ml import compute_knn_ad, fit_knn_ad, library_versions, valid_regression_metrics


def prepare_model(source, verification, output, output_dir, *, ad_coverage=.2):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('元の検証済みモデルと出力先は別にしてください。')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    proof = json.loads(Path(verification).read_text())
    model = load_artifact(source)
    if proof['model_sha256'] != digest or proof['n_trials_per_fold'] != [1000] * 5:
        raise ValueError('モデルが本番検証記録と一致しません。')
    if model['versions'] != proof['versions'] or model['versions'] != library_versions():
        raise ValueError('学習時とライブラリのバージョンが異なります。')
    if (model['metadata'].get('feature_mode') != 'fixed21'
            or model['full_feature_columns'] != FIXED21_FEATURE_COLUMNS
            or model['selected_feature_columns'] != FIXED21_FEATURE_COLUMNS):
        raise ValueError('指定21項目のモデルではありません。')
    training = load_dataset(ROOT / 'data/DASS_join_data_opera_delete_LLNAnan.csv', feature_columns=FIXED21_FEATURE_COLUMNS)
    external = load_dataset(ROOT / 'data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv',
        feature_columns=FIXED21_FEATURE_COLUMNS, external=True)
    if (training['sha256'] != model['metadata']['train_source']['sha256']
            or external['sha256'] != model['metadata']['external_source']['sha256']):
        raise ValueError('元の学習・外部データが検証済みモデルと異なります。')
    model['metadata'].update(ad_coverage=ad_coverage,
        source_artifact_sha256=digest, weights_reused_without_retraining=True,
        csv_input_workflow='identifiers3_descriptors21_screen_molecular_weight',
        upload_csv_columns=requested_csv_columns(),
        molecular_weight_entry='per_row_screen_input',
        notebook_reproduction_verified=False, notebook_all_specs_verified=False,
        invitro_ad_available=False)
    report = model['reports']['oof'].copy()
    for fold in model['folds']:
        if (len(fold['optimization']['trials']) != 1000 or fold['pipeline'].n_features_in_ != 21
                or fold['model'].n_features_in_ != 21):
            raise ValueError('Foldの学習条件が一致しません。')
        fold['ad_threshold'] = fit_knn_ad(fold['X_train_selected'], knn_k=5, ad_coverage=ad_coverage)
        idx = fold['test_indices']
        selected = fold['pipeline'].transform(training['X'].iloc[idx])[:, fold['selected_feature_mask']]
        np.testing.assert_array_equal(fold['model'].predict(selected), report.iloc[idx]['pred_y'])
        inside, distance, _ = compute_knn_ad(fold['X_train_selected'], selected, knn_k=5, threshold=fold['ad_threshold'])
        report.loc[idx, 'kNN_AD'], report.loc[idx, 'knn_distance'] = inside, distance
    model['reports']['oof'] = report
    inside = report['kNN_AD'].to_numpy(dtype=bool)
    for space, truth, prediction in [('y', 'true_y', 'pred_y'), ('ec3', 'LLNA EC3(NS converted to 150%)', 'Predict_LLNA EC3')]:
        model['metrics']['oof_inAD_' + space] = valid_regression_metrics(report.loc[inside, truth], report.loc[inside, prediction])
    prediction = predict_with_artifact(model, external['frame'])
    prediction['source_row'] = external['source_rows']
    prediction.attrs = {}
    model['reports']['external'] = prediction
    for prefix, y_column, ec3_column in [('', 'pred_y', 'Predict_LLNA EC3'),
        ('inAD_', 'knn_inAD_pred_y', 'knn_inAD_Predict_LLNA EC3'),
        ('mean_', 'pred_y_mean', 'Predict_LLNA EC3_mean')]:
        model['metrics']['external_' + prefix + 'y'] = valid_regression_metrics(prediction['true_y'], prediction[y_column])
        model['metrics']['external_' + prefix + 'ec3'] = valid_regression_metrics(prediction['LLNA EC3(NS converted to 150%)'], prediction[ec3_column])
    ensure_dir(output.parent)
    temporary = output.with_name(output.name + '.tmp')
    try:
        joblib.dump(model, temporary, compress=3)
        load_artifact(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    directory = ensure_dir(output_dir)
    save_json(model['metadata'] | {'versions': model['versions']}, directory / 'conditions.json')
    save_json(model['metrics'], directory / 'metrics.json')
    for name, frame in model['reports'].items():
        frame.to_csv(directory / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError('検証元モデルが変更されました。')
    print(f'Prepared21-feature model without refitting: {output}; CSV24 columns; AD excluded fraction={ad_coverage}')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/fixed21_model_artifact.joblib')
    parser.add_argument('--verification', type=Path, default=ROOT / 'docs/fixed21_validation.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/csv21_model_artifact.joblib')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/csv21')
    args = parser.parse_args()
    prepare_model(args.source, args.verification, args.output, args.output_dir)


if __name__ == '__main__':
    main()
