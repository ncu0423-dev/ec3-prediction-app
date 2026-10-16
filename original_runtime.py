"""Inference with saved fold pipelines. Loading artifacts never starts training.

Only load artifacts created in a trusted environment: joblib/pickle can execute code.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils_io import MW_COL, TARGET_COLUMN_DEFAULT, ensure_dir, read_csv_auto, validate_numeric_columns
from utils_ml import (compute_knn_ad, convert_ec3_to_y, convert_y_to_ec3,
                      extract_invitro_calls, library_versions, shap_explanation,
                      valid_regression_metrics)

ARTIFACT_FORMAT_VERSION = 2


def load_artifact(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'学習済みモデルがありません: {path}。READMEのtrain.pyコマンドで先に学習してください。')
    artifact = joblib.load(path)
    required = {'format_version','full_feature_columns','selected_feature_mask',
                'selected_feature_columns','folds','metadata'}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError('対応するNotebook artifactではありません。既存のmodel/モデルは旧方式です。train.pyで学習してください。')
    if artifact['format_version'] != ARTIFACT_FORMAT_VERSION:
        raise ValueError('artifactの形式バージョンが対応していません。学習コードを確認してください。')
    mask = np.asarray(artifact['selected_feature_mask'], dtype=bool)
    full = list(artifact['full_feature_columns'])
    if len(mask) != len(full) or list(np.asarray(full)[mask]) != list(artifact['selected_feature_columns']):
        raise ValueError('artifactの特徴量順または選択マスクが破損しています。')
    if not artifact['folds']:
        raise ValueError('artifactに学習済みFoldがありません。')
    for fold in artifact['folds']:
        if not {'fold_name', 'pipeline', 'model', 'X_train_selected'}.issubset(fold):
            raise ValueError('artifactのFold情報が不足しています。')
    current = library_versions()
    mismatches = [name for name,version in artifact.get('versions',{}).items()
                  if name in current and version != current[name]]
    if mismatches:
        warnings.warn('学習時とライブラリのバージョンが異なります: ' + ', '.join(mismatches), RuntimeWarning)
    return artifact


def prepare_input_features(artifact, frame):
    """All original descriptors are required because KNN uses full-space distances."""
    return validate_numeric_columns(frame, artifact['full_feature_columns'])


def _select_min_ec3_fold(pred_y_fold, pred_ec3_fold, *, eligibility=None):
    if list(pred_y_fold.columns) != list(pred_ec3_fold.columns) or pred_ec3_fold.shape[1] == 0:
        raise ValueError('Fold予測の列が一致していません。')
    y = pred_y_fold.to_numpy(dtype=float)
    ec3 = pred_ec3_fold.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(ec3) & (ec3 > 0)
    if eligibility is not None:
        valid &= eligibility.to_numpy(dtype=bool)
    indices = np.argmin(np.where(valid, ec3, np.inf), axis=1)
    has_candidate = valid.any(axis=1)
    rows = np.arange(len(y))
    return pd.DataFrame({
        'selected_fold': np.where(has_candidate, np.asarray(pred_y_fold.columns)[indices], None),
        'selected_fold_index': np.where(has_candidate, indices, -1),
        'pred_y': np.where(has_candidate, y[rows, indices], np.nan),
        'pred_ec3': np.where(has_candidate, ec3[rows, indices], np.nan),
        'has_candidate': has_candidate,
    })


def _fold_inputs(artifact, frame, fold):
    full = prepare_input_features(artifact, frame)
    transformed = fold['pipeline'].transform(full)
    if transformed.shape[1] != len(artifact['full_feature_columns']):
        raise ValueError('前処理後の特徴量の列数がartifactと一致していません。')
    return transformed[:, np.asarray(fold.get('selected_feature_mask', artifact['selected_feature_mask']), dtype=bool)]


def predict_with_artifact(artifact, input_df, *, compute_ad=True):
    full = prepare_input_features(artifact, input_df)
    mw = full[MW_COL].to_numpy(dtype=float)
    predictions, ec3_predictions, ad_predictions, distances = {}, {}, {}, {}
    for fold in artifact['folds']:
        name = fold['fold_name']
        transformed = fold['pipeline'].transform(full)
        if transformed.shape[1] != full.shape[1]:
            raise ValueError('前処理の列数がartifactと一致していません。')
        selected = transformed[:, np.asarray(fold.get('selected_feature_mask', artifact['selected_feature_mask']), dtype=bool)]
        predicted_y = np.asarray(fold['model'].predict(selected))
        predicted_ec3 = convert_y_to_ec3(predicted_y, mw)
        if not (np.isfinite(predicted_y).all() and np.isfinite(predicted_ec3).all() and (predicted_ec3>0).all()):
            raise ValueError('予測値が有限の範囲を超えました。入力値と学習済みモデルを確認してください。')
        predictions[name], ec3_predictions[name] = predicted_y, predicted_ec3
        if compute_ad:
            inside, distance, _ = compute_knn_ad(
                fold['X_train_selected'], selected,
                knn_k=artifact['metadata']['knn_k'],
                ad_coverage=artifact['metadata']['ad_coverage'], threshold=fold['ad_threshold'])
            ad_predictions[name], distances[name] = inside, distance
    pred_y_fold, pred_ec3_fold = pd.DataFrame(predictions), pd.DataFrame(ec3_predictions)
    selection = _select_min_ec3_fold(pred_y_fold, pred_ec3_fold)
    # Keep identifiers and unrecognized descriptive columns; raw descriptors are not echoed.
    metadata_columns = [c for c in input_df if c not in artifact['full_feature_columns']
                        and c not in (TARGET_COLUMN_DEFAULT, artifact.get('target_col', TARGET_COLUMN_DEFAULT))]
    out = input_df[metadata_columns].reset_index(drop=True).copy()
    out['selected_fold'] = selection['selected_fold']
    out['pred_y'] = selection['pred_y']
    out['Predict_LLNA EC3'] = selection['pred_ec3']
    if compute_ad:
        ad_fold = pd.DataFrame(ad_predictions)
        eligible = _select_min_ec3_fold(pred_y_fold, pred_ec3_fold, eligibility=ad_fold)
        out['kNN_AD'] = ad_fold.to_numpy()[np.arange(len(out)), selection['selected_fold_index'].to_numpy()]
        out['any_kNN_AD'] = ad_fold.any(axis=1)
        out['selected_inAD_fold'] = eligible['selected_fold']
        out['knn_inAD_pred_y'] = eligible['pred_y']
        out['knn_inAD_Predict_LLNA EC3'] = eligible['pred_ec3']
        for name in ad_fold:
            out['knn_AD_' + name] = ad_fold[name]
            out['knn_distance_' + name] = distances[name]
    else:
        # An uncomputed AD must never be presented as inside.
        for column in ('kNN_AD', 'any_kNN_AD'):
            out[column] = pd.array([pd.NA] * len(out), dtype='boolean')
        out['selected_inAD_fold'] = None
        out['knn_inAD_pred_y'] = np.nan
        out['knn_inAD_Predict_LLNA EC3'] = np.nan
    for name in pred_y_fold:
        out['pred_y_' + name] = pred_y_fold[name]
        out['Predict_LLNA EC3_' + name] = pred_ec3_fold[name]
    out['pred_y_mean'] = pred_y_fold.mean(axis=1)
    out['Predict_LLNA EC3_mean'] = pred_ec3_fold.mean(axis=1)
    out['Predict_LLNA EC3_min'] = pred_ec3_fold.min(axis=1)
    invitro = extract_invitro_calls(input_df)
    out['invitro_AD'], out['vitro_call_sum'] = invitro['invitro_AD'], invitro['vitro_call_sum']
    target_col = artifact.get('target_col', TARGET_COLUMN_DEFAULT)
    if target_col in input_df:
        truth = validate_numeric_columns(input_df, [target_col], require_mw=False)[target_col].to_numpy()
        if np.any(np.isfinite(truth) & (truth <= 0)):
            raise ValueError('評価用LLNA_EC3は0より大きい値が必要です。')
        out['true_y'] = convert_ec3_to_y(truth, mw)
        out['LLNA EC3(NS converted to 150%)'] = convert_y_to_ec3(out['true_y'], mw)
        out['Fold_misprediction'] = out['Predict_LLNA EC3'] / out['LLNA EC3(NS converted to 150%)']
        out.attrs['metrics'] = {'y': valid_regression_metrics(out['true_y'],out['pred_y']),
                                'ec3': valid_regression_metrics(truth,out['Predict_LLNA EC3'])}
    out.attrs['aggregation'] = 'minimum_predicted_ec3_per_row'
    out.attrs['training_mode'] = artifact['metadata']['training_mode']
    return out


def explain_prediction(artifact, frame, *, fold_name='fold_1'):
    fold = next((f for f in artifact['folds'] if f['fold_name'] == fold_name), None)
    if fold is None:
        raise ValueError(f'指定したFoldはありません: {fold_name}')
    selected = _fold_inputs(artifact, frame, fold)
    return shap_explanation(fold['model'], selected, fold['selected_feature_columns'])


def make_prediction_template(artifact):
    return pd.DataFrame(columns=['No.', 'Name', 'CAS Number', 'SMILES', *artifact['full_feature_columns']])


# Clear aliases for applications embedding the runtime.
load_bundle = load_artifact
predict_frame = predict_with_artifact
predict_dataframe = predict_with_artifact


def main():
    parser = argparse.ArgumentParser(description='保存済みNotebook Foldモデルによる予測')
    parser.add_argument('--artifact-path', default='artifacts/original_model_artifact.joblib')
    parser.add_argument('--input-csv')
    parser.add_argument('--output-csv', default='outputs/prediction_results.csv')
    parser.add_argument('--template-csv')
    parser.add_argument('--no-ad', action='store_true')
    args = parser.parse_args()
    artifact = load_artifact(args.artifact_path)
    if args.template_csv:
        path = Path(args.template_csv)
        ensure_dir(path.parent)
        make_prediction_template(artifact).to_csv(path, index=False, encoding='utf-8-sig')
    if args.input_csv:
        frame, _ = read_csv_auto(args.input_csv)
        result = predict_with_artifact(artifact, frame, compute_ad=not args.no_ad)
        path = Path(args.output_csv)
        ensure_dir(path.parent)
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print('予測結果を保存しました:', path)
    if not args.input_csv and not args.template_csv:
        parser.error('--input-csv または --template-csv を指定してください。')


if __name__ == '__main__':
    main()
