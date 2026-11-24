"""Package both saved production workflows without fitting or changing weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_profiles import model_paths, validate_profile_artifact
from original_runtime import load_artifact, predict_with_artifact
from scripts.export_deployment_model import build_deployment_artifact
from utils_io import load_dataset, save_json
from utils_ml import library_versions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_models(*, source: Path, output: Path, proof_path: Path,
                   fixed21: Path, template: Path, report: Path) -> dict:
    """Use the audited Notebook weights; retain the existing 21-column package."""
    source, output, fixed21 = Path(source), Path(output), Path(fixed21)
    if output.resolve() in (source.resolve(), fixed21.resolve()):
        raise ValueError('452項目配備先は元モデル・21項目モデルとは別にしてください。')
    original_hash, source_hash = sha256(fixed21), sha256(source)
    proof = json.loads(Path(proof_path).read_text(encoding='utf-8'))
    model = load_artifact(source)
    validate_profile_artifact('notebook452', model)
    validate_profile_artifact('fixed21', load_artifact(fixed21))
    if source_hash != proof['restored_source_sha256']:
        raise ValueError('452項目モデルが元Notebook方式の検証記録と一致しません。')
    canonical = ROOT / 'artifacts/notebook_model_artifact.joblib'
    if sha256(canonical) != proof['canonical_sha256']:
        raise ValueError('元Notebookの比較基準モデルが変更されています。')
    if model['versions'] != proof['versions'] or model['versions'] != library_versions():
        raise ValueError('Notebook検証時とライブラリのバージョンが一致しません。')
    if (proof['full_cost_trials_per_fold'] != [1000] * 5
            or proof['full_original_source_parity']['checks'] != 112
            or proof['full_original_source_parity']['max_abs_diff'] != 0):
        raise ValueError('元Notebookの本番条件による一致検証が記録されていません。')
    deployed = build_deployment_artifact(model)
    deployed['metadata']['deployment_export']['source_model_sha256'] = source_hash
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    try:
        joblib.dump(deployed, temporary, compress=3)
        reloaded = load_artifact(temporary)
        validate_profile_artifact('notebook452', reloaded)
        for before, after in zip(model['folds'], reloaded['folds']):
            assert before['fold_name'] == after['fold_name']
            assert bytes(before['model'].get_booster().save_raw()) == bytes(after['model'].get_booster().save_raw())
            np.testing.assert_array_equal(before['selected_feature_mask'], after['selected_feature_mask'])
            np.testing.assert_array_equal(before['X_train_selected'], after['X_train_selected'])
            np.testing.assert_array_equal(before['pipeline'].named_steps['inputer']._fit_X,
                                          after['pipeline'].named_steps['inputer']._fit_X)
            for attr in ('mean_', 'scale_', 'var_'):
                np.testing.assert_array_equal(getattr(before['pipeline'].named_steps['standarization'], attr),
                                              getattr(after['pipeline'].named_steps['standarization'], attr))
        training = load_dataset(ROOT / 'data/DASS_join_data_opera_delete_LLNAnan.csv')
        expected_train_hash = model['metadata']['train_source']['sha256']
        if training['sha256'] != expected_train_hash:
            raise ValueError('予測照合に使う原本データがモデルの学習記録と一致しません。')
        original_prediction = predict_with_artifact(model, training['frame'])
        deployed_prediction = predict_with_artifact(reloaded, training['frame'])
        pd.testing.assert_frame_equal(original_prediction, deployed_prediction, check_exact=True)
        for name, frame in reloaded['reports'].items():
            pd.testing.assert_frame_equal(frame, deployed['reports'][name], check_exact=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256(fixed21) != original_hash or sha256(source) != source_hash:
        raise RuntimeError('元の学習済みモデルが変更されました。')
    template = Path(template)
    template.parent.mkdir(parents=True, exist_ok=True)
    columns = ['No.', 'Name', 'CAS Number', *model['full_feature_columns']]
    with template.open('w', encoding='utf-8-sig', newline='') as handle:
        csv.writer(handle, lineterminator='\n').writerow(columns)
    validation = {
        'workflow': 'dual_fixed21_and_notebook452',
        'weights_retrained': False,
        'fixed21_sha256_unchanged': original_hash,
        'notebook452_source_sha256': source_hash,
        'notebook452_deployment_sha256': sha256(output),
        'notebook452_deployment_bytes': output.stat().st_size,
        'full_preprocessing_columns': len(model['full_feature_columns']),
        'selected_feature_columns': model['selected_feature_columns'],
        'template_columns': len(columns),
        'prediction_export_parity': {'rows': len(training['frame']), 'max_abs_diff': 0},
        'canonical_sha256_verified': proof['canonical_sha256'],
        'historical_model_parity': 'unavailable',
        'source_validation': proof['full_original_source_parity'],
        'versions': library_versions(),
    }
    save_json(validation, report)
    return validation


def main() -> None:
    paths = model_paths(ROOT, {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/notebook_restored_model_artifact.joblib')
    parser.add_argument('--output', type=Path, default=paths['notebook452'])
    parser.add_argument('--fixed21', type=Path, default=paths['fixed21'])
    parser.add_argument('--proof', type=Path, default=ROOT / 'docs/notebook_restored_validation.json')
    parser.add_argument('--template', type=Path, default=ROOT / 'templates/csv452_input_template.csv')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/dual_models/packaging_validation.json')
    args = parser.parse_args()
    validation = prepare_models(source=args.source, output=args.output, proof_path=args.proof,
                                fixed21=args.fixed21, template=args.template, report=args.report)
    print(json.dumps({key: validation[key] for key in (
        'weights_retrained', 'fixed21_sha256_unchanged', 'notebook452_deployment_sha256',
        'template_columns', 'prediction_export_parity')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
