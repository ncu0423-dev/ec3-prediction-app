"""Export a trusted model for the private Community Cloud repository.

Numeric fitted KNN/AD/SHAP references remain necessary for prediction. This is
an identity-reduced export, not anonymization of all training information.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from original_runtime import load_artifact

DEPLOYMENT_KEYS = (
    'format_version', 'full_feature_columns', 'selected_feature_mask',
    'selected_feature_columns', 'target_col', 'folds', 'metadata', 'metrics',
    'versions', 'reports',
)
REPORT_COLUMNS = (
    'fold', 'selected_fold', 'true_y', 'LLNA EC3(NS converted to 150%)',
    'pred_y', 'Predict_LLNA EC3', 'kNN_AD', 'any_kNN_AD',
    'selected_inAD_fold', 'knn_inAD_pred_y', 'knn_inAD_Predict_LLNA EC3',
    'knn_distance', 'Fold_misprediction',
)


def build_deployment_artifact(artifact):
    deployed = {key: artifact[key] for key in DEPLOYMENT_KEYS if key in artifact}
    deployed['metadata'] = dict(artifact['metadata'])
    deployed['metadata']['deployment_export'] = {
        'report_identifiers_removed': True,
        'unused_global_training_data_removed': True,
        'numeric_fitted_references_retained': True,
        'purpose': 'private_repository_community_cloud',
    }
    deployed['reports'] = {
        name: frame[[c for c in REPORT_COLUMNS if c in frame]].copy()
        for name, frame in artifact.get('reports', {}).items()
        if name in ('oof', 'external')
    }
    for frame in deployed['reports'].values():
        frame.reset_index(drop=True, inplace=True)
        frame.attrs = {}
        # These are model labels, never user-provided chemical identifiers.
        for column in ('fold', 'selected_fold', 'selected_inAD_fold'):
            if column in frame:
                labels = frame[column].dropna().astype(str)
                if not labels.str.fullmatch(r'fold_\d+').all():
                    raise ValueError(f'Unexpected model label in {column}')
    return deployed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Trusted local joblib artifact')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if source.resolve() == output.resolve():
        parser.error('Keep the source model unchanged; use a separate output path.')
    artifact = load_artifact(source)
    deployed = build_deployment_artifact(artifact)
    deployed['metadata']['deployment_export']['source_model_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    try:
        joblib.dump(deployed, temporary, compress=3)
        load_artifact(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Saved deployment model: {output} ({output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
