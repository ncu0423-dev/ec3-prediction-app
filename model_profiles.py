"""Explicit model choices for the independent 21- and 452-descriptor workflows."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from utils_io import FIXED21_FEATURE_COLUMNS

MODEL_LABELS = {
    'fixed21': '21項目版',
    'notebook452': '452項目版（元Notebook方式）',
}


def model_paths(root: Path, environ: Mapping[str, str] | None = None) -> dict[str, Path]:
    """Resolve each model separately; a missing model never selects the other one."""
    values = os.environ if environ is None else environ
    root = Path(root)
    configured = {
        'fixed21': values.get('EC3_FIXED21_ARTIFACT_PATH') or 'artifacts/original_model_artifact.joblib',
        'notebook452': values.get('EC3_NOTEBOOK452_ARTIFACT_PATH') or 'artifacts/notebook_deployment_artifact.joblib',
    }
    resolved = {}
    for profile, name in configured.items():
        path = Path(name).expanduser()
        resolved[profile] = path if path.is_absolute() else root / path
    return resolved


def validate_profile_artifact(profile: str, artifact: dict) -> None:
    """Reject a mislabeled artifact before showing a workflow's input controls."""
    if profile not in MODEL_LABELS:
        raise ValueError(f'不明なモデル種別です: {profile}')
    full = list(artifact['full_feature_columns'])
    selected = list(artifact['selected_feature_columns'])
    mode = artifact.get('metadata', {}).get('feature_mode', 'boruta')
    expected_count = 21 if profile == 'fixed21' else 452
    expected_mode = 'fixed21' if profile == 'fixed21' else 'boruta'
    if (mode != expected_mode or len(full) != expected_count
            or len(set(full)) != expected_count or selected != FIXED21_FEATURE_COLUMNS
            or (profile == 'fixed21' and full != FIXED21_FEATURE_COLUMNS)
            or (profile == 'notebook452' and 'MolWeight' not in full)):
        raise ValueError(f'{MODEL_LABELS[profile]}に対応する学習済みモデルではありません。')
    folds = artifact['folds']
    if len(folds) != 5:
        raise ValueError(f'{MODEL_LABELS[profile]}には学習済みの5 Foldが必要です。')
    for fold in folds:
        pipeline = fold['pipeline']
        steps = pipeline.steps
        names = list(getattr(pipeline, 'feature_names_in_', []))
        mask = np.asarray(fold.get('selected_feature_mask', artifact['selected_feature_mask']), dtype=bool)
        if (getattr(pipeline, 'n_features_in_', None) != expected_count
                or [name for name, _ in steps] != ['inputer', 'standarization']
                or not isinstance(steps[0][1], KNNImputer)
                or steps[0][1].n_neighbors != 5 or steps[0][1].weights != 'uniform'
                or not isinstance(steps[1][1], StandardScaler)
                or names != full or mask.shape != (expected_count,)
                or list(np.asarray(full)[mask]) != selected
                or fold.get('selected_feature_columns') != selected
                or getattr(fold['model'], 'n_features_in_', None) != len(selected)):
            raise ValueError(f'{MODEL_LABELS[profile]}の前処理・選択項目・モデル入力が一致しません。')
