"""Shared fixtures for locally generated, trusted test models."""
from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ec3-pytest-matplotlib")


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture
def synthetic_artifact():
    """A transparent saved ensemble: y=1 and y=2, so min EC3 must select Fold 2."""
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyRegressor
    from utils_ml import make_preprocess_pipeline

    full_columns = ["x", "MolWeight", "z"]
    train = pd.DataFrame({"x": range(6), "MolWeight": np.arange(1, 7) * 100., "z": range(6)})
    pipeline = make_preprocess_pipeline().fit(train)
    selected = pipeline.transform(train)[:, [True, False, True]]
    folds = []
    for index, y in enumerate([1., 2.], start=1):
        folds.append({
            "fold_name": f"fold_{index}",
            "pipeline": pipeline,
            "model": DummyRegressor(strategy="constant", constant=y).fit(selected, np.repeat(y, 6)),
            "selected_feature_mask": np.array([True, False, True]),
            "selected_feature_columns": ["x", "z"],
            "X_train_selected": selected,
            "ad_threshold": 1000. if index == 1 else 0.,
        })
    return {
        "format_version": 2,
        "full_feature_columns": full_columns,
        "selected_feature_mask": np.array([True, False, True]),
        "selected_feature_columns": ["x", "z"],
        "folds": folds,
        "metadata": {"knn_k": 2, "ad_coverage": .1, "training_mode": "smoke"},
    }


@pytest.fixture
def synthetic_input():
    import pandas as pd

    return pd.DataFrame({
        "Name": ["second ID", "first ID", "same ID"],
        "CAS Number": ["a", "b", "c"],
        "z": [1., 2., 3.],
        "MolWeight": [100., 250., 400.],
        "x": [1., 2., 3.],
    }, index=[8, 8, 2])


@pytest.fixture(scope="session")
def smoke_artifact_path(repo_root):
    """Use the dedicated smoke artifact, without substituting legacy model bundles."""
    configured = os.environ.get("EC3_TEST_ARTIFACT")
    candidates = ([Path(configured)] if configured else [
        repo_root / "artifacts" / "smoke_model_artifact.joblib",
        repo_root / "artifacts" / "original_model_artifact.joblib",
    ])
    for path in candidates:
        if path.is_file():
            return path
    pytest.skip("Generate an explicit smoke artifact with train.py before integration checks.")
