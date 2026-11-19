"""Notebook presentation contracts: full preprocessing and native SHAP scope."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from original_runtime import load_artifact
from streamlit_app import (
    _crossfold_notebook_importance, _input_columns, _notebook_shap_importance,
    _training_explanation, _training_label, _validate_frame,
)


@pytest.fixture(scope='module')
def notebook_artifact(repo_root):
    for name in ('notebook_restored_model_artifact.joblib', 'notebook_model_artifact.joblib'):
        path = repo_root / 'artifacts' / name
        if path.is_file():
            return path, load_artifact(path)
    pytest.skip('A trusted original Notebook artifact is needed.')


def test_original_shap_sum_keeps_native_row_accumulation():
    values = np.array([[1e8], [1], [-1e8]], dtype=np.float32)
    expected = abs(sum(values))
    actual = _notebook_shap_importance(values)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] != abs(values.astype(np.float64).sum(axis=0))[0]


def test_original_shap_includes_every_training_row_and_feature(notebook_artifact):
    _, artifact = notebook_artifact
    for fold in artifact['folds']:
        explanation = _training_explanation(fold)
        assert len(explanation.values) == len(fold['X_train_selected']) > 100
        assert explanation.values.dtype == np.asarray(fold['shap_values']).dtype
        assert explanation.feature_names == fold['selected_feature_columns']
        np.testing.assert_array_equal(explanation.values, fold['shap_values'])
    expected = pd.DataFrame({f['fold_name']: abs(sum(f['shap_values'])) for f in artifact['folds']},
                            index=artifact['selected_feature_columns'])
    expected['Fold平均'] = expected.mean(axis=1)
    actual = _crossfold_notebook_importance(artifact).set_index('特徴量').reindex(expected.index)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)


def test_notebook_requires_full_descriptor_space(notebook_artifact):
    _, artifact = notebook_artifact
    assert _training_label(artifact) == 'Notebook再現条件'
    assert len(_input_columns(artifact)) == len(artifact['full_feature_columns']) == 452
    partial = pd.DataFrame([{name: 1. for name in artifact['selected_feature_columns']}])
    partial['MolWeight'] = 100.
    with pytest.raises(ValueError, match='不足'):
        _validate_frame(partial, artifact)


def test_notebook_home_and_mean_evaluation_are_explicit(repo_root, notebook_artifact, monkeypatch):
    path, artifact = notebook_artifact
    monkeypatch.setenv('EC3_ARTIFACT_PATH', str(path))
    app = AppTest.from_file(str(repo_root / 'streamlit_app.py'), default_timeout=90).run()
    assert not app.exception and not app.error
    text = '\n'.join(item.value for item in app.markdown)
    assert '全452項目' in text and 'KNNImputer' in text and 'Boruta' in text
    assert any('hCLAT.Call' in item.value for item in app.info)
    choices = [item.value for item in app.dataframe if list(item.value.columns) == ['特徴量']]
    assert choices[0]['特徴量'].tolist() == artifact['selected_feature_columns']
    app.sidebar.radio(key='navigation').set_value('モデルの性能評価').run()
    assert not app.exception and not app.error
    app.selectbox(key='external_prediction_method').set_value('平均（Notebookの比較用）').run()
    assert not app.exception and not app.error
    assert any('平均yを逆変換した値' in item.value for item in app.caption)
    labels = [item.label for item in app.expander]
    assert any('分類・5倍範囲評価' in label for label in labels)


def test_primary_ad_uses_any_fold_and_retains_chosen_fold_detail():
    source = """
import pandas as pd
from streamlit_app import _render_prediction
_render_prediction(pd.DataFrame([{'Predict_LLNA EC3': 1., 'selected_fold': 'fold_2',
 'kNN_AD': False, 'any_kNN_AD': True, 'knn_inAD_Predict_LLNA EC3': 3.,
 'selected_inAD_fold': 'fold_1'}]))
"""
    app = AppTest.from_string(source).run()
    assert not app.exception and not app.error
    assert app.metric[1].value == '範囲内'
    assert any('Fold単独のAD: 範囲外' in item.value for item in app.caption)


def test_classification_ui_displays_both_error_directions_and_undefined_rates():
    source = """
import numpy as np
import pandas as pd
from streamlit_app import _notebook_evaluation_tables
_notebook_evaluation_tables(pd.DataFrame({'LLNA EC3(NS converted to 150%)': [.05, 150.],
 'Predict_LLNA EC3': [150., .05]}))
_notebook_evaluation_tables(pd.DataFrame({'LLNA EC3(NS converted to 150%)': [1.],
 'Predict_LLNA EC3': [np.nan]}))
"""
    app = AppTest.from_string(source).run()
    assert not app.exception and not app.error
    text = '\n'.join(item.value for item in app.markdown)
    assert '感作性の過小評価率: 50.000%' in text
    assert '感作性の過大評価率: 50.000%' in text
    assert '感作性の過小評価率: 算出不可' in text
