"""Notebook calculations kept explicit, including documented reproduction limitations."""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

NOTEBOOK_XGB_BASE_PARAMS = {
    'max_depth': 6, 'learning_rate': .03, 'n_estimators': 100000, 'verbosity': 1,
    'booster': 'gbtree', 'n_jobs': 1, 'gamma': 0, 'min_child_weight': 1,
    'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1,
    'colsample_bylevel': 1, 'colsample_bynode': 1, 'reg_alpha': .5,
    'reg_lambda': .1, 'scale_pos_weight': 1, 'base_score': None,
    'random_state': 0, 'missing': float('nan'), 'objective': 'reg:squarederror',
    'eval_metric': 'rmse', 'early_stopping_rounds': 15,
}


def convert_ec3_to_y(ec3, mw):
    return np.log10(np.asarray(mw, dtype=float) / np.asarray(ec3, dtype=float))


def convert_y_to_ec3(y, mw):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        return np.asarray(mw, dtype=float) / (10 ** np.asarray(y))


def make_llna_labels(ec3):
    """Call with round-tripped EC3, as the notebook does (including exact ==150)."""
    values = np.asarray(ec3)
    return np.where(values == 150, 0, np.where(values <= 2, 2, 1)).astype(int)


def make_preprocess_pipeline():
    # Original step names retained for direct independent notebook comparisons.
    return Pipeline([('inputer', KNNImputer(n_neighbors=5, weights='uniform')),
                     ('standarization', StandardScaler())])


def clone_and_transform(X_train, X_query):
    pipeline = make_preprocess_pipeline()
    training = pipeline.fit_transform(X_train)
    query = pipeline.transform(X_query)
    if training.shape[1] != X_train.shape[1]:
        raise ValueError('学習Foldに全欠損の特徴量があります。NotebookのKNNImputerで列が消失するため再現できません。')
    return training, query, pipeline


def fit_global_boruta(X, y, *, rf_depth=5, perc=100, n_jobs=2):
    """Global Boruta intentionally preserves notebook leakage into outer CV."""
    from boruta import BorutaPy
    transformed, _, pipeline = clone_and_transform(X, X)
    selector = BorutaPy(RandomForestRegressor(n_jobs=n_jobs, max_depth=rf_depth),
                        n_estimators='auto', verbose=0, random_state=0, perc=perc)
    selector.fit(transformed, np.asarray(y))
    if not selector.support_.any():
        raise ValueError('Borutaで特徴量が選択されませんでした。データとバージョンを確認してください。')
    return selector, pipeline, transformed


def valid_regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(valid.sum())
    if not n:
        return {'n': 0, 'r2': None, 'rmse': None, 'mae': None}
    return {'n': n, 'r2': float(r2_score(y_true[valid], y_pred[valid])) if n>1 else None,
            'rmse': float(mean_squared_error(y_true[valid], y_pred[valid]) ** .5),
            'mae': float(mean_absolute_error(y_true[valid], y_pred[valid]))}


def fit_knn_ad(X_train, *, knn_k=5, ad_coverage=.1):
    """coverage is the notebook's EXCLUDED fraction, with an order statistic."""
    X_train = np.asarray(X_train, dtype=float)
    if not isinstance(knn_k, int) or not 1 <= knn_k < len(X_train):
        raise ValueError('knn_kは1以上かつ学習行数未満にしてください。')
    if not 0 <= ad_coverage <= 1:
        raise ValueError('AD coverageは0から1の範囲です（AD外割合）。')
    distances = np.sort(cdist(X_train, X_train), axis=1)
    means = np.sort(np.mean(distances[:, 1:knn_k+1], axis=1))
    # Do not replace by np.quantile: interpolation would change notebook behavior.
    return float(means[round(len(X_train) * (1-ad_coverage))-1])


def compute_knn_ad(X_train, X_query, *, knn_k=5, ad_coverage=.1, threshold=None):
    if threshold is None:
        threshold = fit_knn_ad(X_train, knn_k=knn_k, ad_coverage=ad_coverage)
    distances = cdist(np.asarray(X_query), np.asarray(X_train))
    means = np.mean(np.sort(distances, axis=1)[:, :knn_k], axis=1)
    return means <= threshold, means, float(threshold)


def _fit_model(params, X, y, *, eval_set=None):
    model = XGBRegressor(**deepcopy(params))
    kwargs = {'eval_set': eval_set, 'verbose': False} if eval_set is not None else {}
    model.fit(X, y, **kwargs)
    return model


def optimize_xgb_params(X, y, labels, *, n_trials=1000, max_estimators=100000,
                        inner_folds=5, n_jobs=1):
    import optuna
    if n_trials < 1:
        raise ValueError('Optunaは1 trial以上必要です。無効化はできません。')
    labels = np.asarray(labels)
    if min(np.unique(labels, return_counts=True)[1]) < inner_folds:
        raise ValueError('内部StratifiedKFoldに必要な各クラスの行数が不足しています。')
    base = deepcopy(NOTEBOOK_XGB_BASE_PARAMS)
    base.update(n_estimators=max_estimators, n_jobs=n_jobs)
    best_iterations = []
    splits = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=0).split(X, labels))

    def objective(trial):
        search = {
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'gamma': trial.suggest_float('gamma', 1e-8, 1., log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
            'subsample': trial.suggest_float('subsample', .2, 1.),
            'colsample_bytree': trial.suggest_float('colsample_bytree', .2, 1.),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1., log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1., log=True),
        }
        scores, iterations = [], []
        for train_idx, validation_idx in splits:
            model = _fit_model(base | search, X[train_idx], y[train_idx],
                               eval_set=[(X[validation_idx], y[validation_idx])])
            iterations.append(int(model.best_iteration))
            scores.append(float(r2_score(y[validation_idx], model.predict(X[validation_idx]))))
        best_iterations.append(iterations)
        trial.set_user_attr('best_iterations', iterations)
        trial.set_user_attr('inner_r2', scores)
        return float(np.mean(scores))

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    params = base | study.best_params
    # The original uses zero-based best_iteration directly, WITHOUT +1.
    params['n_estimators'] = max(best_iterations[study.best_trial.number])
    del params['early_stopping_rounds']
    details = {'best_trial': study.best_trial.number, 'best_score': study.best_value,
               'trials': [{'number': t.number, 'value': t.value, 'params': t.params,
                           'state': t.state.name, 'user_attrs': t.user_attrs} for t in study.trials],
               'inner_splits': [{'train': a, 'validation': b} for a,b in splits]}
    return params, details


def extract_invitro_calls(X):
    """Never invent hCLAT.Call from hCLAT.score: original call source is missing."""
    n = len(X)
    out = pd.DataFrame(index=np.arange(n))
    if {'DPRA.score', 'KS.Call', 'hCLAT.Call'}.issubset(X.columns):
        calls = ((pd.to_numeric(X['DPRA.score']) > 0).astype(int).to_numpy()
                 + pd.to_numeric(X['KS.Call']).fillna(0).to_numpy()
                 + pd.to_numeric(X['hCLAT.Call']).fillna(0).to_numpy())
        out['vitro_call_sum'], out['invitro_AD'] = calls, calls >= 2
    else:
        out['vitro_call_sum'] = np.nan
        out['invitro_AD'] = pd.array([pd.NA]*n, dtype='boolean')
    return out


def shap_explanation(model, transformed, feature_names):
    import shap
    values = np.asarray(transformed, dtype=float)
    explanation = shap.TreeExplainer(model)(values)
    explanation.feature_names = list(feature_names)
    return explanation


def save_shap_artifacts_for_fold(fold, X, feature_names):
    explanation = shap_explanation(fold['model'], X, feature_names)
    fold.update(shap_values=np.asarray(explanation.values),
                shap_base_values=np.asarray(explanation.base_values), X_shap=np.asarray(X),
                shap_scope='outer_training')
    return explanation


def library_versions():
    names = ('numpy','pandas','scipy','scikit-learn','xgboost','optuna','Boruta','shap','joblib')
    return {name: importlib.metadata.version(name) for name in names}
