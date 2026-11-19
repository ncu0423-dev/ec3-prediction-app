"""EC3 evaluation rules from notebook cells 66, 71 and 72.

The notebook's ``over`` means stronger predicted sensitization (smaller EC3),
and ``under`` means weaker predicted sensitization (larger EC3). Confusion
matrix rows are predicted classes and columns are observed classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils_ml import valid_regression_metrics

GHS_LABELS = ('1A', '1B', 'NC')
ECETOC_LABELS = ('Extreme', 'Strong', 'Moderate', 'Weak', 'ns')


def classify_ghs(ec3):
    """Original boundaries: <=2, 2<x<=100, >100."""
    return np.searchsorted(np.asarray([2., 100.]), np.asarray(ec3, dtype=float), side='left')


def classify_ecetoc(ec3):
    """Original strict upper boundaries: <.1, <1, <10, <100, >=100."""
    return np.searchsorted(np.asarray([.1, 1., 10., 100.]), np.asarray(ec3, dtype=float), side='right')


def _range_row(observed, predicted):
    n = len(observed)
    lower, upper = observed / 5, observed * 5
    within = (lower < predicted) & (predicted < upper)
    over = predicted < lower
    under = upper < predicted
    # Exact 1/5 and 5 boundaries were excluded by all three notebook modes.
    boundary = ~(within | over | under)
    result = {'n': n}
    for name, mask in (('within', within), ('over', over), ('under', under), ('boundary', boundary)):
        count = int(mask.sum())
        result[name + '_count'] = count
        result[name + '_rate'] = count / n if n else None
    return result


def _classification_result(observed, predicted, classifier, labels):
    actual, estimated = classifier(observed), classifier(predicted)
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(counts, (estimated, actual), 1)
    matrix = pd.DataFrame(counts, index=pd.Index(labels, name='Predicted'),
                          columns=pd.Index(labels, name='Observed'))
    n = len(observed)
    return {
        'matrix': matrix,
        'n': n,
        'accuracy': float(np.mean(actual == estimated)) if n else None,
        # Larger category indices mean weaker predicted sensitization.
        'underestimated': float(np.mean(actual < estimated)) if n else None,
        'overestimated': float(np.mean(actual > estimated)) if n else None,
    }


def evaluate_notebook_report(report, prediction_column='Predict_LLNA EC3',
                             truth_column='LLNA EC3(NS converted to 150%)'):
    """Evaluate an EC3 report without pandas index alignment or unit conversion.

    Input rows with missing/nonfinite observed or predicted values are omitted
    consistently from every metric and counted in ``excluded_count``. This
    supports reports in which every Fold is outside AD. A zero-row evaluation
    returns counts of zero and undefined rates of None. Finite nonpositive EC3
    is rejected because the application's EC3 target/inverse require positivity.

    R2/RMSE are original metrics; MAE is retained as an explicitly additional
    metric for callers. This function must not receive log-transformed y values.
    """
    missing = [name for name in (truth_column, prediction_column) if name not in report]
    if missing:
        raise ValueError('評価に必要な列がありません: ' + ', '.join(missing))
    observed = pd.to_numeric(report[truth_column], errors='coerce').to_numpy(dtype=float)
    predicted = pd.to_numeric(report[prediction_column], errors='coerce').to_numpy(dtype=float)
    if observed.ndim != 1 or predicted.ndim != 1:
        raise ValueError('評価列名が重複しています。列名を一意にしてください。')
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if (observed <= 0).any() or (predicted <= 0).any():
        raise ValueError('EC3の評価には0より大きい値が必要です。')
    groups = {'All': np.ones(len(observed), dtype=bool)}
    group_classes = classify_ghs(observed)
    groups.update({label: group_classes == index for index, label in enumerate(GHS_LABELS)})
    ranges = pd.DataFrame.from_dict({
        label: _range_row(observed[mask], predicted[mask]) for label, mask in groups.items()
    }, orient='index')
    ranges.index.name = 'Observed GHS group'
    return {
        'regression': valid_regression_metrics(observed, predicted),
        'range_rates': ranges,
        'ghs': _classification_result(observed, predicted, classify_ghs, GHS_LABELS),
        'ecetoc': _classification_result(observed, predicted, classify_ecetoc, ECETOC_LABELS),
        'excluded_count': int((~valid).sum()),
        'input_count': len(valid),
        'prediction_column': prediction_column,
        'truth_column': truth_column,
        'units': 'EC3 (%)',
        'additional_metrics': ['mae'],
    }
