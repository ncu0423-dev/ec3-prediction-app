"""Real XGBoost SHAP additivity and all requested plot types, using a tiny model."""
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import XGBRegressor

from utils_ml import shap_explanation


def test_real_xgboost_shap_additivity_and_rendering():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=float)
    y = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    model = XGBRegressor(n_estimators=8, max_depth=2, learning_rate=.4, n_jobs=1, random_state=0)
    model.fit(X, y)
    explanation = shap_explanation(model, X, ["first", "second"])
    assert explanation.values.shape == X.shape
    assert explanation.feature_names == ["first", "second"]
    np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1),
                               model.predict(X), rtol=1e-5, atol=1e-6)

    plots = [
        lambda: shap.summary_plot(explanation.values, X, feature_names=explanation.feature_names, show=False),
        lambda: shap.plots.beeswarm(explanation, show=False),
        lambda: shap.plots.waterfall(explanation[0], show=False),
        lambda: shap.plots.bar(explanation, show=False),
    ]
    for plot in plots:
        plot()
        output = BytesIO()
        plt.gcf().savefig(output, format="png", dpi=60)
        assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
        plt.close("all")
