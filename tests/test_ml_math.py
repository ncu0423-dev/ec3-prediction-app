"""Hand-calculated numerical checks independent of the implementation details."""
import numpy as np

from utils_ml import compute_knn_ad, convert_ec3_to_y, convert_y_to_ec3, make_llna_labels


def test_ec3_transform_has_no_extra_percent_or_unit_multiplier():
    ec3 = np.array([0.1, 2.0, 10.0, 150.0])
    mw = np.array([100.0, 200.0, 100.0, 150.0])
    expected_y = np.array([3.0, 2.0, 1.0, 0.0])
    np.testing.assert_allclose(convert_ec3_to_y(ec3, mw), expected_y)
    np.testing.assert_allclose(convert_y_to_ec3(expected_y, mw), ec3)


def test_notebook_labels_keep_exact_150_boundary():
    values = np.array([0.1, 2.0, np.nextafter(2.0, np.inf), 10.0,
                       np.nextafter(150.0, 0), 150.0, 151.0])
    # Notebook first labels ONLY exactly 150 as zero, then <=2 as two.
    np.testing.assert_array_equal(make_llna_labels(values), [2, 2, 1, 1, 1, 0, 1])


def test_knn_ad_excludes_train_self_and_uses_notebook_order_statistic():
    train = np.array([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0]])
    query = np.array([[2.0], [17.0], [18.0]])
    # Train mean two-neighbor distances: [2,1.5,2.5,3.5,4.5,7].
    # The notebook uses sorted[round(6 * .9)-1] == sorted[4] == 4.5.
    # This is distinct from an interpolated 90th percentile (5.75).
    in_ad, distances, threshold = compute_knn_ad(train, query, knn_k=2, ad_coverage=0.1)
    assert threshold == 4.5
    np.testing.assert_allclose(distances, [1.0, 4.5, 5.5])
    np.testing.assert_array_equal(in_ad, [True, True, False])


def test_preprocessing_matches_fit_then_transform_without_refit():
    from utils_ml import make_preprocess_pipeline

    train = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    query = np.array([[2.0, np.nan]])
    pipeline = make_preprocess_pipeline()
    pipeline.fit(train)
    # Mean imputation gives [2,20], exactly the center of the fit scaler.
    np.testing.assert_allclose(pipeline.transform(query), [[0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(pipeline.transform([[4.0, 30.0]]), [[np.sqrt(1.5), np.sqrt(1.5)]])
