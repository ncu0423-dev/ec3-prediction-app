"""Input contracts: preserve identifiers and reject data that cannot be interpreted safely."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from utils_io import load_dataset, read_csv_auto, validate_numeric_columns


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp932"])
def test_japanese_csv_roundtrip_and_column_order(encoding):
    text = "Name,MolWeight,descriptor\n物質Ａ,100,1\n物質Ｂ,200,\n"
    frame, detected = read_csv_auto(BytesIO(text.encode(encoding)))
    assert detected in ("utf-8-sig", "cp932", "shift_jis")
    assert frame.columns.tolist() == ["Name", "MolWeight", "descriptor"]
    assert frame.Name.tolist() == ["物質Ａ", "物質Ｂ"]
    assert np.isnan(frame.loc[1, "descriptor"])


@pytest.mark.parametrize("text", [
    "a,a\n1,2\n",  # pandas otherwise mangles duplicate names without warning
    "a,\n1,2\n",
    "a,b\n1,2,3\n",  # pandas can silently reinterpret the first field as an index
    "a,b\n1\n",  # a missing CSV field is different from an explicitly empty cell
    "a,b\n",
    "",
])
def test_invalid_csv_shape_or_header_is_rejected(text):
    with pytest.raises(ValueError):
        read_csv_auto(text.encode())


def test_blank_numeric_cells_are_imputable_but_absent_columns_are_not():
    frame = pd.DataFrame({"Name": ["a", "b"], "x": ["", "  "], "MolWeight": [100, 200]})
    numeric = validate_numeric_columns(frame, ["MolWeight", "x"])
    assert numeric.columns.tolist() == ["MolWeight", "x"]
    assert numeric.x.isna().all()
    with pytest.raises(ValueError, match="x"):
        validate_numeric_columns(frame.drop(columns="x"), ["MolWeight", "x"])


@pytest.mark.parametrize("bad", ["unknown", "1,234", np.inf, -np.inf])
def test_non_numeric_and_infinite_descriptors_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_numeric_columns(pd.DataFrame({"MolWeight": [100], "x": [bad]}), ["x"])


@pytest.mark.parametrize("mw", [None, "", "invalid", 0, -1, np.inf, -np.inf])
def test_molecular_weight_is_required_positive_finite(mw):
    with pytest.raises(ValueError, match="MolWeight"):
        validate_numeric_columns(pd.DataFrame({"MolWeight": [mw], "x": [1]}), ["x"])


def test_duplicate_dataframe_columns_rejected():
    frame = pd.DataFrame([[100, 1, 2]], columns=["MolWeight", "x", "x"])
    with pytest.raises(ValueError):
        validate_numeric_columns(frame, ["MolWeight", "x"])


def test_external_missing_mw_rows_are_filtered_without_reordering(tmp_path):
    path = tmp_path / "external.csv"
    pd.DataFrame({
        "Name": ["first", "dropped", "last"],
        "MolWeight": [100, np.nan, 200],
        "x": [1, 2, 3],
        "LLNA_EC3": [10, 20, 30],
    }).to_csv(path, index=False)
    result = load_dataset(path, feature_columns=["MolWeight", "x"], external=True)
    assert result["frame"].Name.tolist() == ["first", "last"]
    np.testing.assert_array_equal(result["source_rows"], [0, 2])
    assert result["dropped_missing_mw_rows"] == [1]
    assert result["X"].columns.tolist() == ["MolWeight", "x"]
    np.testing.assert_allclose(result["ec3"], [10, 30])
