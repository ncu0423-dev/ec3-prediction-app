"""CSV headers and row identity follow the user's requested 24-column layout."""
import numpy as np
import pandas as pd
import pytest

from csv_input_schema import (
    IDENTIFIER_COLUMNS,
    make_template_frame,
    normalize_csv_input,
    requested_csv_columns,
)

EXPECTED = [
    'No.', 'Name', 'CAS Number',
    'OVERALL OH rate constant', 'FM advection sediment', 'LUMO Energy',
    'HOMO Energy', 'CombDipolPolariz', 'LogHL_pred', 'LogKM_pred',
    'alert_Protein_binding_by_OASIS',
    'judge_Protein_binding_alerts_for_skin_sensitization_according_to_GHS',
    'potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS',
    'DPRA.percCysdep', 'DPRA.percLysdep', 'DPRA.score',
    'hCLAT.CD86.EC150..ug.ml.', 'h.CLAT.CV75', 'hCLAT.MIT', 'hCLAT.score',
    'KS.EC1.5', 'KS.EC3', 'KS.IC50', 'sum_invitro',
]


@pytest.fixture
def input_frame():
    frame = pd.DataFrame({column: [1., 2., 3.] for column in EXPECTED[3:]}, index=[9, 9, 2])
    frame['No.'] = ['001', '001', '003']
    frame['Name'] = ['物質B', '物質A', '物質A']
    frame['CAS Number'] = ['50-00-0', '50-00-0', '64-17-5']
    return frame[EXPECTED]


def test_exact_requested_template_has_no_extra_descriptors_or_mw():
    assert IDENTIFIER_COLUMNS == ('No.', 'Name', 'CAS Number')
    assert requested_csv_columns() == EXPECTED
    assert len(EXPECTED) == 24
    assert make_template_frame().empty
    assert make_template_frame().columns.tolist() == EXPECTED
    assert not {'MolWeight', 'SMILES', 'VdW surface DPSA2', 'FM advection air'} & set(EXPECTED)
    detached = requested_csv_columns()
    detached.pop()
    assert requested_csv_columns() == EXPECTED


def test_reorders_without_losing_duplicate_ids_rows_or_index(input_frame):
    original = input_frame.copy(deep=True)
    result = normalize_csv_input(input_frame[EXPECTED[::-1]])
    assert result.columns.tolist() == EXPECTED
    assert result.index.tolist() == [9, 9, 2]
    pd.testing.assert_frame_equal(result[list(IDENTIFIER_COLUMNS)], original[list(IDENTIFIER_COLUMNS)])
    pd.testing.assert_frame_equal(input_frame, original)
    pd.testing.assert_frame_equal(result, original)


def test_no_mw_needed_until_separate_conversion_input(input_frame):
    result = normalize_csv_input(input_frame)
    assert 'MolWeight' not in result
    assert len(result.columns) == 24


def test_optional_mw_is_preserved_at_end_for_later_validation(input_frame):
    input_frame.insert(0, 'MolWeight', [None, '180.16', 'not ready'])
    result = normalize_csv_input(input_frame)
    assert result.columns.tolist() == EXPECTED + ['MolWeight']
    pd.testing.assert_series_equal(result['MolWeight'], input_frame['MolWeight'])
    assert pd.isna(result['MolWeight'].iloc[0])
    assert result['MolWeight'].iloc[1:].tolist() == ['180.16', 'not ready']


def test_numeric_text_and_empty_cells_are_normalized(input_frame):
    input_frame['HOMO Energy'] = ['-2.5', ' ', None]
    result = normalize_csv_input(input_frame)
    assert result['HOMO Energy'].iloc[0] == -2.5
    assert result['HOMO Energy'].iloc[1:].isna().all()
    assert np.issubdtype(result['HOMO Energy'].dtype, np.floating)


@pytest.mark.parametrize('missing', ['No.', 'Name', 'CAS Number', 'FM advection sediment'])
def test_required_identifier_and_descriptor_columns_are_not_silently_added(input_frame, missing):
    with pytest.raises(ValueError, match='不足'):
        normalize_csv_input(input_frame.drop(columns=missing))


@pytest.mark.parametrize('column', ['VdW surface DPSA2', 'SMILES', 'LLNA_EC3'])
def test_unrequested_columns_are_rejected(input_frame, column):
    input_frame[column] = 'unexpected'
    with pytest.raises(ValueError, match='指定外'):
        normalize_csv_input(input_frame)


@pytest.mark.parametrize('value', ['not a number', np.inf, -np.inf])
def test_invalid_descriptor_values_fail_before_prediction(input_frame, value):
    input_frame['KS.EC3'] = pd.Series([value] * len(input_frame), index=input_frame.index)
    with pytest.raises(ValueError, match='KS.EC3'):
        normalize_csv_input(input_frame)


def test_duplicate_headers_are_rejected(input_frame):
    duplicated = pd.concat([input_frame, input_frame[['HOMO Energy']]], axis=1)
    with pytest.raises(ValueError, match='重複'):
        normalize_csv_input(duplicated)


@pytest.mark.parametrize('frame', [None, pd.DataFrame(columns=EXPECTED)])
def test_empty_or_non_dataframe_inputs_are_rejected(frame):
    with pytest.raises(ValueError, match='データ行'):
        normalize_csv_input(frame)
