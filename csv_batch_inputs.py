"""Per-row conversion inputs, joined by upload-row position rather than chemical IDs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from csv_input_schema import IDENTIFIER_COLUMNS
from utils_io import MW_COL, validate_numeric_columns

ROW_COLUMN = '入力行'


def mw_entry_frame(frame):
    entries = frame[list(IDENTIFIER_COLUMNS)].reset_index(drop=True).copy()
    entries.insert(0, ROW_COLUMN, np.arange(1, len(frame) + 1))
    entries[MW_COL] = (pd.to_numeric(frame[MW_COL], errors='coerce').to_numpy(dtype=float)
                       if MW_COL in frame else np.full(len(frame), np.nan))
    return entries


def attach_molecular_weights(frame, edited):
    if (not isinstance(edited, pd.DataFrame) or not edited.columns.is_unique
            or not {ROW_COLUMN, MW_COL}.issubset(edited.columns)
            or len(edited) != len(frame)):
        raise ValueError('分子量入力の行数・列が一致しません。CSVを再度選択してください。')
    row_ids = pd.to_numeric(edited[ROW_COLUMN], errors='coerce')
    expected = np.arange(1, len(frame) + 1)
    if row_ids.isna().any() or not np.array_equal(np.sort(row_ids.to_numpy()), expected):
        raise ValueError('分子量入力の行の対応が不正です。CSVを再度選択してください。')
    weights = pd.Series(edited[MW_COL].to_numpy(), index=row_ids).reindex(expected)
    combined = frame.copy()
    combined[MW_COL] = weights.to_numpy()
    # The runtime validates again; do not infer a mass or silently replace invalid cells.
    combined[MW_COL] = validate_numeric_columns(combined, [MW_COL])[MW_COL].to_numpy()
    return combined
