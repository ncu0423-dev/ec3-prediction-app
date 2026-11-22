"""Ordered CSV schema for the current 21-descriptor input workflow.

Molecular weight is deliberately outside the requested 24-column template.
An optional MolWeight column is retained for callers that collect it in CSV;
its conversion requirements must be validated before prediction.
"""
from __future__ import annotations

import pandas as pd

from utils_io import FIXED21_FEATURE_COLUMNS, MW_COL, validate_numeric_columns

IDENTIFIER_COLUMNS = ('No.', 'Name', 'CAS Number')


def requested_csv_columns() -> list[str]:
    """Return a fresh list: three identifiers, then the user's 21 descriptors."""
    return [*IDENTIFIER_COLUMNS, *FIXED21_FEATURE_COLUMNS]


def make_template_frame() -> pd.DataFrame:
    """Create an empty CSV template in exactly the requested 24-column order."""
    return pd.DataFrame(columns=requested_csv_columns())


def normalize_csv_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and reorder a 21-descriptor CSV without changing row identities.

    Every requested header is required, but descriptor cells may be missing.
    Numeric strings are converted to floats; other strings and infinities fail.
    An optional MolWeight column is kept last, as provided, and is not a
    descriptor. It may still be incomplete while a user edits it on screen.
    Extra columns are rejected rather than silently becoming model inputs or
    leaking into exported results. Input frames are never modified in place.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('入力CSVにデータ行がありません。1行以上のデータを指定してください。')
    if not frame.columns.is_unique:
        raise ValueError('入力CSVに重複した列名があります。列名を一意にしてください。')
    required = requested_csv_columns()
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError('必要な列が不足しています: ' + ', '.join(missing))
    unexpected = [column for column in frame if column not in required and column != MW_COL]
    if unexpected:
        raise ValueError('指定外の列があります: ' + ', '.join(map(str, unexpected))
                         + '。入力は識別情報3列と指定21項目に揃えてください。MolWeightは換算用として任意で末尾に追加できます。')
    numeric = validate_numeric_columns(frame, FIXED21_FEATURE_COLUMNS, require_mw=False)
    columns = required + ([MW_COL] if MW_COL in frame else [])
    result = frame.loc[:, columns].copy()
    # Positional assignment preserves duplicate/nonmonotonic input indices.
    for column in FIXED21_FEATURE_COLUMNS:
        result[column] = numeric[column].to_numpy()
    return result
