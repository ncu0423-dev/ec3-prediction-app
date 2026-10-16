"""CSV validation shared by training and inference; uploaded data is never saved implicitly."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

MW_COL = 'MolWeight'
TARGET_COLUMN_DEFAULT = 'LLNA_EC3'
META_COLUMNS = ('No.', 'Name', 'CAS Number', 'SMILES', 'CAS_Number_qsar', 'SMILES_qsar')
NOTEBOOK_SELECTED_FEATURE_COLUMNS = [
    'OVERALL OH rate constant', 'FM advection air', 'VdW surface DPSA2',
    'LUMO Energy', 'HOMO Energy', 'MolWeight', 'CombDipolPolariz', 'LogHL_pred',
    'LogKM_pred', 'alert_Protein_binding_by_OASIS',
    'potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS',
    'DPRA.percCysdep', 'DPRA.percLysdep', 'DPRA.score',
    'hCLAT.CD86.EC150..ug.ml.', 'hCLAT.CD54.EC200..ug.ml.', 'h.CLAT.CV75',
    'hCLAT.MIT', 'hCLAT.score', 'KS.EC1.5', 'KS.EC3', 'KS.IC50', 'sum_invitro',
]


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_auto(source):
    """Read UTF-8 (BOM supported), CP932, or Shift-JIS without losing headers."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, bytes):
        raw = source
    elif hasattr(source, 'getvalue'):
        raw = source.getvalue()
    else:
        raw = source.read()
    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    if not raw:
        raise ValueError('CSVファイルが空です。')
    last_error = None
    for encoding in ('utf-8-sig', 'cp932', 'shift_jis'):
        try:
            content = raw.decode(encoding)
        except UnicodeDecodeError as error:
            last_error = error
            continue
        parsed_rows = csv.reader(io.StringIO(content))
        header = next(parsed_rows, [])
        if not header or any(not c.strip() for c in header):
            raise ValueError('CSVの列名が空です。ヘッダー行を確認してください。')
        if len(header) != len(set(header)):
            raise ValueError('CSVに重複する列名があります。列名を一意にしてください。')
        for line_number, row in enumerate(parsed_rows, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f'CSVの{line_number}行目の列数がヘッダーと一致しません（{len(row)} / {len(header)}）。')
        try:
            frame = pd.read_csv(io.StringIO(content))
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            raise ValueError(f'CSVの行または列の形式が不正です: {error}') from error
        if frame.empty:
            raise ValueError('CSVにデータ行がありません。')
        return frame, encoding
    raise ValueError('CSVの文字コードを読み取れません。UTF-8またはCP932で保存してください。') from last_error


def validate_numeric_columns(frame, columns, *, require_mw=True):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('入力データが空です。1行以上のデータを指定してください。')
    if not frame.columns.is_unique:
        raise ValueError('入力データに重複した列名があります。')
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError('必要な列が不足しています: ' + ', '.join(missing[:12]) +
                         (f' （ほか{len(missing)-12}列）' if len(missing)>12 else ''))
    numeric_values = {}
    for name in columns:
        values = frame[name].replace(r'^\s*$', np.nan, regex=True)
        converted = pd.to_numeric(values, errors='coerce')
        invalid = values.notna() & converted.isna()
        if invalid.any():
            rows = (np.flatnonzero(invalid.to_numpy())[:5] + 1).tolist()
            raise ValueError(f'{name}: 数値以外の値があります（データ行 {rows}）。欠損値は空欄にしてください。')
        if np.isinf(converted.to_numpy(dtype=float)).any():
            raise ValueError(f'{name}: 無限大は使用できません。有限の数値または空欄を入力してください。')
        numeric_values[name] = converted.astype(float)
    numeric = pd.DataFrame(numeric_values, index=frame.index)
    if require_mw:
        if MW_COL not in numeric:
            if MW_COL not in frame:
                raise ValueError('MolWeight列が必要です。')
            molecular_weight = validate_numeric_columns(frame, [MW_COL], require_mw=False)[MW_COL]
        else:
            molecular_weight = numeric[MW_COL]
        if (molecular_weight.isna() | (molecular_weight <= 0)).any():
            raise ValueError('MolWeightは全行で有限かつ0より大きい数値が必要です。')
    return numeric


def load_dataset(path, *, target_col=TARGET_COLUMN_DEFAULT, feature_columns=None, external=False):
    frame, encoding = read_csv_auto(path)
    original_rows = np.arange(len(frame))
    dropped = []
    if external:
        if MW_COL not in frame:
            raise ValueError('外部検証CSVにMolWeightがありません。')
        # This is the notebook's only external row filter; do not mix into training.
        keep = frame[MW_COL].notna().to_numpy()
        dropped = original_rows[~keep].tolist()
        original_rows = original_rows[keep]
        frame = frame.loc[keep].reset_index(drop=True)
    if target_col not in frame:
        raise ValueError(f'評価・学習用CSVには目的変数 {target_col} が必要です。')
    if feature_columns is None:
        if frame.columns[-1] != target_col or len(frame.columns) <= 7:
            raise ValueError('Notebook形式では先頭6列が識別情報、最終列がLLNA_EC3である必要があります。')
        feature_columns = frame.columns[6:-1].tolist()
    X = validate_numeric_columns(frame, list(feature_columns))
    target = validate_numeric_columns(frame, [target_col], require_mw=False)[target_col]
    if target.isna().any() or (target <= 0).any():
        raise ValueError('LLNA_EC3は全行で有限かつ0より大きい数値が必要です。')
    return {'frame': frame, 'X': X, 'ec3': target.to_numpy(),
            'mw': X[MW_COL].to_numpy(), 'encoding': encoding,
            'source_rows': original_rows, 'dropped_missing_mw_rows': dropped,
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(value, path):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
