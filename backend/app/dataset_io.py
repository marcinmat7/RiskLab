import csv
import io
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as parquet
from fastapi import HTTPException, UploadFile

MAX_ROWS = 1_000_000
SUPPORTED_EXTENSIONS = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".parquet": "Parquet",
    ".feather": "Feather",
}
CSV_DELIMITERS = ",;\t|"
DATA_DIR = Path(__file__).resolve().parents[1] / ".risklab_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_DATASET_CACHE: dict[str, dict[str, Any]] = {}


def supported_extensions_label() -> str:
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _validate_columns(columns: list[str]) -> list[str]:
    cleaned = [str(column).strip() for column in columns]
    if not cleaned or all(not column for column in cleaned):
        raise HTTPException(status_code=400, detail="The dataset must contain column names.")
    if any(not column for column in cleaned):
        raise HTTPException(status_code=400, detail="Empty column names are not supported.")
    if len(set(cleaned)) != len(cleaned):
        raise HTTPException(status_code=400, detail="Duplicate column names are not supported.")
    if len(cleaned) < 2:
        raise HTTPException(status_code=400, detail="The dataset must contain at least two columns.")
    return cleaned


async def _read_bytes(file: UploadFile) -> bytes:
    await file.seek(0)
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return payload


def _read_delimited(payload: bytes, extension: str) -> tuple[list[str], list[dict[str, str]], str]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV and TSV files must be UTF-8 encoded.") from exc

    if extension == ".tsv":
        delimiter = "\t"
    else:
        sample = text[:8192]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS).delimiter
        except csv.Error:
            delimiter = ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="The dataset must contain a header row.")

    raw_columns = list(reader.fieldnames)
    columns = _validate_columns(raw_columns)
    rename = dict(zip(raw_columns, columns))
    rows: list[dict[str, str]] = []
    for raw in reader:
        rows.append({rename[raw_column]: _stringify(raw.get(raw_column)) for raw_column in raw_columns})
        if len(rows) > MAX_ROWS:
            raise HTTPException(
                status_code=413,
                detail=f"Dataset exceeds the {MAX_ROWS:,} row limit.",
            )
    return columns, rows, delimiter


def _table_to_rows(table: pa.Table) -> tuple[list[str], list[dict[str, str]]]:
    raw_columns = list(table.column_names)
    columns = _validate_columns(raw_columns)
    rename = dict(zip(raw_columns, columns))
    if table.num_rows > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Dataset exceeds the {MAX_ROWS:,} row limit.",
        )
    rows = [
        {rename[raw_column]: _stringify(row.get(raw_column)) for raw_column in raw_columns}
        for row in table.to_pylist()
    ]
    return columns, rows


def _read_parquet(payload: bytes) -> tuple[list[str], list[dict[str, str]], str]:
    try:
        table = parquet.read_table(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read Parquet file: {exc}") from exc
    columns, rows = _table_to_rows(table)
    return columns, rows, ""


def _read_feather(payload: bytes) -> tuple[list[str], list[dict[str, str]], str]:
    try:
        table = feather.read_table(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read Feather file: {exc}") from exc
    columns, rows = _table_to_rows(table)
    return columns, rows, ""


def _dataset_paths(dataset_id: str) -> tuple[Path, Path]:
    return DATA_DIR / f"{dataset_id}.feather", DATA_DIR / f"{dataset_id}.json"


def _persist_dataset(dataset: dict[str, Any]) -> None:
    data_path, meta_path = _dataset_paths(dataset["dataset_id"])
    table = pa.Table.from_pylist(dataset["rows"])
    feather.write_feather(table, data_path)
    metadata = {key: value for key, value in dataset.items() if key != "rows"}
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_persisted_dataset(dataset_id: str) -> dict[str, Any]:
    data_path, meta_path = _dataset_paths(dataset_id)
    if not data_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Upload the dataset again.")
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        table = feather.read_table(data_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stored dataset could not be loaded: {exc}") from exc
    rows = [
        {column: _stringify(row.get(column)) for column in metadata["columns"]}
        for row in table.to_pylist()
    ]
    dataset = {**metadata, "rows": rows}
    _DATASET_CACHE[dataset_id] = dataset
    return dataset


async def register_dataset(file: UploadFile) -> dict[str, Any]:
    filename = file.filename or "dataset"
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension or '(none)'}'. Supported: {supported_extensions_label()}.",
        )

    payload = await _read_bytes(file)
    if extension in {".csv", ".tsv"}:
        columns, rows, delimiter = _read_delimited(payload, extension)
    elif extension == ".parquet":
        columns, rows, delimiter = _read_parquet(payload)
    else:
        columns, rows, delimiter = _read_feather(payload)

    if not rows:
        raise HTTPException(status_code=400, detail="The dataset contains column names but no data rows.")

    dataset_id = uuid4().hex
    dataset = {
        "dataset_id": dataset_id,
        "filename": filename,
        "extension": extension,
        "format": SUPPORTED_EXTENSIONS[extension],
        "delimiter": delimiter,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "file_size_bytes": len(payload),
    }
    _DATASET_CACHE[dataset_id] = dataset
    _persist_dataset(dataset)
    return dataset


def get_dataset(dataset_id: str) -> dict[str, Any]:
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required.")
    return _DATASET_CACHE.get(dataset_id) or _load_persisted_dataset(dataset_id)
