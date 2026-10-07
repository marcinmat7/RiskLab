import csv
import io
import random
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.calibration import run_calibration
from app.discrimination import run_discrimination
from app.eda import run_eda

app = FastAPI(title="RiskLab API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_ROWS = 1_000_000
LARGE_DATASET_WARNING_ROWS = 1_000
PREVIEW_ROWS = 10
RANDOM_PREVIEW_ROWS = 10


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "risklab-api"}


def row_to_record(columns: list[str], row: list[str]) -> dict[str, str]:
    padded = row + [""] * max(0, len(columns) - len(row))
    return {column: padded[index] if index < len(padded) else "" for index, column in enumerate(columns)}


def detect_value_type(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if lowered in {"true", "false", "yes", "no", "y", "n"}:
        return "boolean"

    try:
        int(stripped)
        return "integer"
    except ValueError:
        pass

    try:
        float(stripped)
        return "numeric"
    except ValueError:
        pass

    try:
        datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        return "datetime"
    except ValueError:
        return "string"


def resolve_column_type(observed_types: set[str]) -> str:
    if not observed_types:
        return "empty"
    if observed_types <= {"integer"}:
        return "integer"
    if observed_types <= {"integer", "numeric"}:
        return "numeric"
    if observed_types <= {"boolean"}:
        return "boolean"
    if observed_types <= {"datetime"}:
        return "datetime"
    return "string"


@app.post("/datasets/preview")
async def preview_dataset(file: UploadFile = File(...)) -> dict[str, object]:
    filename = file.filename or "dataset.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported in RiskLab v0.1.")

    await file.seek(0)
    sample_bytes = await file.read(8192)
    if not sample_bytes:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    try:
        sample = sample_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="The CSV file must be UTF-8 encoded.") from exc

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    await file.seek(0)
    text_stream = io.TextIOWrapper(file.file, encoding="utf-8-sig", newline="")
    reader = csv.reader(text_stream, dialect=dialect)

    try:
        header = next(reader, None)
        if header is None:
            raise HTTPException(status_code=400, detail="The uploaded CSV file contains no rows.")

        columns = [column.strip() for column in header]
        if not columns or all(not column for column in columns):
            raise HTTPException(status_code=400, detail="The CSV file must contain a header row.")
        if len(set(columns)) != len(columns):
            raise HTTPException(status_code=400, detail="Duplicate column names are not supported.")
        if len(columns) < 2:
            raise HTTPException(status_code=400, detail="The CSV file must contain at least two columns.")

        row_count = 0
        first_preview: list[dict[str, object]] = []
        random_preview: list[dict[str, object]] = []
        reservoir_seen = 0
        observed_types: dict[str, set[str]] = {column: set() for column in columns}

        for row in reader:
            row_count += 1
            if row_count > MAX_ROWS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Dataset exceeds the {MAX_ROWS:,} row limit. Please upload a smaller CSV file.",
                )

            record = row_to_record(columns, row)
            for column, value in record.items():
                detected_type = detect_value_type(value)
                if detected_type is not None:
                    observed_types[column].add(detected_type)

            item = {"row_number": row_count, "values": record}

            if row_count <= PREVIEW_ROWS:
                first_preview.append(item)
                continue

            reservoir_seen += 1
            if len(random_preview) < RANDOM_PREVIEW_ROWS:
                random_preview.append(item)
            else:
                replacement_index = random.randint(1, reservoir_seen)
                if replacement_index <= RANDOM_PREVIEW_ROWS:
                    random_preview[replacement_index - 1] = item

        if row_count == 0:
            raise HTTPException(status_code=400, detail="The CSV file contains a header but no data rows.")

        random_preview.sort(key=lambda item: int(item["row_number"]))
        warning = None
        if row_count > LARGE_DATASET_WARNING_ROWS:
            warning = (
                f"Large dataset: {row_count:,} rows. Files above {LARGE_DATASET_WARNING_ROWS:,} rows "
                "may take longer to process."
            )

        column_types = {column: resolve_column_type(observed_types[column]) for column in columns}

        return {
            "filename": filename,
            "file_size_bytes": file.size,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
            "column_types": column_types,
            "first_preview": first_preview,
            "random_preview": random_preview,
            "delimiter": dialect.delimiter,
            "warning": warning,
            "max_rows": MAX_ROWS,
        }
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="The CSV file must be UTF-8 encoded.") from exc
    finally:
        text_stream.detach()


@app.post("/eda/run")
async def eda_run(file: UploadFile = File(...), config: str = Form(...), time_granularity: str = Form("monthly")) -> dict[str, object]:
    return await run_eda(file=file, config=config, time_granularity=time_granularity)


@app.post("/discrimination/run")
async def discrimination_run(
    file: UploadFile = File(...),
    config: str = Form(...),
    time_granularity: str = Form("monthly"),
) -> dict[str, object]:
    return await run_discrimination(
        file=file,
        config=config,
        time_granularity=time_granularity,
    )


@app.post("/calibration/run")
async def calibration_run(
    file: UploadFile = File(...),
    config: str = Form(...),
) -> dict[str, object]:
    return await run_calibration(file=file, config=config)
