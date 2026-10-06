import csv
import io
import random

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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

        for row in reader:
            row_count += 1
            if row_count > MAX_ROWS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Dataset exceeds the {MAX_ROWS:,} row limit. Please upload a smaller CSV file.",
                )

            record = row_to_record(columns, row)
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

        return {
            "filename": filename,
            "file_size_bytes": file.size,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
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
