import csv
import io

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "risklab-api"}


@app.post("/datasets/preview")
async def preview_dataset(file: UploadFile = File(...)) -> dict[str, object]:
    filename = file.filename or "dataset.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported in RiskLab v0.1.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="The CSV file must be UTF-8 encoded.",
        ) from exc

    try:
        sample = text[:8192]
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(text), dialect=dialect)
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="The uploaded CSV file contains no rows.")

    columns = [column.strip() for column in rows[0]]
    if not columns or all(not column for column in columns):
        raise HTTPException(status_code=400, detail="The CSV file must contain a header row.")

    if len(set(columns)) != len(columns):
        raise HTTPException(status_code=400, detail="Duplicate column names are not supported.")

    if len(columns) < 2:
        raise HTTPException(status_code=400, detail="The CSV file must contain at least two columns.")

    data_rows = rows[1:]
    preview_rows: list[dict[str, str | None]] = []

    for row in data_rows[:10]:
        padded = row + [""] * max(0, len(columns) - len(row))
        preview_rows.append(
            {
                column: (padded[index] if index < len(padded) else "")
                for index, column in enumerate(columns)
            }
        )

    return {
        "filename": filename,
        "row_count": len(data_rows),
        "column_count": len(columns),
        "columns": columns,
        "preview": preview_rows,
        "delimiter": dialect.delimiter,
    }
