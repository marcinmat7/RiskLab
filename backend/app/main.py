import random
from datetime import datetime

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.calibration import run_calibration
from app.dataset_io import MAX_ROWS, register_dataset
from app.discrimination import run_discrimination
from app.eda import run_eda
from app.stability import run_stability

app = FastAPI(title="RiskLab API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LARGE_DATASET_WARNING_ROWS = 1_000
PREVIEW_ROWS = 10
RANDOM_PREVIEW_ROWS = 10


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "risklab-api"}


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


@app.post("/datasets/upload")
@app.post("/datasets/preview")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, object]:
    dataset = await register_dataset(file)
    columns = dataset["columns"]
    rows = dataset["rows"]
    row_count = len(rows)

    observed_types: dict[str, set[str]] = {column: set() for column in columns}
    for row in rows:
        for column, value in row.items():
            detected_type = detect_value_type(value)
            if detected_type is not None:
                observed_types[column].add(detected_type)

    first_rows = rows[:PREVIEW_ROWS]
    remaining_indices = list(range(PREVIEW_ROWS, row_count))
    sample_count = min(RANDOM_PREVIEW_ROWS, len(remaining_indices))
    sampled_indices = sorted(random.sample(remaining_indices, sample_count)) if sample_count else []

    first_preview = [
        {"row_number": index + 1, "values": row}
        for index, row in enumerate(first_rows)
    ]
    random_preview = [
        {"row_number": index + 1, "values": rows[index]}
        for index in sampled_indices
    ]

    warning = None
    if row_count > LARGE_DATASET_WARNING_ROWS:
        warning = (
            f"Large dataset: {row_count:,} rows. Files above {LARGE_DATASET_WARNING_ROWS:,} rows "
            "may take longer to process."
        )

    return {
        "dataset_id": dataset["dataset_id"],
        "filename": dataset["filename"],
        "file_format": dataset["format"],
        "file_size_bytes": dataset["file_size_bytes"],
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "column_types": {
            column: resolve_column_type(observed_types[column])
            for column in columns
        },
        "first_preview": first_preview,
        "random_preview": random_preview,
        "delimiter": dataset["delimiter"] or None,
        "warning": warning,
        "max_rows": MAX_ROWS,
    }


@app.post("/eda/run")
async def eda_run(
    dataset_id: str = Form(...),
    config: str = Form(...),
    time_granularity: str = Form("monthly"),
) -> dict[str, object]:
    return await run_eda(
        dataset_id=dataset_id,
        config=config,
        time_granularity=time_granularity,
    )


@app.post("/discrimination/run")
async def discrimination_run(
    dataset_id: str = Form(...),
    config: str = Form(...),
    time_granularity: str = Form("monthly"),
) -> dict[str, object]:
    return await run_discrimination(
        dataset_id=dataset_id,
        config=config,
        time_granularity=time_granularity,
    )


@app.post("/calibration/run")
async def calibration_run(
    dataset_id: str = Form(...),
    config: str = Form(...),
) -> dict[str, object]:
    return await run_calibration(dataset_id=dataset_id, config=config)


@app.post("/stability/run")
async def stability_run(
    dataset_id: str = Form(...),
    config: str = Form(...),
) -> dict[str, object]:
    return await run_stability(dataset_id=dataset_id, config=config)
