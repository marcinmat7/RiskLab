import json
import math
from datetime import datetime
from typing import Any

from fastapi import Form, HTTPException

from app.dataset_io import get_dataset

MISSING_VALUES = {"", "na", "n/a", "null", "none", "nan"}
BOOLEAN_VALUES = {"true", "false", "yes", "no", "y", "n", "1", "0"}


def _is_missing(value: str | None) -> bool:
    return value is None or str(value).strip().lower() in MISSING_VALUES


def _parse_number(value: str) -> bool:
    try:
        number = float(value)
        return math.isfinite(number)
    except ValueError:
        return False


def _parse_datetime(value: str) -> bool:
    raw = value.strip()
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            datetime.fromisoformat(candidate)
            return True
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            datetime.strptime(raw, fmt)
            return True
        except ValueError:
            pass
    return False


def _physical_type(values: list[str]) -> str:
    usable = [str(value).strip() for value in values if not _is_missing(value)]
    if not usable:
        return "empty"

    lowered = [value.lower() for value in usable]
    if all(value in {"true", "false", "yes", "no", "y", "n"} for value in lowered):
        return "boolean"

    integer = True
    for value in usable:
        try:
            int(value)
        except ValueError:
            integer = False
            break
    if integer:
        return "integer"

    if all(_parse_number(value) for value in usable):
        return "numeric"

    if all(_parse_datetime(value) for value in usable):
        return "datetime"

    return "string"


def _fallback_semantic(physical_type: str) -> str:
    if physical_type in {"integer", "numeric"}:
        return "continuous"
    if physical_type == "boolean":
        return "boolean"
    if physical_type == "datetime":
        return "datetime"
    if physical_type == "empty":
        return "ignore"
    return "categorical"


def _can_parse(value: str, semantic_type: str) -> bool:
    if semantic_type in {"continuous", "ordinal"}:
        return _parse_number(value)
    if semantic_type == "datetime":
        return _parse_datetime(value)
    if semantic_type == "boolean":
        return value.strip().lower() in BOOLEAN_VALUES
    return True


async def run_validation(dataset_id: str = Form(...), config: str = Form(...)) -> dict[str, Any]:
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid validation configuration.") from exc

    dataset = get_dataset(dataset_id)
    columns: list[str] = list(dataset["columns"])
    rows: list[dict[str, str]] = dataset["rows"]
    configured_types: dict[str, str] = cfg.get("semanticTypes", {})

    effective_types: dict[str, str] = {}
    column_results: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for column in columns:
        values = [row.get(column, "") for row in rows]
        detected_physical_type = _physical_type(values)
        fallback_type = _fallback_semantic(detected_physical_type)
        requested_type = configured_types.get(column, fallback_type)

        invalid_values: list[str] = []
        invalid_count = 0
        non_missing_count = 0

        for raw in values:
            if _is_missing(raw):
                continue
            non_missing_count += 1
            value = str(raw).strip()
            if not _can_parse(value, requested_type):
                invalid_count += 1
                if len(invalid_values) < 5 and value not in invalid_values:
                    invalid_values.append(value)

        effective_type = requested_type if invalid_count == 0 else fallback_type
        effective_types[column] = effective_type

        result = {
            "column": column,
            "detected_physical_type": detected_physical_type,
            "requested_semantic_type": requested_type,
            "effective_semantic_type": effective_type,
            "non_missing_count": non_missing_count,
            "invalid_count": invalid_count,
            "invalid_examples": invalid_values,
            "status": "warning" if invalid_count else "ok",
        }
        column_results.append(result)

        if invalid_count:
            warnings.append({
                **result,
                "message": (
                    f'Column "{column}" could not be parsed as {requested_type} for '
                    f"{invalid_count:,} non-missing value(s). RiskLab will use "
                    f"{effective_type}, consistent with detected physical type "
                    f"{detected_physical_type}."
                ),
            })

    return {
        "status": "warning" if warnings else "ok",
        "effective_semantic_types": effective_types,
        "column_results": column_results,
        "warnings": warnings,
    }
