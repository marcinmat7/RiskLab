import csv
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile

MAX_CATEGORICAL_GROUPS = 20
CALIBRATION_BINS = 10


def _to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def _to_datetime(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None
    raw = value.strip()
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None


def _time_bucket(dt: datetime, granularity: str) -> str:
    if granularity == "daily":
        return dt.strftime("%Y-%m-%d")
    if granularity == "weekly":
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"
    if granularity == "quarterly":
        return f"{dt.year}-Q{((dt.month - 1) // 3) + 1}"
    if granularity == "yearly":
        return str(dt.year)
    return dt.strftime("%Y-%m")


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _spiegelhalter(rows: list[tuple[float, int]]) -> tuple[float | None, float | None]:
    numerator = 0.0
    variance = 0.0
    for pd, target in rows:
        weight = 1.0 - 2.0 * pd
        numerator += (target - pd) * weight
        variance += weight * weight * pd * (1.0 - pd)
    if variance <= 0:
        return None, None
    z_score = numerator / math.sqrt(variance)
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z_score)))
    return z_score, max(0.0, min(1.0, p_value))


def _wilson_interval(defaults: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    phat = defaults / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _summary(rows: list[tuple[float, int]]) -> dict[str, Any]:
    observations = len(rows)
    defaults = sum(target for _, target in rows)
    expected_defaults = sum(pd for pd, _ in rows)
    mean_pd = expected_defaults / observations if observations else 0.0
    default_rate = defaults / observations if observations else 0.0
    brier = sum((target - pd) ** 2 for pd, target in rows) / observations if observations else None
    spiegelhalter_z, spiegelhalter_p = _spiegelhalter(rows)
    return {
        "observations": observations,
        "defaults": defaults,
        "default_rate": default_rate,
        "mean_pd": mean_pd,
        "expected_defaults": expected_defaults,
        "oe_ratio": defaults / expected_defaults if expected_defaults > 0 else None,
        "brier_score": brier,
        "spiegelhalter_z": spiegelhalter_z,
        "spiegelhalter_p_value": spiegelhalter_p,
    }


def _calibration_bins(rows: list[tuple[float, int]], bins: int = CALIBRATION_BINS) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: item[0])
    total = len(ordered)
    result: list[dict[str, Any]] = []
    for index in range(bins):
        start = round(index * total / bins)
        end = round((index + 1) * total / bins)
        chunk = ordered[start:end]
        if not chunk:
            continue
        defaults = sum(target for _, target in chunk)
        mean_pd = sum(pd for pd, _ in chunk) / len(chunk)
        observed = defaults / len(chunk)
        lower, upper = _wilson_interval(defaults, len(chunk))
        result.append({
            "bin": index + 1,
            "observations": len(chunk),
            "defaults": defaults,
            "mean_pd": mean_pd,
            "observed_default_rate": observed,
            "ci_lower": lower,
            "ci_upper": upper,
        })
    return result


def _build_dimension_labels(
    records: list[dict[str, Any]],
    column: str,
    semantic_type: str,
) -> tuple[dict[int, str | None], dict[str, Any]]:
    labels: dict[int, str | None] = {}

    if semantic_type in {"continuous", "ordinal"}:
        values = [
            value for record in records
            if (value := _to_float(record["row"].get(column))) is not None
        ]
        if not values:
            return labels, {"mode": "quantiles", "bins": []}

        q1 = _quantile(values, 1 / 3)
        q2 = _quantile(values, 2 / 3)
        bin_labels = [
            f"Q1 · ≤ {q1:.4g}",
            f"Q2 · {q1:.4g}–{q2:.4g}",
            f"Q3 · > {q2:.4g}",
        ]
        for index, record in enumerate(records):
            value = _to_float(record["row"].get(column))
            if value is None:
                labels[index] = None
            elif value <= q1:
                labels[index] = bin_labels[0]
            elif value <= q2:
                labels[index] = bin_labels[1]
            else:
                labels[index] = bin_labels[2]
        return labels, {"mode": "quantiles", "bins": bin_labels}

    raw_values = [
        str(record["row"].get(column) or "").strip()
        for record in records
        if str(record["row"].get(column) or "").strip()
    ]
    counts = Counter(raw_values)
    keep = {value for value, _ in counts.most_common(MAX_CATEGORICAL_GROUPS)}
    use_other = len(counts) > MAX_CATEGORICAL_GROUPS
    for index, record in enumerate(records):
        value = str(record["row"].get(column) or "").strip()
        if not value:
            labels[index] = None
        elif use_other and value not in keep:
            labels[index] = "Other"
        else:
            labels[index] = value
    return labels, {
        "mode": "categories",
        "bins": [value for value, _ in counts.most_common(MAX_CATEGORICAL_GROUPS)] + (["Other"] if use_other else []),
    }


def _segment_summary(
    records: list[dict[str, Any]],
    name: str,
    columns: list[str],
    label_maps: dict[str, dict[int, str | None]],
) -> dict[str, Any]:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, record in enumerate(records):
        parts = [label_maps[column].get(index) for column in columns]
        if any(part is None for part in parts):
            continue
        label = " × ".join(str(part) for part in parts)
        groups[label].append(record["pair"])

    return {
        "key": " x ".join(columns),
        "name": name,
        "columns": columns,
        "groups": [
            {"value": label, **_summary(rows)}
            for label, rows in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
        ],
    }


def _time_summary(
    records: list[dict[str, Any]],
    granularity: str,
    membership: set[int] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, record in enumerate(records):
        if membership is not None and index not in membership:
            continue
        parsed = record["date"]
        if parsed is None:
            continue
        buckets[_time_bucket(parsed, granularity)].append(record["pair"])

    return [
        {"bucket": bucket, **_summary(rows)}
        for bucket, rows in sorted(buckets.items())
    ]


async def run_calibration(file: UploadFile = File(...), config: str = Form(...)) -> dict[str, Any]:
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid calibration configuration.") from exc

    if cfg.get("predictionType") != "pd":
        raise HTTPException(status_code=400, detail="Calibration analysis requires Probability of Default (PD) output.")

    prediction_column = cfg.get("predictionColumn", "")
    target_column = cfg.get("targetColumn", "")
    positive_class = str(cfg.get("positiveClass", ""))
    time_column = cfg.get("timeColumn", "")
    population_columns = list(cfg.get("populationColumns", []))
    sensitive_columns = list(cfg.get("sensitiveColumns", []))
    semantic_types: dict[str, str] = cfg.get("semanticTypes", {})
    time_cutoff_date = str(cfg.get("timeCutoffDate", "") or "").strip()
    time_cutoff_as_segment = bool(cfg.get("timeCutoffAsSegment", False))
    before_label = str(cfg.get("timeCutoffBeforeLabel", "Pre cut-off") or "Pre cut-off").strip()
    after_label = str(cfg.get("timeCutoffAfterLabel", "Post cut-off") or "Post cut-off").strip()

    if not prediction_column or not target_column or not positive_class:
        raise HTTPException(status_code=400, detail="Prediction, target and positive class are required.")

    cutoff_datetime = _to_datetime(time_cutoff_date) if time_cutoff_date else None
    if time_cutoff_date and cutoff_datetime is None:
        raise HTTPException(status_code=400, detail="Time cut-off must be a valid date.")

    segment_columns: list[str] = []
    for column in [*population_columns, *sensitive_columns]:
        if column and column not in segment_columns and column not in {prediction_column, target_column, time_column}:
            segment_columns.append(column)

    await file.seek(0)
    sample = await file.read(8192)
    if not sample:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
    try:
        sample_text = sample.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="The CSV file must be UTF-8 encoded.") from exc

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    await file.seek(0)
    stream = io.TextIOWrapper(file.file, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(stream, dialect=dialect)
    if not reader.fieldnames:
        stream.detach()
        raise HTTPException(status_code=400, detail="The CSV file must contain a header row.")

    records: list[dict[str, Any]] = []
    excluded_missing_prediction = 0
    excluded_missing_target = 0
    excluded_invalid_pd = 0

    try:
        for row in reader:
            pd = _to_float(row.get(prediction_column))
            if pd is None:
                excluded_missing_prediction += 1
                continue
            if pd < 0.0 or pd > 1.0:
                excluded_invalid_pd += 1
                continue

            raw_target = row.get(target_column)
            if raw_target is None or raw_target.strip() == "":
                excluded_missing_target += 1
                continue

            target = 1 if str(raw_target).strip() == positive_class else 0
            records.append({
                "pair": (pd, target),
                "row": row,
                "date": _to_datetime(row.get(time_column)) if time_column else None,
            })
    finally:
        stream.detach()

    pairs = [record["pair"] for record in records]
    if not pairs:
        raise HTTPException(status_code=400, detail="No usable rows for calibration analysis.")

    overall = _summary(pairs)
    overall["calibration_bins"] = _calibration_bins(pairs)

    label_maps: dict[str, dict[int, str | None]] = {}
    dimensions: list[dict[str, Any]] = []
    for column in segment_columns:
        labels, meta = _build_dimension_labels(records, column, semantic_types.get(column, "categorical"))
        label_maps[column] = labels
        dimensions.append({
            "column": column,
            "display_name": column,
            "semantic_type": semantic_types.get(column, "categorical"),
            **meta,
        })

    cutoff_key = "__time_cutoff__"
    cutoff_enabled = bool(time_column and cutoff_datetime and time_cutoff_as_segment)
    if cutoff_enabled and cutoff_datetime is not None:
        cutoff_date = cutoff_datetime.date()
        cutoff_labels: dict[int, str | None] = {}
        for index, record in enumerate(records):
            parsed = record["date"]
            if parsed is None:
                cutoff_labels[index] = None
            elif parsed.date() < cutoff_date:
                cutoff_labels[index] = before_label
            else:
                cutoff_labels[index] = after_label
        label_maps[cutoff_key] = cutoff_labels
        dimensions.append({
            "column": cutoff_key,
            "display_name": "Time cut-off",
            "semantic_type": "categorical",
            "mode": "categories",
            "bins": [before_label, after_label],
        })

    segments = [
        _segment_summary(records, column, [column], label_maps)
        for column in segment_columns
    ]
    segments.extend(
        _segment_summary(records, f"{left} × {right}", [left, right], label_maps)
        for left, right in combinations(segment_columns, 2)
    )
    if cutoff_enabled:
        segments.append(_segment_summary(records, "Time cut-off", [cutoff_key], label_maps))
        segments.extend(
            _segment_summary(records, f"{column} × Time cut-off", [column, cutoff_key], label_maps)
            for column in segment_columns
        )

    time_performance = None
    if time_column:
        granularities: dict[str, Any] = {}
        for granularity in ("daily", "weekly", "monthly", "quarterly", "yearly"):
            segment_series = []
            for segment in segments:
                groups: dict[str, set[int]] = defaultdict(set)
                for index in range(len(records)):
                    parts = [label_maps[column].get(index) for column in segment["columns"]]
                    if any(part is None for part in parts):
                        continue
                    label = " × ".join(str(part) for part in parts)
                    groups[label].add(index)
                segment_series.append({
                    "key": segment["key"],
                    "name": segment["name"],
                    "groups": [
                        {"value": label, "buckets": _time_summary(records, granularity, membership)}
                        for label, membership in groups.items()
                    ],
                })
            granularities[granularity] = {
                "overall": _time_summary(records, granularity),
                "segments": segment_series,
            }
        time_performance = {"time_column": time_column, "granularities": granularities}

    return {
        "overall": overall,
        "segment_dimensions": dimensions,
        "segment_performance": segments,
        "time_performance": time_performance,
        "excluded": {
            "missing_prediction": excluded_missing_prediction,
            "missing_target": excluded_missing_target,
            "invalid_pd": excluded_invalid_pd,
        },
    }
