import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any

from fastapi import Form, HTTPException

from app.dataset_io import get_dataset

MAX_CATEGORICAL_GROUPS = 20
QUANTILE_GROUPS = 3


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


def _roc_points(rows: list[tuple[float, int]]) -> tuple[list[dict[str, float]], float, float]:
    positives = sum(target for _, target in rows)
    negatives = len(rows) - positives
    if not rows or positives == 0 or negatives == 0:
        return [], 0.0, 0.0

    ordered = sorted(rows, key=lambda item: item[0], reverse=True)
    tp = fp = 0
    points = [{"fpr": 0.0, "tpr": 0.0, "threshold": ordered[0][0] + 1e-12}]
    ks = 0.0
    previous_score: float | None = None

    for score, target in ordered:
        if previous_score is not None and score != previous_score:
            tpr = tp / positives
            fpr = fp / negatives
            points.append({"fpr": fpr, "tpr": tpr, "threshold": previous_score})
            ks = max(ks, abs(tpr - fpr))
        if target:
            tp += 1
        else:
            fp += 1
        previous_score = score

    points.append({"fpr": 1.0, "tpr": 1.0, "threshold": ordered[-1][0]})
    auc = sum(
        (right["fpr"] - left["fpr"]) * (right["tpr"] + left["tpr"]) / 2
        for left, right in zip(points, points[1:])
    )
    return points, auc, ks


def _cap_points(rows: list[tuple[float, int]]) -> list[dict[str, float]]:
    ordered = sorted(rows, key=lambda item: item[0], reverse=True)
    total = len(ordered)
    total_bad = sum(target for _, target in ordered)
    if total == 0 or total_bad == 0:
        return []
    result = [{"population": 0.0, "bad_capture": 0.0}]
    captured = 0
    step = max(1, total // 100)
    for index, (_, target) in enumerate(ordered, start=1):
        captured += target
        if index % step == 0 or index == total:
            result.append({"population": index / total, "bad_capture": captured / total_bad})
    return result


def _metrics(rows: list[tuple[float, int]], include_curves: bool = True) -> dict[str, Any]:
    points, auc, ks = _roc_points(rows)
    ordered = sorted(rows, key=lambda item: item[0], reverse=True)
    top_count = max(1, math.ceil(len(ordered) * 0.10))
    total_bad = sum(target for _, target in ordered)
    captured = sum(target for _, target in ordered[:top_count])
    return {
        "observations": len(rows),
        "defaults": total_bad,
        "default_rate": total_bad / len(rows) if rows else 0.0,
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks,
        "bad_capture_10": captured / total_bad if total_bad else 0.0,
        "roc": points if include_curves else [],
        "cap": _cap_points(rows) if include_curves else [],
    }


def _safe_metrics(rows: list[tuple[float, int]], include_curves: bool = True) -> dict[str, Any]:
    positives = sum(target for _, target in rows)
    if len(rows) < 2 or positives == 0 or positives == len(rows):
        return {
            "observations": len(rows),
            "defaults": positives,
            "default_rate": positives / len(rows) if rows else 0.0,
            "auc": None,
            "gini": None,
            "ks": None,
            "bad_capture_10": None,
            "roc": [],
            "cap": [],
        }
    return _metrics(rows, include_curves=include_curves)


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
        edges = [min(values), q1, q2, max(values)]
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
        return labels, {"mode": "quantiles", "bins": bin_labels, "edges": edges}

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


def _segment_result(
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

    group_results = []
    for label, rows in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        metrics = _safe_metrics(rows, include_curves=True)
        group_results.append({"value": label, **metrics})

    return {"key": " x ".join(columns), "name": name, "columns": columns, "groups": group_results}


def _time_series(
    records: list[dict[str, Any]],
    granularity: str,
    group_labels: dict[int, str | None] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, record in enumerate(records):
        parsed = record["date"]
        if parsed is None:
            continue
        if group_labels is not None and group_labels.get(index) is None:
            continue
        buckets[_time_bucket(parsed, granularity)].append(record["pair"])

    return [
        {"bucket": bucket, **_safe_metrics(rows, include_curves=False)}
        for bucket, rows in sorted(buckets.items())
    ]


async def run_discrimination(
    dataset_id: str = Form(...),
    config: str = Form(...),
    time_granularity: str = Form("monthly"),
) -> dict[str, Any]:
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid discrimination configuration.") from exc

    prediction_column = cfg.get("predictionColumn", "")
    target_column = cfg.get("targetColumn", "")
    positive_class = str(cfg.get("positiveClass", ""))
    prediction_type = cfg.get("predictionType", "pd")
    score_direction = cfg.get("scoreDirection", "higher-risk")
    population_columns = list(cfg.get("populationColumns", []))
    sensitive_columns = list(cfg.get("sensitiveColumns", []))
    semantic_types: dict[str, str] = cfg.get("semanticTypes", {})
    time_column = cfg.get("timeColumn", "")
    time_cutoff_date = str(cfg.get("timeCutoffDate", "") or "").strip()
    time_cutoff_as_segment = bool(cfg.get("timeCutoffAsSegment", False))
    time_cutoff_before_label = str(cfg.get("timeCutoffBeforeLabel", "Pre cut-off") or "Pre cut-off").strip()
    time_cutoff_after_label = str(cfg.get("timeCutoffAfterLabel", "Post cut-off") or "Post cut-off").strip()

    if not prediction_column or not target_column or not positive_class:
        raise HTTPException(status_code=400, detail="Prediction, target and positive class are required.")

    segment_columns = []
    for column in [*population_columns, *sensitive_columns]:
        if column and column not in segment_columns and column not in {prediction_column, target_column, time_column}:
            segment_columns.append(column)

    cutoff_datetime = _to_datetime(time_cutoff_date) if time_cutoff_date else None
    if time_cutoff_date and cutoff_datetime is None:
        raise HTTPException(status_code=400, detail="Time cut-off must be a valid date.")
    cutoff_key = "__time_cutoff__"
    cutoff_enabled = bool(time_column and cutoff_datetime and time_cutoff_as_segment)

    dataset = get_dataset(dataset_id)
    rows = dataset["rows"]

    records: list[dict[str, Any]] = []
    excluded_missing_prediction = 0
    excluded_missing_target = 0

    for row in rows:
        prediction = _to_float(row.get(prediction_column))
        if prediction is None:
            excluded_missing_prediction += 1
            continue
        raw_target = row.get(target_column)
        if raw_target is None or raw_target.strip() == "":
            excluded_missing_target += 1
            continue

        target = 1 if str(raw_target).strip() == positive_class else 0
        risk_score = prediction
        if prediction_type == "score" and score_direction == "lower-risk":
            risk_score = -prediction

        records.append({
            "pair": (risk_score, target),
            "row": row,
            "date": _to_datetime(row.get(time_column)) if time_column else None,
        })

    usable = [record["pair"] for record in records]
    if len(usable) < 2:
        raise HTTPException(status_code=400, detail="Not enough usable rows for discrimination analysis.")
    defaults = sum(target for _, target in usable)
    if defaults == 0 or defaults == len(usable):
        raise HTTPException(status_code=400, detail="Target must contain both positive and negative classes.")

    overall = _metrics(usable)

    label_maps: dict[str, dict[int, str | None]] = {}
    dimension_meta = []
    for column in segment_columns:
        labels, meta = _build_dimension_labels(records, column, semantic_types.get(column, "categorical"))
        label_maps[column] = labels
        dimension_meta.append({
            "column": column,
            "semantic_type": semantic_types.get(column, "categorical"),
            **meta,
        })

    if cutoff_enabled and cutoff_datetime is not None:
        cutoff_labels: dict[int, str | None] = {}
        cutoff_date = cutoff_datetime.date()
        for index, record in enumerate(records):
            parsed = record["date"]
            if parsed is None:
                cutoff_labels[index] = None
            elif parsed.date() < cutoff_date:
                cutoff_labels[index] = time_cutoff_before_label
            else:
                cutoff_labels[index] = time_cutoff_after_label
        label_maps[cutoff_key] = cutoff_labels
        dimension_meta.append({
            "column": cutoff_key,
            "semantic_type": "categorical",
            "mode": "categories",
            "bins": [time_cutoff_before_label, time_cutoff_after_label],
            "display_name": "Time cut-off",
            "cutoff_date": time_cutoff_date,
        })

    segments = [
        _segment_result(records, column, [column], label_maps)
        for column in segment_columns
    ]
    segments.extend(
        _segment_result(records, f"{left} × {right}", [left, right], label_maps)
        for left, right in combinations(segment_columns, 2)
    )

    if cutoff_enabled:
        segments.append(_segment_result(records, "Time cut-off", [cutoff_key], label_maps))
        segments.extend(
            _segment_result(records, f"{column} × Time cut-off", [column, cutoff_key], label_maps)
            for column in segment_columns
        )

    granularities: dict[str, dict[str, Any]] = {}
    if time_column:
        for granularity in ("daily", "weekly", "monthly", "quarterly", "yearly"):
            overall_time = _time_series(records, granularity)
            segment_time = []
            for segment in segments:
                group_series = []
                columns = segment["columns"]
                groups: dict[str, dict[int, str | None]] = {}
                for index in range(len(records)):
                    parts = [label_maps[column].get(index) for column in columns]
                    if any(part is None for part in parts):
                        continue
                    label = " × ".join(str(part) for part in parts)
                    groups.setdefault(label, {})[index] = label

                for group_name, membership in groups.items():
                    group_labels = {
                        index: (group_name if index in membership else None)
                        for index in range(len(records))
                    }
                    group_series.append({
                        "value": group_name,
                        "buckets": _time_series(records, granularity, group_labels),
                    })
                segment_time.append({
                    "key": segment["key"],
                    "name": segment["name"],
                    "columns": columns,
                    "groups": group_series,
                })

            granularities[granularity] = {
                "overall": overall_time,
                "segments": segment_time,
            }

    return {
        "overall": overall,
        "segment_dimensions": dimension_meta,
        "segment_performance": segments,
        "time_performance": {
            "time_column": time_column,
            "granularities": granularities,
        } if time_column else None,
        "excluded": {
            "missing_prediction": excluded_missing_prediction,
            "missing_target": excluded_missing_target,
        },
        "direction": "higher-is-riskier",
        "time_cutoff": {
            "date": time_cutoff_date,
            "as_segment": cutoff_enabled,
            "before_label": time_cutoff_before_label,
            "after_label": time_cutoff_after_label,
        } if time_cutoff_date else None,
    }
