import csv
import io
import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile


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


def _time_bucket(dt: datetime, granularity: str = "monthly") -> str:
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


def _roc_points(rows: list[tuple[float, int]]) -> tuple[list[dict[str, float]], float, float]:
    if not rows:
        return [], 0.0, 0.0
    positives = sum(target for _, target in rows)
    negatives = len(rows) - positives
    if positives == 0 or negatives == 0:
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
    ks = max(ks, abs(1.0 - 1.0))

    auc = 0.0
    for left, right in zip(points, points[1:]):
        auc += (right["fpr"] - left["fpr"]) * (right["tpr"] + left["tpr"]) / 2
    return points, auc, ks


def _rank_table(rows: list[tuple[float, int]], bins: int = 10) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: item[0], reverse=True)
    total = len(ordered)
    total_bad = sum(target for _, target in ordered)
    result = []
    cumulative_bad = 0
    for index in range(bins):
        start = round(index * total / bins)
        end = round((index + 1) * total / bins)
        chunk = ordered[start:end]
        if not chunk:
            continue
        defaults = sum(target for _, target in chunk)
        cumulative_bad += defaults
        result.append({
            "bin": index + 1,
            "observations": len(chunk),
            "defaults": defaults,
            "default_rate": defaults / len(chunk),
            "mean_prediction": sum(score for score, _ in chunk) / len(chunk),
            "min_prediction": min(score for score, _ in chunk),
            "max_prediction": max(score for score, _ in chunk),
            "cumulative_population": end / total,
            "cumulative_bad_capture": cumulative_bad / total_bad if total_bad else 0.0,
        })
    return result


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
            result.append({
                "population": index / total,
                "bad_capture": captured / total_bad,
            })
    return result


def _metrics(rows: list[tuple[float, int]]) -> dict[str, Any]:
    points, auc, ks = _roc_points(rows)
    rank = _rank_table(rows)
    capture10 = 0.0
    if rank:
        capture10 = rank[0]["cumulative_bad_capture"]
    return {
        "observations": len(rows),
        "defaults": sum(target for _, target in rows),
        "default_rate": sum(target for _, target in rows) / len(rows) if rows else 0.0,
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks,
        "bad_capture_10": capture10,
        "roc": points,
        "rank_table": rank,
        "cap": _cap_points(rows),
    }


async def run_discrimination(
    file: UploadFile = File(...),
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
    population_columns = cfg.get("populationColumns", [])
    time_column = cfg.get("timeColumn", "")

    if not prediction_column or not target_column or not positive_class:
        raise HTTPException(status_code=400, detail="Prediction, target and positive class are required.")

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

    usable: list[tuple[float, int]] = []
    excluded_missing_prediction = 0
    excluded_missing_target = 0
    by_population: dict[str, dict[str, list[tuple[float, int]]]] = {
        column: defaultdict(list) for column in population_columns
    }
    by_time: dict[str, list[tuple[float, int]]] = defaultdict(list)

    try:
        for row in reader:
            raw_prediction = row.get(prediction_column)
            raw_target = row.get(target_column)
            prediction = _to_float(raw_prediction)
            if prediction is None:
                excluded_missing_prediction += 1
                continue
            if raw_target is None or raw_target.strip() == "":
                excluded_missing_target += 1
                continue

            target = 1 if str(raw_target).strip() == positive_class else 0
            risk_score = prediction
            if prediction_type == "score" and score_direction == "lower-risk":
                risk_score = -prediction

            pair = (risk_score, target)
            usable.append(pair)

            for column in population_columns:
                value = str(row.get(column) or "").strip()
                if value:
                    by_population[column][value].append(pair)

            if time_column:
                parsed = _to_datetime(row.get(time_column))
                if parsed:
                    by_time[_time_bucket(parsed, time_granularity)].append(pair)
    finally:
        stream.detach()

    if len(usable) < 2:
        raise HTTPException(status_code=400, detail="Not enough usable rows for discrimination analysis.")
    defaults = sum(target for _, target in usable)
    if defaults == 0 or defaults == len(usable):
        raise HTTPException(status_code=400, detail="Target must contain both positive and negative classes.")

    overall = _metrics(usable)

    populations = []
    for column, groups in by_population.items():
        group_results = []
        for value, rows in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
            if len(rows) < 2:
                continue
            positives = sum(target for _, target in rows)
            if positives == 0 or positives == len(rows):
                group_results.append({
                    "value": value,
                    "observations": len(rows),
                    "defaults": positives,
                    "default_rate": positives / len(rows),
                    "auc": None,
                    "gini": None,
                    "ks": None,
                })
                continue
            metrics = _metrics(rows)
            group_results.append({
                "value": value,
                "observations": metrics["observations"],
                "defaults": metrics["defaults"],
                "default_rate": metrics["default_rate"],
                "auc": metrics["auc"],
                "gini": metrics["gini"],
                "ks": metrics["ks"],
            })
        populations.append({"column": column, "groups": group_results})

    time_results = []
    for bucket, rows in sorted(by_time.items()):
        positives = sum(target for _, target in rows)
        if len(rows) < 2 or positives == 0 or positives == len(rows):
            time_results.append({
                "bucket": bucket,
                "observations": len(rows),
                "defaults": positives,
                "default_rate": positives / len(rows) if rows else 0.0,
                "auc": None,
                "gini": None,
                "ks": None,
            })
        else:
            metrics = _metrics(rows)
            time_results.append({
                "bucket": bucket,
                "observations": metrics["observations"],
                "defaults": metrics["defaults"],
                "default_rate": metrics["default_rate"],
                "auc": metrics["auc"],
                "gini": metrics["gini"],
                "ks": metrics["ks"],
            })

    return {
        "overall": overall,
        "population_performance": populations,
        "time_performance": {
            "time_column": time_column or None,
            "granularity": time_granularity,
            "buckets": time_results,
        } if time_column else None,
        "excluded": {
            "missing_prediction": excluded_missing_prediction,
            "missing_target": excluded_missing_target,
        },
        "direction": "higher-is-riskier",
    }
