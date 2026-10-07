import csv
import hashlib
import io
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile

EDA_SAMPLE_ROWS = 20_000
MAX_TRACKED_UNIQUES = 10_000
TOP_CATEGORIES = 50
MAX_RELATIONSHIP_COLUMNS = 80


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in {"", "na", "n/a", "null", "none", "nan"}


def _to_float(value: str | None) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(str(value).strip())
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _to_datetime(value: str | None) -> datetime | None:
    if _is_missing(value):
        return None
    raw = str(value).strip()
    candidates = [raw, raw.replace("Z", "+00:00")]
    for candidate in candidates:
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


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _histogram(values: list[float], bins: int = 12) -> list[dict[str, Any]]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [{"label": f"{low:.4g}", "count": len(values)}]
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1
    return [
        {
            "label": f"{low + i * width:.4g}–{low + (i + 1) * width:.4g}",
            "count": count,
        }
        for i, count in enumerate(counts)
    ]


def _empirical_cdf(values: list[float], points: int = 100) -> list[dict[str, float]]:
    if not values:
        return []
    ordered = sorted(values)
    if len(ordered) <= points:
        return [
            {"x": value, "cdf": (index + 1) / len(ordered)}
            for index, value in enumerate(ordered)
        ]
    result: list[dict[str, float]] = []
    for index in range(points):
        position = round(index * (len(ordered) - 1) / (points - 1))
        result.append({
            "x": ordered[position],
            "cdf": (position + 1) / len(ordered),
        })
    return result


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def _cramers_v(pairs: list[tuple[str, str]]) -> float | None:
    if len(pairs) < 3:
        return None
    row_values = sorted({a for a, _ in pairs})
    col_values = sorted({b for _, b in pairs})
    if len(row_values) < 2 or len(col_values) < 2:
        return None
    if len(row_values) > 25 or len(col_values) > 25:
        return None

    rows = {value: i for i, value in enumerate(row_values)}
    cols = {value: i for i, value in enumerate(col_values)}
    table = [[0 for _ in col_values] for _ in row_values]
    for a, b in pairs:
        table[rows[a]][cols[b]] += 1

    n = len(pairs)
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[i][j] for i in range(len(row_values))) for j in range(len(col_values))]
    chi2 = 0.0
    for i in range(len(row_values)):
        for j in range(len(col_values)):
            expected = row_totals[i] * col_totals[j] / n
            if expected > 0:
                chi2 += (table[i][j] - expected) ** 2 / expected

    denominator = min(len(row_values) - 1, len(col_values) - 1)
    if denominator <= 0:
        return None
    return math.sqrt((chi2 / n) / denominator)


def _numeric_categorical_effect(rows: list[dict[str, str]], numeric: str, categorical: str) -> float | None:
    groups: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for row in rows:
        number = _to_float(row.get(numeric))
        category = row.get(categorical, "").strip()
        if number is None or not category:
            continue
        groups[category].append(number)
        all_values.append(number)
    if len(groups) < 2 or len(all_values) < 5:
        return None
    if len(groups) > 25:
        return None

    overall_mean = statistics.fmean(all_values)
    total_ss = sum((value - overall_mean) ** 2 for value in all_values)
    if total_ss == 0:
        return None
    between_ss = sum(
        len(values) * (statistics.fmean(values) - overall_mean) ** 2
        for values in groups.values()
        if values
    )
    return math.sqrt(max(0.0, min(between_ss / total_ss, 1.0)))


def _missingness_association(rows: list[dict[str, str]], target: str, other: str, other_type: str) -> tuple[float, str] | None:
    usable = [row for row in rows if not _is_missing(row.get(other))]
    if len(usable) < 20:
        return None

    if other_type in {"continuous", "ordinal"}:
        missing_values = [_to_float(row.get(other)) for row in usable if _is_missing(row.get(target))]
        observed_values = [_to_float(row.get(other)) for row in usable if not _is_missing(row.get(target))]
        x = [v for v in missing_values if v is not None]
        y = [v for v in observed_values if v is not None]
        if len(x) < 5 or len(y) < 5:
            return None
        combined = x + y
        spread = statistics.pstdev(combined)
        if spread == 0:
            return None
        score = min(abs(statistics.fmean(x) - statistics.fmean(y)) / spread, 5.0)
        return score, f"standardized mean difference {score:.2f}"

    categories: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in usable:
        category = str(row.get(other, "")).strip()
        categories[category][1] += 1
        if _is_missing(row.get(target)):
            categories[category][0] += 1
    valid_rates = [missing / total for missing, total in categories.values() if total >= 5]
    if len(valid_rates) < 2:
        return None
    score = max(valid_rates) - min(valid_rates)
    return score, f"missing-rate range {score * 100:.1f} pp across groups"


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


async def run_eda(file: UploadFile = File(...), config: str = Form(...), time_granularity: str = Form("monthly")) -> dict[str, Any]:
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid EDA configuration.") from exc

    semantic_types: dict[str, str] = cfg.get("semanticTypes", {})
    population_columns: list[str] = cfg.get("populationColumns", [])
    time_column: str = cfg.get("timeColumn", "")

    await file.seek(0)
    sample_bytes = await file.read(8192)
    if not sample_bytes:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
    try:
        sample_text = sample_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="The CSV file must be UTF-8 encoded.") from exc

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    await file.seek(0)
    text_stream = io.TextIOWrapper(file.file, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(text_stream, dialect=dialect)
    if not reader.fieldnames:
        text_stream.detach()
        raise HTTPException(status_code=400, detail="The CSV file must contain a header row.")

    columns = [column.strip() for column in reader.fieldnames]
    row_count = 0
    missing_counts = Counter({column: 0 for column in columns})
    unique_values: dict[str, set[str]] = {column: set() for column in columns}
    unique_capped: set[str] = set()
    duplicate_hashes: set[bytes] = set()
    duplicate_count = 0
    sample_rows: list[dict[str, str]] = []

    numeric_count = Counter()
    numeric_sum = Counter()
    numeric_sum_sq = Counter()
    numeric_min: dict[str, float] = {}
    numeric_max: dict[str, float] = {}

    try:
        for raw in reader:
            row_count += 1
            row = {column: (raw.get(column) or "") for column in columns}

            digest = hashlib.blake2b(
                "\x1f".join(row.get(column, "") for column in columns).encode("utf-8"),
                digest_size=8,
            ).digest()
            if digest in duplicate_hashes:
                duplicate_count += 1
            else:
                duplicate_hashes.add(digest)

            for column in columns:
                value = row[column]
                if _is_missing(value):
                    missing_counts[column] += 1
                    continue

                if column not in unique_capped:
                    unique_values[column].add(value)
                    if len(unique_values[column]) > MAX_TRACKED_UNIQUES:
                        unique_capped.add(column)
                        unique_values[column].clear()

                if semantic_types.get(column) in {"continuous", "ordinal"}:
                    number = _to_float(value)
                    if number is not None:
                        numeric_count[column] += 1
                        numeric_sum[column] += number
                        numeric_sum_sq[column] += number * number
                        numeric_min[column] = min(numeric_min.get(column, number), number)
                        numeric_max[column] = max(numeric_max.get(column, number), number)

            if len(sample_rows) < EDA_SAMPLE_ROWS:
                sample_rows.append(row)
            else:
                replacement = random.randint(1, row_count)
                if replacement <= EDA_SAMPLE_ROWS:
                    sample_rows[replacement - 1] = row

        if row_count == 0:
            raise HTTPException(status_code=400, detail="The CSV file contains a header but no data rows.")
    finally:
        text_stream.detach()

    profiles: list[dict[str, Any]] = []
    for column in columns:
        semantic = semantic_types.get(column, "categorical")
        missing = missing_counts[column]
        non_missing = row_count - missing
        profile: dict[str, Any] = {
            "column": column,
            "semantic_type": semantic,
            "missing_count": missing,
            "missing_rate": missing / row_count,
            "sample_missing_count": sum(_is_missing(row.get(column)) for row in sample_rows),
            "unique_count": None if column in unique_capped else len(unique_values[column]),
            "unique_count_capped": column in unique_capped,
        }

        sample_non_missing = [row[column] for row in sample_rows if not _is_missing(row.get(column))]
        if semantic in {"continuous", "ordinal"}:
            values = [number for value in sample_non_missing if (number := _to_float(value)) is not None]
            count = numeric_count[column]
            if count:
                mean = numeric_sum[column] / count
                variance = max(numeric_sum_sq[column] / count - mean * mean, 0.0)
                profile["numeric_summary"] = {
                    "count": count,
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "min": numeric_min.get(column),
                    "p25": _quantile(values, 0.25),
                    "median": _quantile(values, 0.5),
                    "p75": _quantile(values, 0.75),
                    "max": numeric_max.get(column),
                    "histogram": _histogram(values),
                    "cdf": _empirical_cdf(values),
                }
        elif semantic not in {"identifier", "text", "ignore"}:
            counts = Counter(sample_non_missing)
            total = sum(counts.values())
            top = counts.most_common(TOP_CATEGORIES)
            top_count = sum(count for _, count in top)
            categories = [
                {"value": value, "count": count, "share": count / total if total else 0.0}
                for value, count in top
            ]
            if total > top_count:
                categories.append({
                    "value": "Other",
                    "count": total - top_count,
                    "share": (total - top_count) / total if total else 0.0,
                })
            profile["categories"] = categories
        elif semantic == "identifier":
            profile["uniqueness_rate"] = (
                (len(unique_values[column]) / non_missing)
                if non_missing and column not in unique_capped
                else None
            )
        elif semantic == "text":
            lengths = [len(value) for value in sample_non_missing]
            if lengths:
                profile["text_summary"] = {
                    "mean_length": statistics.fmean(lengths),
                    "median_length": statistics.median(lengths),
                    "max_length": max(lengths),
                }

        profiles.append(profile)

    active_columns = [
        column for column in columns
        if semantic_types.get(column, "categorical") not in {"identifier", "text", "ignore", "datetime"}
    ][:MAX_RELATIONSHIP_COLUMNS]

    relationships: list[dict[str, Any]] = []
    for index, left in enumerate(active_columns):
        for right in active_columns[index + 1:]:
            left_type = semantic_types.get(left, "categorical")
            right_type = semantic_types.get(right, "categorical")
            item: dict[str, Any] | None = None

            if left_type in {"continuous", "ordinal"} and right_type in {"continuous", "ordinal"}:
                pairs = [
                    (_to_float(row.get(left)), _to_float(row.get(right)))
                    for row in sample_rows
                ]
                xs = [a for a, b in pairs if a is not None and b is not None]
                ys = [b for a, b in pairs if a is not None and b is not None]
                score = _pearson(xs, ys)
                if score is not None:
                    item = {"left": left, "right": right, "kind": "Pearson correlation", "score": score}
            elif left_type in {"categorical", "boolean"} and right_type in {"categorical", "boolean"}:
                pairs = [
                    (str(row.get(left, "")).strip(), str(row.get(right, "")).strip())
                    for row in sample_rows
                    if not _is_missing(row.get(left)) and not _is_missing(row.get(right))
                ]
                score = _cramers_v(pairs)
                if score is not None:
                    item = {"left": left, "right": right, "kind": "Cramér's V", "score": score}
            else:
                numeric = left if left_type in {"continuous", "ordinal"} else right
                categorical = right if numeric == left else left
                score = _numeric_categorical_effect(sample_rows, numeric, categorical)
                if score is not None:
                    item = {"left": left, "right": right, "kind": "Correlation ratio η", "score": score}

            if item:
                relationships.append(item)

    relationships.sort(key=lambda item: abs(float(item["score"])), reverse=True)
    relationship_columns = active_columns

    population_comparison: list[dict[str, Any]] = []
    for population_column in population_columns:
        if population_column not in columns:
            continue
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in sample_rows:
            value = row.get(population_column, "").strip()
            if value and len(groups) < 30 or value in groups:
                groups[value].append(row)
        group_results = []
        for value, rows in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[:20]:
            total_cells = len(rows) * len(columns)
            missing_cells = sum(_is_missing(row.get(column)) for row in rows for column in columns)
            group_results.append({
                "value": value,
                "sample_rows": len(rows),
                "share": len(rows) / len(sample_rows) if sample_rows else 0.0,
                "missing_rate": missing_cells / total_cells if total_cells else 0.0,
            })
        population_comparison.append({"column": population_column, "groups": group_results})

    time_analysis: dict[str, Any] | None = None
    if time_column and time_column in columns:
        buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in sample_rows:
            parsed = _to_datetime(row.get(time_column))
            if parsed:
                buckets[_time_bucket(parsed, time_granularity)].append(row)

        bucket_rows = []
        continuous_columns = [
            column for column in columns
            if semantic_types.get(column) in {"continuous", "ordinal"} and column != time_column
        ][:8]
        for bucket, rows in sorted(buckets.items()):
            total_cells = len(rows) * len(columns)
            missing_cells = sum(_is_missing(row.get(column)) for row in rows for column in columns)
            variable_stats = {}
            for column in continuous_columns:
                values = [number for row in rows if (number := _to_float(row.get(column))) is not None]
                if values:
                    variable_stats[column] = {
                        "mean": statistics.fmean(values),
                        "median": statistics.median(values),
                        "p25": _quantile(values, 0.25),
                        "p75": _quantile(values, 0.75),
                        "missing_rate": sum(_is_missing(row.get(column)) for row in rows) / len(rows),
                    }
            column_missing_rates = {
                column: sum(_is_missing(row.get(column)) for row in rows) / len(rows)
                for column in columns
            }
            bucket_rows.append({
                "bucket": bucket,
                "observations": len(rows),
                "missing_rate": missing_cells / total_cells if total_cells else 0.0,
                "column_missing_rates": column_missing_rates,
                "variables": variable_stats,
            })
        time_analysis = {
            "time_column": time_column,
            "granularity": time_granularity,
            "buckets": bucket_rows,
        }

    missingness_diagnostics = []
    for column in columns:
        missing = missing_counts[column]
        if missing == 0:
            continue

        associations = []
        for other in columns:
            if other == column or semantic_types.get(other) in {"ignore", "identifier", "text", "datetime"}:
                continue
            result = _missingness_association(
                sample_rows,
                column,
                other,
                semantic_types.get(other, "categorical"),
            )
            if result:
                score, evidence = result
                associations.append({"column": other, "score": score, "evidence": evidence})

        associations.sort(key=lambda item: item["score"], reverse=True)
        strongest = associations[:3]
        top_score = strongest[0]["score"] if strongest else 0.0

        if top_score >= 0.25:
            assessment = "MAR plausible"
            explanation = "Missingness shows a material association with observed variables, which is inconsistent with a simple MCAR assumption."
        elif top_score < 0.10 and len(sample_rows) >= 100:
            assessment = "MCAR remains plausible"
            explanation = "No strong association with the tested observed variables was detected. This does not prove MCAR."
        else:
            assessment = "Insufficient evidence"
            explanation = "Observed associations are weak or mixed, so the missingness mechanism cannot be classified reliably."

        missingness_diagnostics.append({
            "column": column,
            "missing_count": missing,
            "missing_rate": missing / row_count,
            "assessment": assessment,
            "explanation": explanation,
            "strongest_associations": strongest,
            "mnar_note": "MNAR cannot be confirmed from observed data alone and requires domain knowledge and/or sensitivity analysis.",
        })

    total_missing = sum(missing_counts.values())
    total_cells = row_count * len(columns)

    return {
        "sample_size": len(sample_rows),
        "sample_limit": EDA_SAMPLE_ROWS,
        "overview": {
            "rows": row_count,
            "columns": len(columns),
            "missing_cells": total_missing,
            "missing_rate": total_missing / total_cells if total_cells else 0.0,
            "duplicate_rows": duplicate_count,
            "duplicate_rate": duplicate_count / row_count,
        },
        "profiles": profiles,
        "relationships": relationships,
        "relationship_columns": relationship_columns,
        "relationship_columns_total": len([
            column for column in columns
            if semantic_types.get(column, "categorical") not in {"identifier", "text", "ignore", "datetime"}
        ]),
        "relationship_columns_limit": MAX_RELATIONSHIP_COLUMNS,
        "population_comparison": population_comparison,
        "time_analysis": time_analysis,
        "missingness_diagnostics": missingness_diagnostics,
    }
