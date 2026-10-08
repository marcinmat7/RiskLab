import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from fastapi import Form, HTTPException

from app.dataset_io import get_dataset

EPSILON = 1e-6
NUMERIC_BINS = 10
TOP_CATEGORIES = 20


def _is_missing(value: str | None) -> bool:
    return value is None or not value.strip()


def _to_float(value: str | None) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def _to_datetime(value: str | None) -> datetime | None:
    if _is_missing(value):
        return None
    raw = str(value).strip()
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
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _psi_component(reference_share: float, comparison_share: float) -> float:
    ref = max(reference_share, EPSILON)
    comp = max(comparison_share, EPSILON)
    return (comp - ref) * math.log(comp / ref)


def _numeric_distribution(
    reference_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    column: str,
) -> tuple[float, list[dict[str, Any]]]:
    reference_values = [
        number for row in reference_rows
        if (number := _to_float(row.get(column))) is not None
    ]
    if not reference_values:
        return 0.0, []

    edges = sorted({_quantile(reference_values, index / NUMERIC_BINS) for index in range(NUMERIC_BINS + 1)})
    if len(edges) == 1:
        edges = [edges[0], edges[0] + 1e-12]

    def counts(rows: list[dict[str, str]]) -> tuple[list[int], int]:
        bin_counts = [0] * (len(edges) - 1)
        missing = 0
        for row in rows:
            value = _to_float(row.get(column))
            if value is None:
                missing += 1
                continue
            selected = len(bin_counts) - 1
            for index in range(len(edges) - 1):
                left, right = edges[index], edges[index + 1]
                if (index < len(edges) - 2 and left <= value < right) or (index == len(edges) - 2 and left <= value <= right):
                    selected = index
                    break
                if value < edges[0]:
                    selected = 0
                    break
            bin_counts[selected] += 1
        return bin_counts, missing

    ref_counts, ref_missing = counts(reference_rows)
    comp_counts, comp_missing = counts(comparison_rows)
    ref_total = max(len(reference_rows), 1)
    comp_total = max(len(comparison_rows), 1)

    result = []
    psi = 0.0
    for index, (ref_count, comp_count) in enumerate(zip(ref_counts, comp_counts)):
        ref_share = ref_count / ref_total
        comp_share = comp_count / comp_total
        psi += _psi_component(ref_share, comp_share)
        result.append({
            "label": f"[{edges[index]:.4g}, {edges[index + 1]:.4g}]" if index == len(ref_counts) - 1 else f"[{edges[index]:.4g}, {edges[index + 1]:.4g})",
            "reference_share": ref_share,
            "comparison_share": comp_share,
        })

    ref_missing_share = ref_missing / ref_total
    comp_missing_share = comp_missing / comp_total
    psi += _psi_component(ref_missing_share, comp_missing_share)
    result.append({
        "label": "Missing",
        "reference_share": ref_missing_share,
        "comparison_share": comp_missing_share,
    })
    return psi, result


def _categorical_distribution(
    reference_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    column: str,
) -> tuple[float, list[dict[str, Any]]]:
    ref_counter = Counter(str(row.get(column) or "").strip() for row in reference_rows if not _is_missing(row.get(column)))
    comp_counter = Counter(str(row.get(column) or "").strip() for row in comparison_rows if not _is_missing(row.get(column)))
    labels = [
        value for value, _ in (ref_counter + comp_counter).most_common(TOP_CATEGORIES)
    ]
    ref_total = max(len(reference_rows), 1)
    comp_total = max(len(comparison_rows), 1)
    result = []
    psi = 0.0

    for label in labels:
        ref_share = ref_counter[label] / ref_total
        comp_share = comp_counter[label] / comp_total
        psi += _psi_component(ref_share, comp_share)
        result.append({
            "label": label,
            "reference_share": ref_share,
            "comparison_share": comp_share,
        })

    ref_other = sum(count for value, count in ref_counter.items() if value not in labels)
    comp_other = sum(count for value, count in comp_counter.items() if value not in labels)
    if ref_other or comp_other:
        ref_share = ref_other / ref_total
        comp_share = comp_other / comp_total
        psi += _psi_component(ref_share, comp_share)
        result.append({"label": "Other", "reference_share": ref_share, "comparison_share": comp_share})

    ref_missing = sum(_is_missing(row.get(column)) for row in reference_rows) / ref_total
    comp_missing = sum(_is_missing(row.get(column)) for row in comparison_rows) / comp_total
    psi += _psi_component(ref_missing, comp_missing)
    result.append({"label": "Missing", "reference_share": ref_missing, "comparison_share": comp_missing})
    return psi, result


def _variable_stats(rows: list[dict[str, str]], column: str, semantic: str) -> dict[str, Any]:
    missing_rate = sum(_is_missing(row.get(column)) for row in rows) / len(rows) if rows else 0.0
    if semantic in {"continuous", "ordinal"}:
        values = [number for row in rows if (number := _to_float(row.get(column))) is not None]
        return {
            "mean": statistics.fmean(values) if values else None,
            "missing_rate": missing_rate,
        }
    return {"mean": None, "missing_rate": missing_rate}


def _compare_variable(
    reference_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    column: str,
    semantic: str,
) -> dict[str, Any]:
    if semantic in {"continuous", "ordinal"}:
        psi, distribution = _numeric_distribution(reference_rows, comparison_rows, column)
    else:
        psi, distribution = _categorical_distribution(reference_rows, comparison_rows, column)

    ref_stats = _variable_stats(reference_rows, column, semantic)
    comp_stats = _variable_stats(comparison_rows, column, semantic)
    return {
        "column": column,
        "semantic_type": semantic,
        "psi": psi,
        "reference_mean": ref_stats["mean"],
        "comparison_mean": comp_stats["mean"],
        "reference_missing_rate": ref_stats["missing_rate"],
        "comparison_missing_rate": comp_stats["missing_rate"],
        "missing_delta": comp_stats["missing_rate"] - ref_stats["missing_rate"],
        "distribution": distribution,
    }


async def run_stability(dataset_id: str = Form(...), config: str = Form(...)) -> dict[str, Any]:
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid stability configuration.") from exc

    semantic_types: dict[str, str] = cfg.get("semanticTypes", {})
    prediction_column = str(cfg.get("predictionColumn", "") or "")
    feature_columns = [str(column) for column in cfg.get("featureColumns", [])]
    population_columns = [str(column) for column in cfg.get("populationColumns", [])]
    time_column = str(cfg.get("timeColumn", "") or "")
    time_cutoff_date = str(cfg.get("timeCutoffDate", "") or "").strip()
    time_cutoff_as_segment = bool(cfg.get("timeCutoffAsSegment", False))
    before_label = str(cfg.get("timeCutoffBeforeLabel", "Pre cut-off") or "Pre cut-off").strip()
    after_label = str(cfg.get("timeCutoffAfterLabel", "Post cut-off") or "Post cut-off").strip()

    cutoff_datetime = _to_datetime(time_cutoff_date) if time_cutoff_date else None
    if time_cutoff_date and cutoff_datetime is None:
        raise HTTPException(status_code=400, detail="Time cut-off must be a valid date.")

    dataset = get_dataset(dataset_id)
    columns = list(dataset["columns"])
    rows: list[dict[str, str]] = dataset["rows"]

    configured_variables = []
    for column in [prediction_column, *feature_columns]:
        if (
            column
            and column in columns
            and column not in configured_variables
            and semantic_types.get(column, "categorical") not in {"identifier", "text", "ignore", "datetime"}
        ):
            configured_variables.append(column)

    if not configured_variables:
        configured_variables = [
            column for column in columns
            if semantic_types.get(column, "categorical") not in {"identifier", "text", "ignore", "datetime"}
            and column not in population_columns
        ][:40]

    sources: list[dict[str, Any]] = []
    source_groups: dict[str, dict[str, list[dict[str, str]]]] = {}

    for column in population_columns:
        if column not in columns:
            continue
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            value = str(row.get(column) or "").strip()
            if value:
                grouped[value].append(row)
        if len(grouped) < 2:
            continue
        ordered = dict(sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:20])
        source_groups[column] = ordered
        sources.append({
            "key": column,
            "name": column,
            "groups": [{"value": value, "observations": len(group_rows)} for value, group_rows in ordered.items()],
        })

    cutoff_key = "__time_cutoff__"
    if time_column and time_cutoff_as_segment and cutoff_datetime is not None:
        grouped = defaultdict(list)
        cutoff_date = cutoff_datetime.date()
        for row in rows:
            parsed = _to_datetime(row.get(time_column))
            if parsed is None:
                continue
            grouped[before_label if parsed.date() < cutoff_date else after_label].append(row)
        if len(grouped) >= 2:
            source_groups[cutoff_key] = dict(grouped)
            sources.append({
                "key": cutoff_key,
                "name": "Time cut-off",
                "groups": [{"value": value, "observations": len(group_rows)} for value, group_rows in grouped.items()],
            })

    comparisons: list[dict[str, Any]] = []
    for source in sources:
        key = source["key"]
        groups = source_groups[key]
        for reference, reference_rows in groups.items():
            for comparison, comparison_rows in groups.items():
                if reference == comparison:
                    continue
                variables = [
                    _compare_variable(
                        reference_rows,
                        comparison_rows,
                        column,
                        semantic_types.get(column, "categorical"),
                    )
                    for column in configured_variables
                ]
                variables.sort(key=lambda item: item["psi"], reverse=True)
                comparisons.append({
                    "source_key": key,
                    "source_name": source["name"],
                    "reference": reference,
                    "comparison": comparison,
                    "reference_observations": len(reference_rows),
                    "comparison_observations": len(comparison_rows),
                    "variables": variables,
                })

    time_analysis = None
    if time_column:
        parsed_rows = [
            (parsed, row)
            for row in rows
            if (parsed := _to_datetime(row.get(time_column))) is not None
        ]
        granularities: dict[str, list[dict[str, Any]]] = {}
        for granularity in ("daily", "weekly", "monthly", "quarterly", "yearly"):
            buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
            for parsed, row in parsed_rows:
                buckets[_time_bucket(parsed, granularity)].append(row)

            bucket_rows = []
            for bucket, bucket_data in sorted(buckets.items()):
                item: dict[str, Any] = {
                    "bucket": bucket,
                    "observations": len(bucket_data),
                    "references": [],
                }
                for source in sources:
                    key = source["key"]
                    for reference, reference_rows in source_groups[key].items():
                        variable_psis = []
                        for column in configured_variables:
                            semantic = semantic_types.get(column, "categorical")
                            comparison = _compare_variable(reference_rows, bucket_data, column, semantic)
                            variable_psis.append({
                                "column": column,
                                "psi": comparison["psi"],
                            })
                        item["references"].append({
                            "source_key": key,
                            "reference": reference,
                            "variables": variable_psis,
                        })
                bucket_rows.append(item)
            granularities[granularity] = bucket_rows
        time_analysis = {
            "time_column": time_column,
            "granularities": granularities,
        }

    return {
        "sources": sources,
        "comparisons": comparisons,
        "variables": [
            {"column": column, "semantic_type": semantic_types.get(column, "categorical")}
            for column in configured_variables
        ],
        "time_analysis": time_analysis,
        "thresholds": {
            "moderate": 0.10,
            "high": 0.25,
            "note": "Heuristic defaults; not universal regulatory thresholds.",
        },
    }
