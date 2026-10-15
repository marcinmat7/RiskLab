import json
import math
from collections import Counter, defaultdict
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


def _roc_metrics(rows: list[tuple[float, int]]) -> tuple[float | None, float | None]:
    positives = sum(target for _, target in rows)
    negatives = len(rows) - positives
    if len(rows) < 2 or positives == 0 or negatives == 0:
        return None, None

    ordered = sorted(rows, key=lambda item: item[0], reverse=True)
    tp = fp = 0
    previous_score: float | None = None
    points = [{"fpr": 0.0, "tpr": 0.0}]
    ks = 0.0

    for score, target in ordered:
        if previous_score is not None and score != previous_score:
            tpr = tp / positives
            fpr = fp / negatives
            points.append({"fpr": fpr, "tpr": tpr})
            ks = max(ks, abs(tpr - fpr))
        if target:
            tp += 1
        else:
            fp += 1
        previous_score = score

    points.append({"fpr": 1.0, "tpr": 1.0})
    auc = sum(
        (right["fpr"] - left["fpr"]) * (right["tpr"] + left["tpr"]) / 2
        for left, right in zip(points, points[1:])
    )
    return auc, ks


def _build_labels(
    records: list[dict[str, Any]],
    column: str,
    semantic_type: str,
) -> tuple[dict[int, str | None], dict[str, Any]]:
    labels: dict[int, str | None] = {}

    if semantic_type in {"continuous", "ordinal"}:
        values = [
            value
            for record in records
            if (value := _to_float(record["row"].get(column))) is not None
        ]
        if not values:
            return labels, {"mode": "quantiles", "bins": [], "edges": []}

        q1 = _quantile(values, 1 / 3)
        q2 = _quantile(values, 2 / 3)
        minimum = min(values)
        maximum = max(values)
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
        return labels, {
            "mode": "quantiles",
            "bins": bin_labels,
            "edges": [minimum, q1, q2, maximum],
        }

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
        "edges": [],
    }


def _group_metrics(
    rows: list[dict[str, Any]],
    risk_threshold: float,
    prediction_type: str,
) -> dict[str, Any]:
    observations = len(rows)
    defaults = sum(int(row["target"]) for row in rows)
    non_defaults = observations - defaults
    pairs = [(float(row["risk_score"]), int(row["target"])) for row in rows]
    auc, ks = _roc_metrics(pairs)

    high_risk = [row for row in rows if float(row["risk_score"]) >= risk_threshold]
    low_risk = observations - len(high_risk)

    true_positive = sum(1 for row in high_risk if int(row["target"]) == 1)
    false_positive = len(high_risk) - true_positive
    false_negative = defaults - true_positive
    true_negative = non_defaults - false_positive

    mean_raw_prediction = (
        sum(float(row["prediction"]) for row in rows) / observations
        if observations else 0.0
    )
    mean_risk_score = (
        sum(float(row["risk_score"]) for row in rows) / observations
        if observations else 0.0
    )
    default_rate = defaults / observations if observations else 0.0

    result: dict[str, Any] = {
        "observations": observations,
        "defaults": defaults,
        "default_rate": default_rate,
        "mean_prediction": mean_raw_prediction,
        "mean_risk_score": mean_risk_score,
        "auc": auc,
        "ks": ks,
        "high_risk_rate": len(high_risk) / observations if observations else 0.0,
        "favorable_selection_rate": low_risk / observations if observations else 0.0,
        "tpr_bad_detection": true_positive / defaults if defaults else None,
        "fpr_bad_detection": false_positive / non_defaults if non_defaults else None,
        "tnr_good_acceptance": true_negative / non_defaults if non_defaults else None,
        "fnr_bad_missed": false_negative / defaults if defaults else None,
        "ppv_default_given_high_risk": true_positive / len(high_risk) if high_risk else None,
        "npv_non_default_given_low_risk": true_negative / low_risk if low_risk else None,
    }

    if prediction_type == "pd":
        expected_defaults = sum(float(row["prediction"]) for row in rows)
        mean_pd = mean_raw_prediction
        result.update({
            "mean_pd": mean_pd,
            "calibration_gap": default_rate - mean_pd,
            "oe_ratio": defaults / expected_defaults if expected_defaults > 0 else None,
        })
    else:
        result.update({
            "mean_pd": None,
            "calibration_gap": None,
            "oe_ratio": None,
        })

    return result


def _spread(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return max(clean) - min(clean) if len(clean) >= 2 else None


async def run_fairness(
    dataset_id: str = Form(...),
    config: str = Form(...),
    high_risk_share: float = Form(0.20),
) -> dict[str, Any]:
    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid fairness configuration.") from exc

    prediction_column = cfg.get("predictionColumn", "")
    target_column = cfg.get("targetColumn", "")
    positive_class = str(cfg.get("positiveClass", ""))
    prediction_type = cfg.get("predictionType", "pd")
    score_direction = cfg.get("scoreDirection", "higher-risk")
    sensitive_columns = list(cfg.get("sensitiveColumns", []))
    semantic_types: dict[str, str] = cfg.get("semanticTypes", {})

    if not sensitive_columns:
        raise HTTPException(
            status_code=400,
            detail="Fairness analysis requires at least one Sensitive attribute in Validation.",
        )
    if not prediction_column or not target_column or not positive_class:
        raise HTTPException(
            status_code=400,
            detail="Prediction, target and positive class are required.",
        )
    if not 0.01 <= high_risk_share <= 0.99:
        raise HTTPException(
            status_code=400,
            detail="High-risk share must be between 0.01 and 0.99.",
        )

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
        if raw_target is None or str(raw_target).strip() == "":
            excluded_missing_target += 1
            continue

        target = 1 if str(raw_target).strip() == positive_class else 0
        risk_score = prediction
        if prediction_type == "score" and score_direction == "lower-risk":
            risk_score = -prediction

        records.append({
            "prediction": prediction,
            "risk_score": risk_score,
            "target": target,
            "row": row,
        })

    if len(records) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough usable observations for fairness analysis.",
        )

    risk_scores = [float(record["risk_score"]) for record in records]
    risk_threshold = _quantile(risk_scores, 1.0 - high_risk_share)

    attributes = []
    for column in sensitive_columns:
        labels, metadata = _build_labels(
            records,
            column,
            semantic_types.get(column, "categorical"),
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        excluded_missing_sensitive = 0

        for index, record in enumerate(records):
            label = labels.get(index)
            if label is None:
                excluded_missing_sensitive += 1
                continue
            grouped[label].append(record)

        groups = [
            {
                "value": label,
                **_group_metrics(group_rows, risk_threshold, prediction_type),
            }
            for label, group_rows in sorted(
                grouped.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        ]

        favorable_rates = [float(group["favorable_selection_rate"]) for group in groups]
        tpr_values = [group["tpr_bad_detection"] for group in groups]
        fpr_values = [group["fpr_bad_detection"] for group in groups]
        auc_values = [group["auc"] for group in groups]
        calibration_values = [
            abs(float(group["calibration_gap"]))
            if group["calibration_gap"] is not None else None
            for group in groups
        ]

        max_favorable = max(favorable_rates) if favorable_rates else None
        min_favorable = min(favorable_rates) if favorable_rates else None
        disparate_impact_ratio = (
            min_favorable / max_favorable
            if max_favorable is not None and max_favorable > 0 and min_favorable is not None
            else None
        )

        tpr_gap = _spread(tpr_values)
        fpr_gap = _spread(fpr_values)

        attributes.append({
            "column": column,
            "semantic_type": semantic_types.get(column, "categorical"),
            "mode": metadata["mode"],
            "bins": metadata["bins"],
            "edges": metadata["edges"],
            "excluded_missing_sensitive": excluded_missing_sensitive,
            "groups": groups,
            "summary": {
                "demographic_parity_gap": _spread(favorable_rates),
                "disparate_impact_ratio": disparate_impact_ratio,
                "bad_detection_tpr_gap": tpr_gap,
                "bad_detection_fpr_gap": fpr_gap,
                "equalized_odds_gap": max(
                    [value for value in [tpr_gap, fpr_gap] if value is not None],
                    default=None,
                ),
                "auc_gap": _spread(auc_values),
                "calibration_gap_spread": _spread(calibration_values),
            },
        })

    return {
        "high_risk_share": high_risk_share,
        "risk_threshold": risk_threshold,
        "risk_direction": "higher-is-riskier",
        "prediction_type": prediction_type,
        "attributes": attributes,
        "excluded": {
            "missing_prediction": excluded_missing_prediction,
            "missing_target": excluded_missing_target,
        },
        "notes": [
            "Continuous and ordinal sensitive attributes are split into three approximately equal-frequency groups.",
            "Categorical sensitive attributes keep up to 20 most common categories; remaining values are grouped as Other.",
            "The operating point is defined as the highest-risk share of the full usable portfolio, not as a legal or business approval threshold.",
            "Demographic parity uses the low-risk classification as the favorable outcome.",
            "Fairness diagnostics do not by themselves establish legal or regulatory compliance.",
        ],
    }
