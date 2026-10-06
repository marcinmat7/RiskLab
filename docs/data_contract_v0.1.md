# RiskLab Data Contract v0.1

Status: **Frozen for MVP v0.1**

This document defines the input data contract for the first RiskLab MVP.

## 1. Scope

RiskLab v0.1 supports CSV input and the following analytical flow:

`CSV -> column mapping -> semantic validation -> AUC/Gini/KS -> calibration -> PSI -> dashboard`

The system must not infer column semantics solely from column names. The user explicitly maps dataset columns to RiskLab roles.

## 2. Required roles

### Target

A column representing the observed binary outcome.

Examples:
- `0 / 1`
- `GOOD / BAD`
- `N / Y`
- `False / True`

The user must identify which value represents the positive/default class.

### Prediction

A numeric model output used for ranking risk.

Supported prediction types:

1. **Probability of Default (PD)**
   - numeric
   - expected range: `[0, 1]`
   - higher value means higher risk
   - enables discrimination and calibration analysis

2. **Score**
   - numeric
   - no fixed range requirement
   - user must specify score direction:
     - higher value = higher risk, or
     - higher value = lower risk
   - enables discrimination analysis
   - calibration analysis is unavailable unless a calibrated PD is supplied

## 3. Optional roles

### Date

Optional observation/scoring date.

Used for:
- time-based slicing,
- reference/comparison period construction,
- future stability and monitoring analyses.

The selected column must be parseable as a date or datetime.

### ID

Optional record/customer/application identifier.

RiskLab does not use this column directly for model metrics in v0.1, but it may be retained for traceability and future drill-down functionality.

### Segments

Optional set of segment columns.

Cardinality:
- zero or more segment columns may be selected,
- the data model should not assume a single segment column.

Examples:
- `country`
- `product`
- `channel`
- `customer_type`

Segment columns enable metric breakdowns by population subgroup.

## 4. Mapping flow

After CSV upload, RiskLab follows this order:

1. Basic file validation
2. Dataset preview
3. Column mapping
4. Semantic configuration
5. Semantic validation
6. Error/warning display
7. Analysis
8. Dashboard

### Mapping configuration

The user maps arbitrary CSV column names to these roles:

- `target` — required
- `prediction` — required
- `date` — optional
- `id` — optional
- `segments[]` — optional, zero or more columns

The user also configures:

- prediction type: `PD` or `Score`
- positive/default target class
- score direction when prediction type is `Score`

## 5. Validation rules

RiskLab distinguishes between blocking **errors** and non-blocking **warnings**.

### File-level validation

Before semantic mapping:

- file can be parsed as CSV,
- file is not empty,
- header row exists,
- at least two columns exist,
- duplicate column names are not allowed.

### Target validation

Blocking errors:

- mapped target column does not exist,
- target contains fewer than two classes,
- selected positive/default class is absent,
- target cannot be reduced to a binary outcome.

Warnings:

- missing target values are present,
- one class is extremely rare.

### PD validation

Blocking errors:

- prediction column is non-numeric,
- prediction contains non-finite values,
- values fall outside `[0, 1]`.

Warnings:

- missing prediction values are present,
- prediction has too few unique values,
- prediction is constant or nearly constant.

### Score validation

Blocking errors:

- prediction column is non-numeric,
- prediction contains non-finite values.

Warnings:

- missing prediction values are present,
- score has too few unique values,
- score is constant or nearly constant.

### Date validation

Blocking errors:

- selected date column cannot be parsed as date/datetime.

Warnings:

- missing dates are present.

### Segment validation

Blocking errors:

- selected segment column does not exist.

Warnings:

- segment is entirely missing,
- segment has only one distinct value,
- segment has very high cardinality,
- segment appears identifier-like and may be unsuitable for subgroup analysis.

## 6. Errors vs warnings

Examples:

### Error

`PD contains values greater than 1. Analysis cannot continue.`

### Warning

`17% of observations have missing prediction values. These rows will be excluded from affected analyses.`

### Warning

`Segment 'customer_id' has very high cardinality and may not be suitable as a segment.`

## 7. Analysis availability by prediction type

| Analysis | PD | Score |
|---|---:|---:|
| ROC AUC | Yes | Yes |
| Gini | Yes | Yes |
| KS | Yes | Yes |
| Calibration curve | Yes | No |
| Brier score | Yes | No |
| Observed vs expected | Yes | No |
| PSI | Yes | Yes |

PSI requires a reference population and a comparison population. In v0.1 these populations may be derived from a mapped date/period field or another explicit population split supported by the UI.

## 8. Example CSV

```csv
customer_id,bad_12m,model_pd,observation_date,country,product
100001,0,0.018,2025-01-15,PL,BNPL
100002,1,0.214,2025-01-16,DE,Loan
100003,0,0.041,2025-01-18,SE,BNPL
100004,0,0.027,2025-02-02,PL,Loan
```

Possible mapping:

- target -> `bad_12m`
- prediction -> `model_pd`
- prediction type -> `PD`
- positive/default class -> `1`
- date -> `observation_date`
- id -> `customer_id`
- segments -> `country`, `product`

## 9. Frozen MVP v0.1 contract

Required:

- `target`
- `prediction`

Optional:

- `date`
- `id`
- `segments[]`

Semantics are defined by explicit user mapping, not by column names.

Any change to this contract after MVP implementation starts should be treated as an explicit versioned change to the data contract.
