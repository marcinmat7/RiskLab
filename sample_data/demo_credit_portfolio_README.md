# RiskLab demo credit portfolio

`demo_credit_portfolio.csv` is a deterministic synthetic dataset designed to exercise the RiskLab workflow end to end.

## Suggested Validation setup

### Required

- **Model output column:** `model_pd`
- **Output type:** Probability of Default (PD)
- **Default indicator:** `default_12m`
- **Default class:** `1`

You can also try `application_score` as the model output with **Output type = Score** and **Higher score = lower risk**.

### Optional analysis settings

- **Time column:** `observation_date`
- **Sample / population column:** `sample_type`
- **Sensitive attributes:** `gender`, `age`
- **Model features:** for example
  - `age`
  - `employment_type`
  - `income_monthly`
  - `debt_to_income`
  - `utilization`
  - `delinquencies_12m`
  - `has_mortgage`
  - `tenure_months`
  - `bureau_score`
  - `employer_size`

## Recommended semantic types

| Column | Semantic type |
| --- | --- |
| customer_id | Identifier |
| observation_date | Date / time |
| sample_type | Categorical |
| country | Categorical |
| channel | Categorical |
| gender | Categorical |
| age | Continuous |
| employment_type | Categorical |
| income_monthly | Continuous |
| debt_to_income | Continuous |
| utilization | Continuous |
| delinquencies_12m | Ordinal |
| has_mortgage | Boolean |
| tenure_months | Continuous |
| bureau_score | Continuous |
| employer_size | Ordinal or Categorical |
| marketing_source | Categorical |
| customer_note | Text |
| application_score | Continuous |
| model_pd | Continuous |
| default_12m | Boolean |

## What the dataset is designed to demonstrate

- **1,208 rows**, so the CSV upload workflow shows the large-dataset warning.
- **TRAIN / TEST / OOT** populations with modest distribution shifts, so Population comparison has something visible to compare.
- Data spans roughly **2024 through mid-2026**, enabling Time analysis.
- `income_monthly` missingness becomes more frequent in later periods, so missingness-over-time charts should show a visible change.
- `bureau_score` missingness is more frequent for younger customers, creating an observed-variable association that can look MAR-like.
- `employer_size` is structurally absent for several employment groups, also creating a strong observed association.
- `marketing_source` has approximately random missingness, useful as a contrasting missingness pattern.
- Eight exact duplicate rows are appended so duplicate detection is non-zero.
- Numeric, categorical, boolean, ordinal, identifier, date/time and text-style columns are all represented.
- Continuous variables have deliberately correlated structure so the Relationships view is non-empty.
- `model_pd` and `default_12m` are generated from a simple synthetic risk process; the dataset is for product demonstration only and is not intended to represent a real credit portfolio.

All records are fully synthetic and contain no real customer data.
