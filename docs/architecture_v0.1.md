# RiskLab Architecture v0.1

Status: **Proposed foundation for MVP v0.1**

## Goal

Build RiskLab as a product-oriented web application from the start, while keeping all credit-risk logic independent from the UI.

The first vertical slice should prove the full path:

`React UI -> FastAPI -> Python risk engine -> JSON response -> React UI`

No risk metric is required in the bootstrap slice yet; the first endpoint only proves frontend-backend connectivity.

## Technology choices

### Frontend

- React
- TypeScript
- Vite

Planned later:
- Tailwind CSS
- shadcn/ui
- Plotly.js or ECharts

### Backend

- FastAPI
- Pydantic
- Python 3.12+

### Risk engine

Pure Python package, independent from FastAPI and React.

Planned modules:

- data loading
- column mapping
- semantic validation
- discrimination metrics
- calibration metrics
- stability metrics

## Architectural rule

The frontend must not implement credit-risk calculations.

The FastAPI layer must stay thin and act as an adapter between HTTP requests and the Python risk engine.

The Python risk engine must be callable without the web application, for example from tests or notebooks.

## Repository layout

```text
RiskLab/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── api/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   └── pyproject.toml
│
├── risklab/
│   ├── __init__.py
│   ├── data/
│   ├── metrics/
│   └── models/
│
├── tests/
├── docs/
└── README.md
```

## MVP application flow

1. Upload CSV
2. Preview dataset
3. Map columns
4. Configure semantics
5. Validate dataset
6. Run analysis
7. Show dashboard

## Planned implementation slices

### Slice 0 - Bootstrap

- Vite React TypeScript app
- FastAPI app
- `GET /health`
- frontend calls backend and shows connectivity status

### Slice 1 - CSV upload and mapping

- CSV file input
- backend CSV parsing
- dataset preview
- column role mapping

### Slice 2 - Semantic validation

- target validation
- prediction validation
- PD vs score handling
- errors vs warnings

### Slice 3 - Discrimination

- ROC AUC
- Gini
- KS
- ROC chart

### Slice 4 - Calibration

- observed default rate
- average PD
- Brier score
- observed/expected
- calibration curve

### Slice 5 - Stability

- reference/comparison populations
- PSI

### Slice 6 - Segments

- metric breakdowns across zero or more segment columns

## Out of scope for bootstrap

- authentication
- database
- persistent file storage
- model inventory
- scheduled monitoring
- report generation
- LLM features
- deployment automation

These can be introduced after the analytical MVP works end-to-end.
