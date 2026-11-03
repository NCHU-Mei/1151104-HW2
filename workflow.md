# HW2 Regression Lab — Workflow

```mermaid
flowchart TD
  A[Select Human-AI / Flood / CSV URL] --> B[Validate schema and target]
  B --> C[CRISP-DM: Business and Data Understanding]
  C --> D[Separate labeled data and optional Kaggle unlabeled test]
  D --> E[Split labeled data into train and holdout]
  E --> F[Fit preprocessing inside CV folds]
  F --> G[Baseline and 5 feature-selection pipelines]
  G --> H[5-fold CV on training data]
  H --> I[Select method without viewing holdout]
  I --> J[One-time holdout evaluation]
  J --> K[Metrics, diagnostics and 95% prediction interval]
  K --> L[Charts, reports and downloadable outputs]
  L --> M{Flood Kaggle submission?}
  M -->|Yes| N[Refit selected pipeline on all labeled rows]
  N --> O[Predict official unlabeled test and export submission]
  M -->|No| P[Predict user-provided new rows]
```

## UI navigation
Overview → Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.

## Reproducibility
Set random_state=42, pin versions, record sample mode and seed, retain model configuration and feature lists with each run. Run unit tests before GitHub push and Streamlit Cloud deployment.
