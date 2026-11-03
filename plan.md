# HW2 — ML Regression Lab: Project Plan

## 1. Objective
Build a deployable Streamlit + scikit-learn web application demonstrating the six CRISP-DM phases for multiple linear regression. Built-in datasets: Human–AI Interaction and Kaggle Playground Series S4E5 Flood Prediction. Support arbitrary publicly accessible CSV URL and user-selected numeric regression target.

## 2. Verified data schema
- Human-AI Interaction.csv: 21 columns, 20 raw predictors, target `latent_trust_score`; mixed numeric and categorical features. Missing values require inspection and preprocessing.
- Flood train.csv: 22 columns: `id`, 20 predictors, `FloodProbability` target.
- Flood test.csv: 21 columns: `id` and 20 predictors; **no ground truth**.
- Flood sample_submission.csv: `id`, `FloodProbability`.
- Never use `id` as a predictor. Never score Kaggle's unlabeled test.csv with RMSE/R².

## 3. Stack
Python 3.11+, Streamlit, pandas, numpy, scikit-learn, scipy, statsmodels, plotly, matplotlib, pytest. GitHub + Streamlit Community Cloud. Pin tested dependency versions.

## 4. Product requirements
1. Sidebar dataset selector (Human–AI, Flood, Custom CSV URL); display provenance, target, raw feature count and data quality.
2. CSV URL input with timeouts, size caps, error handling, CSV validation, target selector, and controls to exclude ID/leakage columns. For hosted service, restrict unsafe network targets (SSRF prevention), allow only http(s) public URLs and cap response size.
3. Six CRISP-DM tabs plus Overview. EDA: head, schema, missingness, descriptive stats, correlations, target distribution.
4. Data preparation: labeled holdout split (default 80/20, random_state=42); training-only imputers, encoding, scaling and selection via Pipeline/ColumnTransformer. Avoid target leakage.
5. Baseline all features + five selection schemes: SelectKBest(f_regression), RFE(LinearRegression), SequentialFeatureSelector, Lasso-based selection, and permutation-importance ranking computed using training folds. Hyperparameter/feature count selection must not use holdout. For mixed data, clearly show encoded vs raw features.
6. Training-only 5-fold CV and final untouched holdout evaluation: MAE, MSE, RMSE, R², adjusted R² (only when mathematically defined), CV mean/std, runtime and feature count.
7. Visualization: actual vs predicted with y=x, residuals vs fitted, histogram, Q-Q plot, coefficient/feature ranking, performance comparison, 95% prediction intervals. Compute valid OLS intervals using statsmodels on a selected fixed design and document assumptions; avoid labeling arbitrary RMSE bands as calibrated 95% intervals.
8. Regression diagnostics: linearity/residual patterns, independence (Durbin–Watson with caveats for cross-sectional data), homoscedasticity (Breusch–Pagan), residual normality (Q-Q and optional test), multicollinearity (VIF). Interpret rather than only displaying p-values.
9. Flood Kaggle submission: train selected model on all labeled training rows after evaluation, predict unlabeled test.csv and download CSV matching sample_submission.csv.
10. Download charts, predictions, metrics and selected features. Present GPT-assisted content and a user-completed NotebookLM research summary (>=100 Chinese characters, genuine source citations). Export conversation PDF separately.

## 5. Performance modes
Quick: 10,000 train rows; Standard: 100,000 train rows; Full: all rows. Fixed holdout from labeled data first, then training subset for sampling; reproducible random seeds. Cache loaded data and fitted results keyed by settings. Warn that SFS + 5-fold CV on full Flood data is expensive; cap or disable SFS in Full mode with an explicit explanation rather than silently switching algorithms.

## 6. Suggested repository
```
HW2-Regression-Lab/
  app.py
  requirements.txt
  README.md
  plan.md
  workflow.md
  .streamlit/config.toml
  data/README.md
  src/data_loader.py
  src/eda.py
  src/preprocessing.py
  src/feature_selection.py
  src/modeling.py
  src/evaluation.py
  src/diagnostics.py
  src/plots.py
  src/submission.py
  tests/test_data_loader.py
  tests/test_pipeline.py
  tests/test_evaluation.py
  reports/notebooklm_summary.md
  reports/gpt_assistance.md
```
Large Kaggle CSV files should be fetched or hosted in a size-appropriate manner, not assumed available on Community Cloud. GitHub LFS is not automatically equivalent to runtime download. Include reproducible setup and data-source instructions.

## 7. Acceptance criteria
- Both built-in problems load and display correct target/features.
- Custom public CSV URL loads, user selects target, and helpful validation errors appear.
- Six CRISP-DM stages accessible.
- Baseline + five methods produce comparable train CV and holdout metrics without leakage.
- Prediction interval plot labels method and assumptions.
- Flood sample submission schema matches template and preserves ids/order.
- Tests pass and deployed Streamlit link works.
- README includes dataset attribution, commands, limitations, NotebookLM section and GPT disclosure.

## 8. Milestones
M1: dataset validation + docs; M2: leak-free ML engine + tests; M3: Streamlit UI; M4: plots, diagnostics and export; M5: deployment and report compliance.
