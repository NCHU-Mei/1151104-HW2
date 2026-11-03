# Phase 4 acceptance checklist

1. `python -m pip install -r requirements.txt`
2. `python -m pytest -q` (expected: four tests)
3. `streamlit run app.py`
4. Human–AI: run Baseline on a 2,000-row cap, inspect all seven existing tabs and new Diagnostics tab.
5. Diagnostics: run RESET, DW, BP, JB, VIF; inspect residual and Q-Q plots; export both CSV files.
6. Flood: run Baseline with 2,000-row cap; open Deployment; click Generate Kaggle Submission (leave full-data refit off for quick smoke test); verify 745,305 rows and exact `id,FloodProbability` header and official order.
7. Flood: optional full-data refit may be slow and memory intensive; verify only on sufficiently provisioned machine.
8. Custom CSV URL: test public HTTPS CSV; verify target selection and invalid URL errors.
9. Verify no stale model results are shown after changing dataset or hyperparameters.
10. Validate mobile layout, chart interaction, and download buttons in a real browser.

Known caveats: Streamlit runtime/browser test not executed in this environment; performance of full-data refit and SFS depends on hardware; prediction intervals are conditional on the selected model, not selection-adjusted.
