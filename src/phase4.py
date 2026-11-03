"""Phase 4 diagnostics and Kaggle submission utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .evaluation import _design
from .pipeline import selected_features


def regression_diagnostics(pipe, X, y, max_rows=5000, seed=42):
    """Exploratory in-sample diagnostics, sampled only from the model training set."""
    if len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    Z = _design(pipe, X)
    yy = np.asarray(y, dtype=float)
    names = selected_features(pipe)
    # Singular columns are explicitly reported instead of silently claiming reliable inference.
    design = sm.add_constant(Z, has_constant='add')
    rank = np.linalg.matrix_rank(design)
    if len(yy) <= design.shape[1] + 2:
        raise ValueError('Insufficient rows for statistical diagnostics')
    fit = sm.OLS(yy, design).fit()
    residuals = np.asarray(fit.resid)
    fitted = np.asarray(fit.fittedvalues)
    bp = het_breuschpagan(residuals, design)
    jb = stats.jarque_bera(residuals)
    try:
        reset_p = float(linear_reset(fit, power=2, use_f=True).pvalue)
    except (ValueError, TypeError, ZeroDivisionError):
        reset_p = np.nan
    vif_rows=[]
    if Z.shape[1] <= 40:
        for i, name in enumerate(names):
            try:
                val = float(variance_inflation_factor(design, i + 1))
            except (ValueError, np.linalg.LinAlgError):
                val = np.inf
            vif_rows.append({'Feature': name, 'VIF': val})
    summary = {
        'Linearity (RESET) p-value':reset_p,
        'Independence (Durbin–Watson)':float(durbin_watson(residuals)),
        'Homoscedasticity (Breusch–Pagan) p-value':float(bp[1]),
        'Normality (Jarque–Bera) p-value':float(jb.pvalue),
        'Multicollinearity (maximum VIF)':max((r['VIF'] for r in vif_rows), default=np.nan),
        'Design rank':int(rank), 'Design columns':int(design.shape[1]),
        'Rows used':int(len(yy))
    }
    return summary, pd.DataFrame(vif_rows), pd.DataFrame({'Fitted':fitted,'Residual':residuals})


def make_flood_submission(pipe, official_test, sample_submission=None):
    required = list(pipe.feature_names_in_) if hasattr(pipe,'feature_names_in_') else None
    if required is None:
        raise ValueError('Model missing feature_names_in_')
    missing = [c for c in required if c not in official_test.columns]
    if missing:
        raise ValueError(f'Official test missing columns: {missing}')
    if 'id' not in official_test:
        raise ValueError('Official test missing id')
    pred = np.asarray(pipe.predict(official_test[required]), dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError('Model generated nonfinite predictions')
    submission = pd.DataFrame({'id':official_test['id'].to_numpy(), 'FloodProbability':pred})
    if sample_submission is not None:
        if len(sample_submission) != len(submission):
            raise ValueError('Sample submission row count mismatch')
        if list(sample_submission.columns) != ['id','FloodProbability']:
            raise ValueError('Unexpected sample submission columns')
        if not np.array_equal(sample_submission['id'].to_numpy(),submission['id'].to_numpy()):
            raise ValueError('Sample submission id order mismatch')
    return submission
