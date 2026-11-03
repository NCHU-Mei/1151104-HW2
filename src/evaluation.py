"""Holdout metrics, OLS prediction intervals, and residual diagnostics."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


def metrics(y, pred, n_features):
    y = np.asarray(y)
    pred = np.asarray(pred)
    n = len(y)
    r2 = r2_score(y, pred)
    return {'MAE': float(mean_absolute_error(y, pred)),
            'MSE': float(mean_squared_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2),
            'Adjusted_R2': float(1 - (1-r2)*(n-1)/(n-n_features-1)) if n > n_features+1 else np.nan,
            'n_features': int(n_features)}


def _design(pipe, X):
    arr = pipe.named_steps['prep'].transform(X)
    if 'select' in pipe.named_steps:
        arr = pipe.named_steps['select'].transform(arr)
    return np.asarray(arr, dtype=float)


def ols_prediction_intervals(pipe, X_train, y_train, X_test, alpha=0.05):
    """Conditional OLS intervals for the fixed selected design; not selection-adjusted."""
    train_design = sm.add_constant(_design(pipe, X_train), has_constant='add')
    test_design = sm.add_constant(_design(pipe, X_test), has_constant='add')
    if train_design.shape[0] <= train_design.shape[1]:
        raise ValueError('Insufficient training rows for OLS prediction intervals')
    fitted = sm.OLS(np.asarray(y_train, dtype=float), train_design).fit()
    sf = fitted.get_prediction(test_design).summary_frame(alpha=alpha)
    return pd.DataFrame({'predicted': sf['mean'].to_numpy(),
                         'mean_ci_lower': sf['mean_ci_lower'].to_numpy(),
                         'mean_ci_upper': sf['mean_ci_upper'].to_numpy(),
                         'prediction_lower': sf['obs_ci_lower'].to_numpy(),
                         'prediction_upper': sf['obs_ci_upper'].to_numpy()})


def diagnostics(pipe, X_train, y_train, max_vif_features=40):
    design = _design(pipe, X_train)
    fit = sm.OLS(np.asarray(y_train), sm.add_constant(design, has_constant='add')).fit()
    resid = fit.resid
    exog = fit.model.exog
    bp = het_breuschpagan(resid, exog)
    result = {'durbin_watson': float(durbin_watson(resid)),
              'breusch_pagan_p': float(bp[1]),
              'jarque_bera_p': float(stats.jarque_bera(resid).pvalue)}
    if design.shape[1] <= max_vif_features:
        result['vif'] = dict(zip(selected_feature_names(pipe),
                                  [float(variance_inflation_factor(exog, i+1)) for i in range(design.shape[1])]))
    return result


def selected_feature_names(pipe):
    from .pipeline import selected_features
    return selected_features(pipe)
