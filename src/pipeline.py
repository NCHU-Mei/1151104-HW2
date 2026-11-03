"""Leakage-safe linear regression pipeline and feature selectors."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SequentialFeatureSelector, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class TopKPermutationSelector(BaseEstimator, TransformerMixin):
    """Fits a model on training fold only and selects columns by training permutation importance."""
    def __init__(self, k=10, n_repeats=3, random_state=42):
        self.k = k
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        model = LinearRegression().fit(X, y)
        scores = permutation_importance(model, X, y, scoring='neg_mean_squared_error',
                                        n_repeats=self.n_repeats, random_state=self.random_state,
                                        n_jobs=1).importances_mean
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[np.argsort(scores)[-min(self.k, X.shape[1]):]] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_


def make_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=False)),
                         ('scale', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))])
    return ColumnTransformer([('num', num_pipe, numeric), ('cat', cat_pipe, categorical)],
                             remainder='drop', verbose_feature_names_out=True)


def build_pipeline(X, method='baseline', k=10, random_state=42):
    prep = make_preprocessor(X)
    # Determine expanded feature count using the training partition only; do not fit on holdout.
    steps = [('prep', prep)]
    if method == 'kbest':
        steps.append(('select', SelectKBest(score_func=f_regression, k=k)))
    elif method == 'rfe':
        steps.append(('select', RFE(LinearRegression(), n_features_to_select=k, step=0.2)))
    elif method == 'sfs':
        steps.append(('select', SequentialFeatureSelector(LinearRegression(), n_features_to_select=k,
                          direction='forward', cv=3, n_jobs=1)))
    elif method == 'lasso':
        steps.append(('select', SelectFromModel(LassoCV(cv=3, random_state=random_state,
                              max_iter=5000, n_jobs=1), threshold=1e-8)))
    elif method == 'permutation':
        steps.append(('select', TopKPermutationSelector(k=k, random_state=random_state)))
    elif method != 'baseline':
        raise ValueError(f'Unknown feature selection method: {method}')
    steps.append(('model', LinearRegression()))
    return Pipeline(steps)


def selected_features(fitted_pipeline):
    names = fitted_pipeline.named_steps['prep'].get_feature_names_out()
    if 'select' not in fitted_pipeline.named_steps:
        return list(names)
    mask = fitted_pipeline.named_steps['select'].get_support()
    return list(names[mask])
