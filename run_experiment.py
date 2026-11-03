"""CLI: python run_experiment.py --dataset human --mode quick"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.base import clone
from src.data import load_dataset
from src.pipeline import build_pipeline, selected_features
from src.evaluation import metrics, ols_prediction_intervals

METHODS = ['baseline', 'kbest', 'rfe', 'sfs', 'lasso', 'permutation']

def run(dataset='human', data_dir='data', output_dir='outputs', mode='quick', k=10,
        methods=None, csv_path=None, target=None, random_state=42, cv_folds=5):
    X, y = load_dataset(dataset, data_dir, csv_path, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    cap = {'quick': 10000, 'standard': 100000, 'full': None}[mode]
    if cap and len(X_train) > cap:
        chosen = X_train.sample(n=cap, random_state=random_state).index
        X_train, y_train = X_train.loc[chosen], y_train.loc[chosen]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    methods = methods or METHODS
    rows, predictions, selected = [], {}, {}
    for method in methods:
        print(f'Running {method} on {len(X_train):,} train rows...', flush=True)
        t0 = time.time()
        pipe = build_pipeline(X_train, method=method, k=k, random_state=random_state)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring={'r2':'r2', 'mae':'neg_mean_absolute_error', 'rmse':'neg_root_mean_squared_error'}, n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        names = selected_features(pipe)
        row = {'method':method, **metrics(y_test, pred, len(names)),
               'CV_R2_mean':float(np.mean(scores['test_r2'])),
               'CV_R2_std':float(np.std(scores['test_r2'])),
               'CV_MAE_mean':float(-np.mean(scores['test_mae'])),
               'CV_RMSE_mean':float(-np.mean(scores['test_rmse'])),
               'runtime_seconds':round(time.time()-t0, 2)}
        rows.append(row)
        predictions[method] = pred
        selected[method] = names
        print(row, flush=True)
    comparison = pd.DataFrame(rows).sort_values('CV_RMSE_mean')
    comparison.to_csv(output/'model_comparison.csv', index=False)
    pd.DataFrame({'actual':y_test.to_numpy(), **predictions}).to_csv(output/'holdout_predictions.csv', index=False)
    (output/'selected_features.json').write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding='utf-8')
    best = comparison.iloc[0]['method']
    best_pipe = build_pipeline(X_train, method=best, k=k, random_state=random_state).fit(X_train, y_train)
    try:
        intervals = ols_prediction_intervals(best_pipe, X_train, y_train, X_test.iloc[:min(500, len(X_test))])
        intervals.insert(0, 'actual', y_test.iloc[:len(intervals)].to_numpy())
        intervals.to_csv(output/'prediction_intervals.csv', index=False)
    except (ValueError, np.linalg.LinAlgError) as exc:
        (output/'prediction_interval_warning.txt').write_text(str(exc))
    return comparison

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['human','flood'],default='human')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--output-dir',default='outputs')
    p.add_argument('--mode',choices=['quick','standard','full'],default='quick')
    p.add_argument('--k',type=int,default=10)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--csv-path')
    p.add_argument('--target')
    a=p.parse_args()
    print(run(**vars(a)).to_string(index=False))
