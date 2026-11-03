import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.pipeline import build_pipeline, selected_features
from src.evaluation import metrics, ols_prediction_intervals

def test_all_methods_fit():
    rng=np.random.default_rng(42)
    X=pd.DataFrame(rng.normal(size=(90,5)),columns=list('abcde'))
    X['cat']=np.where(X.a>0,'x','y')
    y=2*X.a-3*X.b+rng.normal(size=90)*0.1
    for method in ['baseline','kbest','rfe','sfs','lasso','permutation']:
        p=build_pipeline(X,method=method,k=3).fit(X,y)
        assert len(p.predict(X.iloc[:5]))==5
        assert len(selected_features(p))>=1

def test_intervals_and_metrics():
    rng=np.random.default_rng(3)
    X=pd.DataFrame(rng.normal(size=(100,3)),columns=['a','b','c'])
    y=2*X.a+X.b+rng.normal(size=100)
    p=build_pipeline(X).fit(X.iloc[:80],y.iloc[:80])
    intervals=ols_prediction_intervals(p,X.iloc[:80],y.iloc[:80],X.iloc[80:])
    assert len(intervals)==20
    assert (intervals.prediction_lower < intervals.prediction_upper).all()
    assert metrics(y.iloc[80:],p.predict(X.iloc[80:]),3)['RMSE']>0
