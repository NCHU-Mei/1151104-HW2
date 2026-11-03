import pandas as pd
import numpy as np
from src.pipeline import build_pipeline
from src.evaluation import metrics

def test_metrics_reasonable():
    out=metrics([1,2,3,4],[1,2,3,4],1)
    assert out['R2']==1 and out['RMSE']==0

def test_pipeline_does_not_fit_on_holdout():
    X=pd.DataFrame({'a':np.arange(30,dtype=float),'b':np.arange(30,dtype=float)**2})
    y=pd.Series(np.arange(30,dtype=float))
    pipe=build_pipeline(X,'kbest',k=1)
    pipe.fit(X.iloc[:20],y.iloc[:20])
    assert pipe.named_steps['prep'].named_transformers_['num'].named_steps['imputer'].statistics_[0]==9.5
