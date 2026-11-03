import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from src.phase4 import make_flood_submission, regression_diagnostics
from src.pipeline import build_pipeline


def test_submission_columns_and_id_order():
    rng=np.random.default_rng(4)
    train=pd.DataFrame({'a':rng.normal(size=100),'b':rng.normal(size=100)})
    model=LinearRegression().fit(train,train.a+train.b)
    test=pd.DataFrame({'id':[17,3,25],'a':[1,2,3],'b':[2,3,4]})
    template=pd.DataFrame({'id':[17,3,25],'FloodProbability':[0.,0.,0.]})
    result=make_flood_submission(model,test,template)
    assert result.columns.tolist()==['id','FloodProbability']
    assert result.id.tolist()==[17,3,25]
    with pytest.raises(ValueError):
        make_flood_submission(model,test,template.iloc[:2])


def test_diagnostics_smoke():
    rng=np.random.default_rng(9)
    X=pd.DataFrame({'x1':rng.normal(size=120),'x2':rng.normal(size=120)})
    y=2*X.x1+X.x2+rng.normal(size=120)
    model=build_pipeline(X,'baseline').fit(X,y)
    summary,vif,resid=regression_diagnostics(model,X,y)
    assert 'Linearity (RESET) p-value' in summary
    assert len(vif)==2 and len(resid)==120
