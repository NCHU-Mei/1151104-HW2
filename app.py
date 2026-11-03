"""HW2 interactive CRISP-DM regression lab. Run: streamlit run app.py"""
from __future__ import annotations
import io
import json
import time
import hashlib
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from src.pipeline import build_pipeline, selected_features
from src.evaluation import metrics, ols_prediction_intervals
from src.phase4 import regression_diagnostics, make_flood_submission

st.set_page_config(page_title="ML Regression Lab | HW2", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1400px; padding-top: 1.5rem;}
h1,h2,h3 {letter-spacing: -.025em;}
[data-testid="stMetric"] {background: rgba(120,130,150,.075); border:1px solid rgba(120,130,150,.15); padding: 14px; border-radius: 12px;}
[data-testid="stSidebar"] {border-right:1px solid rgba(120,130,150,.12);}
</style>""", unsafe_allow_html=True)
METHODS = ["baseline", "kbest", "rfe", "sfs", "lasso", "permutation"]
LABELS = {"baseline":"Baseline", "kbest":"SelectKBest", "rfe":"RFE", "sfs":"Sequential FS", "lasso":"Lasso", "permutation":"Permutation"}
TARGETS = {"Human–AI Interaction":"latent_trust_score", "Flood Prediction":"FloodProbability"}
FILES = {"Human–AI Interaction":"data/Human-AI Interaction.csv", "Flood Prediction":"data/train.csv"}

@st.cache_data(show_spinner=False)
def load_builtin(name, max_rows):
    if max_rows is None:
        return pd.read_csv(FILES[name])
    # Sample uniformly across the file using stable random priorities per row.
    if name == "Flood Prediction":
        rng = np.random.default_rng(42)
        parts = []
        seen = 0
        for chunk in pd.read_csv(FILES[name], chunksize=50000):
            priorities = rng.random(len(chunk))
            chunk = chunk.copy()
            chunk["__sample_priority__"] = priorities
            parts.append(chunk.nsmallest(min(max_rows, len(chunk)), "__sample_priority__"))
            seen += len(chunk)
        out = pd.concat(parts, ignore_index=True).nsmallest(max_rows, "__sample_priority__")
        return out.drop(columns="__sample_priority__").reset_index(drop=True)
    return pd.read_csv(FILES[name], nrows=max_rows)

@st.cache_data(show_spinner=False, ttl=3600)
def load_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("僅支援公開 HTTPS CSV URL，且不可包含帳號密碼。")
    # Avoid requesting arbitrary internal hosts in a hosted deployment.
    import ipaddress
    import socket
    host = parsed.hostname
    if host.lower() in ("localhost",) or host.endswith((".local", ".internal")):
        raise ValueError("不允許內部網路位址。")
    try:
        addresses = socket.getaddrinfo(host, None)
        if any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ValueError("僅允許公開網際網路主機。")
    except socket.gaierror as e:
        raise ValueError("無法解析 CSV 主機名稱。") from e
    # Redirects disabled to prevent redirects to private services.
    with requests.get(url, timeout=(5, 25), stream=True, allow_redirects=False) as r:
        r.raise_for_status()
        chunks=[]; size=0
        for chunk in r.iter_content(chunk_size=65536):
            size+=len(chunk)
            if size > 15*1024*1024:
                raise ValueError("遠端 CSV 超過 15 MB 限制，請改用較小資料集。")
            chunks.append(chunk)
    return pd.read_csv(io.BytesIO(b"".join(chunks)))

def prepare(df, target, excluded):
    if target not in df.columns:
        raise ValueError("Target 欄位不存在。")
    df=df.copy()
    df[target]=pd.to_numeric(df[target], errors="coerce")
    df=df.dropna(subset=[target])
    X=df.drop(columns=[target]+[c for c in excluded if c!=target and c in df.columns])
    X=X.drop(columns=X.columns[X.isna().all()])
    if len(X)<30 or X.shape[1]<2:
        raise ValueError("需要至少 30 筆有效資料與 2 個特徵。")
    if X.select_dtypes(include="number").shape[1]==0:
        raise ValueError("至少需要一個數值特徵。")
    if not X.select_dtypes(exclude="number").empty and X.select_dtypes(exclude="number").nunique(dropna=True).max() > 150:
        raise ValueError("類別欄位有過多不同值，請排除 ID 或高基數文字欄位。")
    return X, df[target]

def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable content-aware identity including columns, index and cell values."""
    digest = hashlib.sha256()
    digest.update(repr((tuple(df.columns), tuple(map(str, df.dtypes)), df.shape)).encode())
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()


def train_models(X, y, methods, k, cap, seed, cv_folds, progress):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed)
    if cap and len(Xtr)>cap:
        idx=Xtr.sample(n=cap,random_state=seed).index
        Xtr,ytr=Xtr.loc[idx],ytr.loc[idx]
    rows=[]; preds={}; names={}; fitted={}
    dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    dummy_rmse = float(np.sqrt(mean_squared_error(yte, dummy.predict(Xte))))
    for i,method in enumerate(methods):
        progress.progress(i/len(methods),text=f"訓練 {LABELS[method]} · {i+1}/{len(methods)}")
        start=time.perf_counter()
        pipe=build_pipeline(Xtr,method=method,k=k,random_state=seed)
        cv=KFold(n_splits=cv_folds,shuffle=True,random_state=seed)
        scores=cross_validate(pipe,Xtr,ytr,cv=cv,scoring={"r2":"r2","rmse":"neg_root_mean_squared_error","mae":"neg_mean_absolute_error"},n_jobs=1)
        pipe.fit(Xtr,ytr)
        yp=pipe.predict(Xte)
        feature_names=selected_features(pipe)
        rows.append({"Method":LABELS[method],"method":method,**metrics(yte,yp,len(feature_names)),
                     "CV_R2":float(np.mean(scores["test_r2"])),"CV_RMSE":float(-np.mean(scores["test_rmse"])),
                     "CV_MAE":float(-np.mean(scores["test_mae"])),"CV_RMSE_STD":float(np.std(-scores["test_rmse"])),"CV_R2_STD":float(np.std(scores["test_r2"])),"Seconds":round(time.perf_counter()-start,2)})
        preds[method]=yp;names[method]=feature_names;fitted[method]=pipe
    progress.progress(1.0,text="模型訓練完成")
    return {"comparison":pd.DataFrame(rows).sort_values("CV_RMSE"),"predictions":preds,"selected":names,
            "models":fitted,"dummy_rmse":dummy_rmse,"Xtr":Xtr,"Xte":Xte,"ytr":ytr,"yte":yte}

st.title("📈 ML Regression Lab")
st.caption("HW2 · Multiple Linear Regression · CRISP-DM · Scikit-learn")
with st.expander("📖 第一次使用？點此展開中文操作教學", expanded=True):
    st.markdown("""
    **建議操作順序：**
    1. 左側選擇 **Human–AI Interaction**，確認目標欄位為 `latent_trust_score`。
    2. 勾選 **Baseline、SelectKBest、RFE、Lasso、Permutation**；特徵數 k 設為 10。
    3. 訓練上限先選 **2,000 筆**，交叉驗證選 **5-Fold**，Random State 設為 **42**。
    4. 按左側 **▶ 開始分析（Run Analysis）**，等待訓練完成。
    5. 依序查看 **④ 建模 → ⑤ 評估 → 🔬 假設檢驗**，最後下載 CSV。

    **判讀原則：** CV RMSE 越小越好、R² 越高越好；先依交叉驗證選模型，再用 Holdout Test 做最終評估。
    **注意：** 此網站只在你的本機執行；若沒有完成訓練，模型頁面不會有結果。
    """)

with st.sidebar:
    st.header("① 選擇資料集")
    source=st.selectbox("Problem / Dataset",["Human–AI Interaction","Flood Prediction","Custom CSV URL","Upload CSV"])
    df=None
    try:
        if source in FILES:
            limit=st.selectbox("讀取資料筆數",[10000,100000,None],format_func=lambda n:"全部資料" if n is None else f"前 {n:,} 筆",index=0 if source=="Flood Prediction" else 2)
            df=load_builtin(source,limit)
        elif source=="Custom CSV URL":
            url=st.text_input("公開 HTTPS CSV URL",placeholder="https://raw.githubusercontent.com/.../dataset.csv")
            if url and st.button("載入遠端 CSV"):
                st.session_state["remote_url"]=url
            if st.session_state.get("remote_url")==url and url:
                df=load_url(url)
        else:
            upload=st.file_uploader("上傳 CSV（上限 15 MB）",type="csv")
            if upload:
                if upload.size>15*1024*1024: raise ValueError("檔案超過 15 MB。")
                df=pd.read_csv(upload)
    except Exception as e:
        st.error(f"載入失敗：{e}")
    if df is not None:
        target=st.selectbox("預測目標 Target",list(df.columns),index=list(df.columns).index(TARGETS[source]) if source in TARGETS else len(df.columns)-1)
        default_excluded=[c for c in df if c.lower() in ("id","index","unnamed: 0") and c!=target]
        excluded=st.multiselect("排除非特徵欄位",[c for c in df if c!=target],default=default_excluded)
        st.divider()
        st.header("② 模型設定")
        selected=st.multiselect("Feature Selection",METHODS,default=["baseline","kbest","rfe","lasso","permutation"],format_func=lambda x:LABELS[x])
        k=st.slider("選擇特徵數 k",min_value=2,max_value=20,value=10)
        mode=st.selectbox("訓練資料上限",[2000,10000,100000,None],format_func=lambda n:"Full" if n is None else f"{n:,} rows",index=1)
        folds=st.selectbox("交叉驗證折數",[3,5],index=1)
        seed=st.number_input("Random State",value=42,step=1)
        st.caption("💡 k 代表編碼後保留的特徵數；先用 10 測試。SFS 計算量大，建議先用 2,000 筆。")
        with st.expander("❓ 左側參數中文說明"):
            st.markdown("""
            - **Feature Selection**：選擇要比較的特徵篩選方法；Baseline 不刪除特徵。
            - **選擇特徵數 k**：供 SelectKBest、RFE、SFS、Permutation 等方法使用。
            - **訓練資料上限**：限制模型訓練資料量，不等於整份資料的總筆數。
            - **交叉驗證折數**：5-Fold 將訓練資料分成 5 份輪流驗證。
            - **Random State**：固定亂數種子，便於重現結果。
            """)
        go_train=st.button("▶ 開始分析（Run Analysis）",type="primary",use_container_width=True,disabled=not selected)
    else:
        go_train=False

if df is None:
    st.info("請從左側選擇內建資料集，或輸入公開 CSV URL / 上傳 CSV。")
    st.stop()

try:
    X,y=prepare(df,target,excluded)
except Exception as e:
    st.error(str(e));st.stop()

signature=(source,target,tuple(excluded),dataset_fingerprint(df),tuple(selected),k,mode,folds,int(seed))
if go_train:
    try:
        with st.spinner("執行 5-Fold CV 與獨立測試集評估中…"):
            p=st.progress(0)
            result=train_models(X,y,selected,k,mode,int(seed),folds,p)
            st.session_state["result"]=result
            st.session_state["signature"]=signature
    except Exception as e:
        st.error(f"訓練失敗：{type(e).__name__}: {e}")

result=st.session_state.get("result") if st.session_state.get("signature")==signature else None
pages=st.tabs(["📌 總覽","① 商業理解","② 資料理解","③ 資料準備","④ 建模","⑤ 評估","⑥ 部署預測","🔬 假設檢驗"])
with pages[0]:
    a,b,c,d=st.columns(4)
    a.metric("有效樣本數",f"{len(X):,}");b.metric("原始特徵數",len(X.columns));c.metric("數值特徵",len(X.select_dtypes(include="number").columns));d.metric("類別特徵",len(X.select_dtypes(exclude="number").columns))
    st.dataframe(df.head(30),use_container_width=True)
    st.info("請先在左側設定參數，按「▶ 開始分析」，再查看建模與評估頁面。")
    st.markdown("**本頁怎麼看？** 有效樣本數為 Target 非缺失後的資料筆數；原始特徵數不包含 Target 與排除欄位。")
with pages[1]:
    st.subheader("Business Understanding｜商業理解")
    st.info("操作說明：先確認預測目標是否為連續數值，以及這個預測對實際問題有何價值。")
    st.write(f"研究問題：使用 {len(X.columns)} 個候選特徵，預測連續目標 **{target}**。")
    st.write("目標：比較全特徵線性回歸與五種特徵選擇方法，在獨立測試集上的泛化表現。")
    st.write("來源："+source+("（使用者提供）" if source not in FILES else "（Kaggle 資料集）"))
with pages[2]:
    st.subheader("Data Understanding｜資料理解")
    st.info("操作說明：檢查每個欄位的型態與缺失值；相關係數接近 ±1 代表線性關係較強，但不代表因果關係。")
    st.dataframe(df.dtypes.rename("dtype").to_frame().join(df.isna().sum().rename("missing")),use_container_width=True)
    st.dataframe(df.describe(include="all").T,use_container_width=True)
    numeric=df.select_dtypes(include="number")
    if len(numeric.columns)>1:
        fig=px.imshow(numeric.corr(),title="Pearson Correlation",aspect="auto",color_continuous_scale="RdBu_r",zmin=-1,zmax=1)
        st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.histogram(df,x=target,title="Target Distribution"),use_container_width=True)
with pages[3]:
    st.subheader("Data Preparation｜資料準備")
    st.info("操作說明：確認 Target 缺失資料已排除，類別特徵已編碼，且所有補值與特徵選擇只在訓練資料中擬合。")
    st.write("1. 刪除 Target 缺失列；2. 排除 ID；3. 訓練 / 測試集 80/20；4. 訓練折內補值、標準化、類別 One-Hot 編碼。")
    st.write("所有前處理及特徵選擇都在 Scikit-learn Pipeline 內執行，避免測試集資訊洩漏。")
    st.write("原始欄位數與 One-Hot 後的欄位數可能不同；Feature Selection 的 k 是編碼後欄位數。")
with pages[4]:
    st.subheader("Modeling｜建模與 5-Fold 交叉驗證")
    st.info("📘 **怎麼看結果：** 先看 CV_RMSE（越低越好），再看 CV_RMSE_STD（越低代表各折波動較小）；Seconds 是執行秒數，n_features 是模型使用的編碼後特徵數。**不要依 Holdout Test 排名反覆挑模型。**")
    if result is None: st.info("請先執行模型。")
    else:
        comp=result["comparison"]
        st.info(f"DummyRegressor（預測訓練平均值）Holdout RMSE：{result['dummy_rmse']:.4f}。模型應優於此基準。")
        st.caption("依訓練集 CV RMSE 選擇模型；Holdout 指標只用於最終評估，請勿反覆依測試成績調參。")
        st.dataframe(comp,use_container_width=True)
        fig_cv=px.bar(comp,x="Method",y="CV_RMSE",error_y="CV_RMSE_STD",title="5-Fold CV RMSE ± 1 SD（越低越好）",color="Method")
        st.plotly_chart(fig_cv,use_container_width=True)
        st.caption("CV 標準差衡量不同訓練折的波動，並非模型差異的顯著性檢定。")
        st.download_button("下載模型比較 CSV",comp.to_csv(index=False).encode("utf-8-sig"),"model_comparison.csv","text/csv")
        with st.expander("查看各模型選中特徵及回歸係數"):
            selected_method=st.selectbox("特徵清單模型",list(result["models"]),format_func=lambda m:LABELS[m])
            m=result["models"][selected_method]
            coef=pd.DataFrame({"Feature":result["selected"][selected_method],"Coefficient (scaled/encoded)":m.named_steps["model"].coef_})
            st.dataframe(coef.sort_values("Coefficient (scaled/encoded)",key=abs,ascending=False),use_container_width=True)
            st.caption("數值特徵經標準化、類別經編碼，係數不可直接視為原始量尺上的因果效果。")
        st.download_button("下載選取特徵 JSON",json.dumps(result["selected"],ensure_ascii=False,indent=2).encode(),"selected_features.json","application/json")
with pages[5]:
    st.subheader("Evaluation｜獨立測試集評估")
    st.info("📘 **怎麼看結果：** R² 越接近 1 通常越好；RMSE、MAE 越小越好。Actual vs Predicted 的點越接近虛線 y=x，預測越準。殘差若呈現規律形狀，需檢查模型設定。")
    if result is None: st.info("請先執行模型。")
    else:
        comp=result["comparison"]
        best=comp.iloc[0]["method"]
        choice=st.selectbox("檢視模型",list(result["predictions"]),index=list(result["predictions"]).index(best),format_func=lambda x:LABELS[x])
        yp=result["predictions"][choice]; actual=result["yte"].to_numpy()
        a,b,c=st.columns(3)
        row=comp.set_index("method").loc[choice]
        a.metric("R²",f"{row['R2']:.4f}");b.metric("RMSE",f"{row['RMSE']:.5f}");c.metric("MAE",f"{row['MAE']:.5f}")
        view=pd.DataFrame({"Actual":actual,"Predicted":yp,"Residual":actual-yp}).sample(n=min(3000,len(actual)),random_state=42)
        fig=px.scatter(view,x="Actual",y="Predicted",title="Actual vs Predicted",opacity=.55)
        lo=min(view.Actual.min(),view.Predicted.min());hi=max(view.Actual.max(),view.Predicted.max())
        fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Ideal y=x",line=dict(dash="dash")))
        st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(px.scatter(view,x="Predicted",y="Residual",title="Residual vs Fitted"),use_container_width=True)
        st.caption("95% Prediction Interval：固定已選特徵設計下的條件式 OLS 區間，未校正特徵選擇的不確定性。")
        if st.button("計算前 200 筆的 95% 預測區間"):
            try:
                with st.spinner("計算 OLS prediction interval…"):
                    pi=ols_prediction_intervals(result["models"][choice],result["Xtr"],result["ytr"],result["Xte"].iloc[:200])
                pi.insert(0,"Actual",actual[:len(pi)])
                pi.insert(0,"Index",np.arange(len(pi)))
                st.dataframe(pi.head(20),use_container_width=True)
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=pi.Index,y=pi.prediction_upper,line=dict(width=0),showlegend=False))
                fig.add_trace(go.Scatter(x=pi.Index,y=pi.prediction_lower,fill="tonexty",name="95% prediction interval",line=dict(width=0)))
                fig.add_trace(go.Scatter(x=pi.Index,y=pi.predicted,name="Predicted"))
                fig.add_trace(go.Scatter(x=pi.Index,y=pi.Actual,mode="markers",name="Actual"))
                st.plotly_chart(fig,use_container_width=True)
                st.download_button("下載預測區間 CSV",pi.to_csv(index=False).encode("utf-8-sig"),"prediction_intervals.csv","text/csv")
            except Exception as e: st.warning(f"此模型無法計算 OLS 區間：{e}")
        st.download_button("下載測試集預測 CSV",pd.DataFrame({"actual":actual,**result["predictions"]}).to_csv(index=False).encode("utf-8-sig"),"holdout_predictions.csv","text/csv")
with pages[6]:
    st.subheader("Deployment｜新資料預測")
    st.info("操作說明：先完成模型訓練，再上傳具有相同輸入特徵欄位的 CSV。Flood 的 Kaggle 官方 test.csv 不含答案，只能產生 submission.csv，不能計算 R²。")
    if result is None: st.info("請先執行模型。")
    else:
        best=result["comparison"].iloc[0]["method"]
        st.write(f"目前依 CV RMSE 選出的模型：**{LABELS[best]}**")
        st.caption("使用原始資料欄位輸入預測；類別值請與訓練資料一致。")
        uploaded=st.file_uploader("上傳待預測 CSV（欄位須與訓練資料相同）",type="csv",key="predict_upload")
        if uploaded:
            try:
                future=pd.read_csv(uploaded)
                missing=[c for c in X.columns if c not in future.columns]
                if missing: st.error(f"缺少欄位：{missing}")
                else:
                    predicted=result["models"][best].predict(future[X.columns])
                    out=future.copy();out[f"predicted_{target}"]=predicted
                    st.dataframe(out.head(20),use_container_width=True)
                    st.download_button("下載預測 CSV",out.to_csv(index=False).encode("utf-8-sig"),"predictions.csv","text/csv")
            except Exception as e: st.error(f"預測失敗：{e}")
        if source=="Flood Prediction":
            st.divider()
            st.subheader("Kaggle Official Submission")
            st.caption("使用官方 test.csv 產生 id,FloodProbability；不會以官方無標籤測試集計算 R²。")
            use_full=st.checkbox("以目前已載入的有標籤資料重新訓練（若只載入部分資料則非全量）",value=False)
            if st.button("產生 Kaggle Submission",key="make_submission"):
                try:
                    with st.spinner("讀取官方測試集、執行預測…"):
                        official=pd.read_csv("data/test.csv")
                        template=pd.read_csv("data/sample_submission.csv")
                        if use_full:
                            from sklearn.base import clone
                            final_model=clone(result["models"][best]).fit(X,y)
                        else:
                            final_model=result["models"][best]
                        submission=make_flood_submission(final_model,official,template)
                        st.session_state["submission"]=(signature,use_full,submission)
                except Exception as e:
                    st.error(f"Submission 失敗：{e}")
            saved=st.session_state.get("submission")
            if saved and saved[:2]==(signature,use_full):
                submission=saved[2]
                st.success(f"產生 {len(submission):,} 筆預測")
                st.dataframe(submission.head(10),use_container_width=True)
                st.download_button("下載 submission.csv",submission.to_csv(index=False).encode("utf-8"),"submission.csv","text/csv",key="download_submission")

with pages[7]:
    st.subheader("Diagnostics｜五大回歸假設與 VIF")
    st.info("📘 **判讀方式：** RESET 檢查模型設定、Durbin–Watson 接近 2 通常無明顯一階自相關（須有合理資料順序）、Breusch–Pagan 檢查等變異、Jarque–Bera 檢查殘差常態；VIF ≥ 10 建議檢查共線性。p ≥ 0.05 只代表未發現顯著違反，並非證明假設成立。")
    st.caption("診斷使用已訓練模型的訓練資料（最多抽樣 5,000 筆），屬探索性統計檢驗；不是獨立測試集推論。")
    if result is None:
        st.info("請先執行模型。")
    else:
        diagnostic_method=st.selectbox("選擇診斷模型",list(result["models"]),format_func=lambda m:LABELS[m],key="diag_model")
        if st.button("執行五大假設檢驗",key="diag_run"):
            try:
                with st.spinner("計算統計檢驗與 VIF…"):
                    summary,vif,residual=regression_diagnostics(result["models"][diagnostic_method],result["Xtr"],result["ytr"])
                st.session_state["diagnostic_cache"]=(signature,diagnostic_method,summary,vif,residual)
            except Exception as e:
                st.error(f"診斷失敗：{e}")
        cache=st.session_state.get("diagnostic_cache")
        if cache is not None and cache[:2]==(signature,diagnostic_method):
            _,_,summary,vif,residual=cache
            st.dataframe(pd.DataFrame([{"Test":k,"Value":v} for k,v in summary.items()]),use_container_width=True)
            st.caption("RESET：模型設定；DW：一階自相關（資料順序有意義時）；BP：等變異；JB：常態；VIF：共線性。p ≥ 0.05 不代表已證明假設成立。")
            st.markdown("**自動判讀（探索性；非正式統計結論）**")
            for key, label in [("Linearity (RESET) p-value","RESET 模型設定"),("Homoscedasticity (Breusch–Pagan) p-value","BP 等變異"),("Normality (Jarque–Bera) p-value","JB 殘差常態")]:
                pv=summary.get(key,float('nan'))
                st.write(f"- {label}：" + ("無法計算" if not np.isfinite(pv) else ("p < 0.05，存在違反假設的統計證據" if pv<0.05 else "p ≥ 0.05，尚未發現顯著違反證據")))
            st.write(f"- Durbin–Watson = {summary['Independence (Durbin–Watson)']:.3f}；接近 2 通常不顯示明顯一階自相關，但觀測順序必須有意義。")
            if summary["Design rank"] < summary["Design columns"]:
                st.warning("設計矩陣秩不足，存在完全共線性；係數及部分檢定結果不穩定。")
            st.plotly_chart(px.scatter(residual,x="Fitted",y="Residual",title="Residual vs Fitted"),use_container_width=True)
            from scipy.stats import probplot
            osm,osr=probplot(residual["Residual"].to_numpy(),dist="norm",fit=False)
            st.plotly_chart(px.scatter(x=osm,y=osr,labels={"x":"Theoretical quantiles","y":"Residual quantiles"},title="Normal Q-Q Plot"),use_container_width=True)
            if not vif.empty:
                vif=vif.sort_values("VIF",ascending=False).copy()
                vif["判讀"] = np.select([vif.VIF>=10,vif.VIF>=5],["高度共線性警訊","需留意"],default="一般")
                st.dataframe(vif,use_container_width=True)
                st.plotly_chart(px.bar(vif.head(20),x="VIF",y="Feature",orientation="h",title="最高 VIF 特徵（排除截距）"),use_container_width=True)
                if (vif.VIF>=10).any(): st.warning("有特徵 VIF ≥ 10，建議檢查 One-Hot 編碼、冗餘欄位及係數穩定性。")
                st.download_button("下載 VIF CSV",vif.to_csv(index=False).encode("utf-8-sig"),"vif.csv","text/csv")
            st.download_button("下載診斷摘要 CSV",pd.DataFrame([summary]).to_csv(index=False).encode("utf-8-sig"),"diagnostics.csv","text/csv")
