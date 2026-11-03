# HW2：ML Regression Lab

以 **Streamlit、Scikit-learn 與 Statsmodels** 建立的互動式多元線性回歸專案，依 CRISP-DM 六階段完成資料探索、前處理、特徵選擇、模型比較、評估與新資料預測。提供繁體中文操作提示，也可透過命令列執行實驗與輸出圖表。

## 功能總覽

- 資料來源：Human–AI Interaction、Flood Prediction、公開 HTTPS CSV URL、上傳 CSV。
- EDA：欄位型態、缺失值、描述統計、Pearson 相關矩陣及目標分布。
- 前處理：數值中位數補值與標準化；類別眾數補值與 One-Hot Encoding。
- 比較全特徵線性回歸及五種特徵選擇方法；以訓練集交叉驗證選模型。
- 評估 MAE、MSE、RMSE、R²、Adjusted R²，提供預測散點圖、殘差圖與條件式 95% OLS 預測區間。
- 網頁另提供 DummyRegressor 平均值基準、CV 標準差、選中特徵及回歸係數。
- 探索性診斷：RESET、Durbin–Watson、Breusch–Pagan、Jarque–Bera、VIF 與 Q-Q 圖。
- 下載比較表、預測、特徵清單、診斷結果；產生 Flood Kaggle 提交檔。

## 快速開始

建議使用 Python 3.12；此版本建議不代表已驗證所有 Python 版本。請在專案根目錄執行：

```bash
git clone https://github.com/NCHU-Mei/1151104-HW2.git
cd 1151104-HW2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows 啟用虛擬環境可改用 `.venv\Scripts\activate`。啟動後開啟終端顯示的網址，預設為 <http://localhost:8501>。如埠號被占用：

```bash
streamlit run app.py --server.port 8502
```

主要依賴為 NumPy、Pandas、Scikit-learn、SciPy、Statsmodels、Matplotlib、Plotly、Streamlit、Requests 與 Pytest。版本範圍見 [requirements.txt](requirements.txt)；目前沒有鎖定所有套件的精確版本。

## 資料集

| 檔案 | 資料列數（不含標題） | 用途 |
| --- | ---: | --- |
| [Human-AI Interaction.csv](data/Human-AI%20Interaction.csv) | 10,100 | 20 個候選欄位，預測 `latent_trust_score` |
| [train.csv](data/train.csv) | 1,117,957 | Flood 有標籤訓練資料；20 個特徵、`id` 與 `FloodProbability` |
| [test.csv](data/test.csv) | 745,305 | Flood 官方無標籤測試資料，用於產生提交檔 |
| [sample_submission.csv](data/sample_submission.csv) | 745,305 | 提交格式與 ID 順序範本 |

Flood 資料對應 Kaggle [Regression with a Flood Prediction Dataset（Playground Series S4E5）](https://www.kaggle.com/competitions/playground-series-s4e5)。Human–AI 資料的既有文件標示來源為 Kaggle，但專案未附原始資料頁與授權資訊，仍需補上正式來源。資料的使用及再散布應遵守原始來源條款。

四份 CSV 合計約 106.7 MB，單檔皆小於 GitHub 的 100 MiB 限制；本專案直接納入 CSV。Flood 的 `test.csv` 沒有目標欄位，不能用來計算評估指標。

## 網頁操作

1. 左側選擇資料集、目標欄位並排除 ID。自訂資料至少需 30 筆有效樣本、2 個特徵與 1 個數值特徵；目標欄位會轉為數值並移除無效列。
2. 第一次建議選 Human–AI Interaction，使用預設五種方法、`k=10`、訓練上限 2,000、5-fold、Random State 42。
3. 按「開始分析」，依序查看總覽、商業理解、資料理解、資料準備、建模、評估、部署預測及假設檢驗。
4. 依 `CV_RMSE` 選擇模型，再查看獨立 holdout 表現。RMSE／MAE 越低越好；R² 越高通常越好。切換資料或參數後需重新分析。
5. 部署頁可上傳含所需特徵的新 CSV，下載預測結果；假設檢驗頁可另外執行並下載診斷。

網頁資料讀取可選 10,000、100,000 或全部資料。Flood 部分讀取會掃描全檔後，以固定種子抽樣；Human–AI 部分讀取則取前 N 列。讀取筆數與訓練上限是不同設定：前者決定載入資料，後者在 80/20 切分後限制訓練部分。網頁訓練上限可選 2,000、10,000、100,000 或全部。

公開 URL 與上傳 CSV 上限為 15 MB；URL 僅支援 HTTPS，禁止帶帳密、內網位址與重新導向。詳細操作見 [中文操作指南](中文操作指南.md)。

## 模型與實驗流程

| 方法代碼 | 方法 | 行為 |
| --- | --- | --- |
| `baseline` | LinearRegression | 使用全部編碼後特徵 |
| `kbest` | SelectKBest | 依 `f_regression` 保留 k 個特徵 |
| `rfe` | Recursive Feature Elimination | 以線性回歸遞迴移除特徵 |
| `sfs` | Forward Sequential Feature Selection | 以前向選擇與內部 3-fold 選出 k 個特徵 |
| `lasso` | LassoCV + SelectFromModel | 以 LassoCV 選特徵後再擬合 LinearRegression，特徵數不由 k 固定 |
| `permutation` | Permutation Importance | 在當次訓練折內計算重要性並保留前 k 個特徵 |

有標籤資料先以固定種子切為 80% 訓練與 20% holdout，再視設定抽樣訓練集。外層 KFold 只使用訓練資料，補值、編碼、標準化及特徵選擇均在 Pipeline 中隨各折擬合。網頁可選 3-fold 或 5-fold；CLI 預設 5-fold。結果依訓練集 CV RMSE 排序。

`k` 指編碼後特徵數，不是原始欄位數；自訂資料的特徵數較少時需調小 k，SFS 尤其需要 k 小於可用特徵數。Permutation 重要性使用訓練折本身計算，仍可能偏向擬合訓練資料。SFS／Lasso 的內部 CV 接收外層訓練折已前處理的資料，因此其內部驗證並非完全獨立的前處理流程。

## 命令列實驗與繪圖

先以較快的五種方法執行 Human–AI 實驗：

```bash
python run_experiment.py --dataset human --mode quick --k 10 --methods baseline kbest rfe lasso permutation
python plot_results.py --output-dir outputs
```

其他範例：

```bash
# Flood：quick 模式僅限制切分後的訓練集，仍會讀入完整 CSV
python run_experiment.py --dataset flood --mode quick --methods baseline kbest

# 自訂有標籤 CSV
python run_experiment.py --csv-path your_data.csv --target target_column --methods baseline

# 指定輸出位置
python run_experiment.py --dataset human --output-dir outputs/human --methods baseline
python plot_results.py --output-dir outputs/human
```

CLI 的 `quick`、`standard`、`full` 分別代表訓練集上限 10,000、100,000 與不限；預設會執行全部六種方法，包含較耗時的 SFS。可用參數包含 `--dataset`、`--data-dir`、`--output-dir`、`--mode`、`--k`、`--methods`、`--csv-path`、`--target`。

| 輸出 | 內容 |
| --- | --- |
| `model_comparison.csv` | holdout 指標、訓練集 CV 指標及執行時間 |
| `holdout_predictions.csv` | 真實值及各方法預測 |
| `selected_features.json` | 各方法的編碼後特徵名稱 |
| `prediction_intervals.csv` | CV 最佳方法前最多 500 筆 holdout 的 OLS 區間 |
| `rmse_comparison.png` | 各方法 holdout RMSE 圖 |
| `actual_vs_predicted.png`、`residuals.png` | CV 最佳方法的預測及殘差圖 |
| `prediction_intervals.png` | 區間 CSV 存在時產生的 95% 預測區間圖 |

無法估計區間時可能輸出 `prediction_interval_warning.txt`。繪圖程式使用 CLI 輸出格式；網頁下載的比較表欄位不同，不宜直接混用。既有結果存於 [outputs](outputs)，屬於可重新產生的範例，不能視為任何新設定的驗證結果。

## Flood 提交檔

1. 選 Flood Prediction，以有標籤資料完成分析。
2. 至部署頁選模型，產生並下載 `submission.csv`。
3. 預設使用先前訓練子集擬合的模型；勾選重新訓練時，只會使用目前已載入的有標籤資料。若需要全量重新擬合，須先將讀取筆數設為「全部資料」。
4. 程式確認所需特徵、有限數值、範本欄位、列數與 ID 順序，輸出 `id,FloodProbability`。

線性回歸預測不會截斷至 [0,1]。此功能僅產生下載檔，不會自動向 Kaggle 提交。

## 專案結構

```text
.
├── app.py                  # Streamlit 主介面、載入、訓練與互動圖表
├── run_experiment.py       # CLI 實驗與 CSV／JSON 輸出
├── plot_results.py         # 從 CLI 結果產生 PNG 圖表
├── requirements.txt        # 套件版本範圍
├── src/
│   ├── data.py             # CLI 資料載入與目標欄位檢查
│   ├── pipeline.py         # 前處理、六種方法與特徵名稱
│   ├── evaluation.py       # 指標、OLS 區間與診斷工具
│   └── phase4.py           # 擴充統計診斷與 Flood 提交驗證
├── tests/                  # Pipeline、區間、診斷與提交格式測試
├── data/                   # 兩個問題的原始 CSV 與提交範本
├── outputs/                # 實驗結果與圖表
├── plan.md                 # 原始規劃（不代表功能全數完成）
├── workflow.md             # 流程說明
├── 中文操作指南.md          # 操作與指標判讀
├── PHASE4_TESTING.md        # 歷史手動驗收清單
└── PHASE4_1_CHANGES.md      # 歷史變更與限制
```

## 測試與部署

```bash
python -m pytest -q
```

目前原始碼有 6 個測試函式，涵蓋六種方法基本擬合、訓練補值不使用 holdout、評估指標、預測區間、診斷與提交格式。自動測試不等於已完成瀏覽器驗收；手動項目見 [PHASE4_TESTING.md](PHASE4_TESTING.md)，其中舊版「四個測試」紀錄屬歷史資訊。

可於 Streamlit Community Cloud 連接此 GitHub 儲存庫，入口指定 `app.py`，由 `requirements.txt` 安裝依賴。必須確保四份 CSV 可在部署環境存取；全量 Flood、密集 One-Hot 編碼與 SFS 可能超出免費環境資源，建議先以小樣本及 Baseline 驗證。專案未附已部署網站連結。

## 判讀限制與待補資料

- OLS 區間以已選中特徵及線性模型假設為條件，沒有校正特徵選擇不確定性；平均值信賴區間與單筆觀測預測區間不同。
- 診斷以最多 5,000 筆訓練資料抽樣後重新擬合 OLS，屬探索性結果；VIF 僅在編碼後特徵不超過 40 個時計算。
- 檢定 p 值大於 0.05 不代表假設成立。Durbin–Watson 需有合理觀測順序；相關性與回歸係數不代表因果。
- 不應反覆依 holdout 成績調參。Adjusted R² 在樣本不足時會回傳 NaN。
- 對外部署若開放 URL 載入，建議加上允許主機清單或經驗證的代理；目前 DNS 檢查不能完整防止 DNS rebinding。
- 原規劃提及 NotebookLM 研究摘要、來源引用與 GPT 協助揭露，但目前未附正式摘要、對話 PDF 或完整協助紀錄，應由作者依實際使用情況補齊。
- 專案目前沒有 LICENSE；不能據此假定程式或資料可任意再散布。
