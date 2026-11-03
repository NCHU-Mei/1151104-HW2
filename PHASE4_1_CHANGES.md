# Phase 4.1 — 更新與驗證

## 已修改
- 使用整份 DataFrame 內容 SHA-256 指紋識別模型結果，防止同筆數不同資料誤用舊快取。
- Flood 的 10k/100k 讀取模式使用全檔隨機優先級抽樣，而非取前 N 筆；抽樣可重現。
- 模型比較增加 `CV_RMSE_STD`、`CV_R2_STD`，CV RMSE 顯示誤差棒。
- 新增 DummyRegressor (mean) 的 holdout RMSE 參考基準。
- 新增每個方法的選中特徵與標準化／編碼後係數表。
- 診斷頁新增中文檢驗判讀、VIF 分級與前 20 項 VIF 圖。
- Submission 重新訓練選項更正描述：只對當前載入資料重新訓練，非必然完整 Kaggle train.csv。

## 執行
```bash
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py --server.port 8502
```

## 重要限制
- 本次環境無 Streamlit 套件，已執行 Python 語法檢查及現有 pytest (4 passed)，**尚未做瀏覽器實測**。
- 95% OLS 預測區間以固定選中特徵為條件，未校正特徵選擇不確定性。
- VIF 與統計檢驗是探索性工具，不能單靠 p-value 判定模型有效。
- 若 Kaggle 提交需要全量訓練，先在左側選「全部資料」，再勾選重新訓練。
- 本次沒有重構整個應用；沿用 Phase 4 原始架構及依賴。
