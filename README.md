# AICompetition

## 提醒： 為了避免專案間的套件版本衝突，請務必使用虛擬環境！

簡易步驟：

啟用： source venv/bin/activate (提示符會多 (venv))

(安裝套件/跑程式)

離開： deactivate


## 爬蟲步驟：

cd /c/Users/翁匡霆/AICompetition(改成自己的)

pip install -r requirements.txt

python main.py


## 使用者需求搜尋熱度時間序列預測：

cd ~/timeseries_UserRequirements

最終結果請使用第一點的json檔案

1. AutoTs自動預測 & 訓練集為5年週資料
python timeseries_UR_autoML.py

輸出為5個維度歷史數據及預測8週的autots_forecast_all_weeks.json


2. 單純LSTM & 訓練集為90日單日資料

python timeseries.py

輸出為5個維度歷史數據及預測30天的lstm_forecast_all.json

## 技術趨勢搜尋熱度時間序列預測：

cd ~/timeseries_TechTrends

python timeseries_TT.py

輸出為氮化鎵充電器、磁吸行動電源、pd 行動電源、type c行動電源之歷史數據及預測24週（半年）的autots_forecast_TechTrends.json

## 1：使用2和4的結果以及爬蟲的評論標題當作promt串api生成
cd AIInsightServer
npm install
建立 .env（每位組員必做）

在 AIInsightServer 之內，手動建立檔案：
.env  (不要push到github，很重要)


內容貼上同一組 API Key（）：GEMINI_API_KEY="你的APIKEY" (Conrad保管)

node insight.js
到瀏覽器貼上http://localhost:8000/api/insight