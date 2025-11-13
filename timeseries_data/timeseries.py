import pandas as pd
import numpy as np
import json
import glob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =============== 設定參數 ===============
look_back = 30
future_days = 30
output_all = {}

# =============== 定義函式：LSTM 預測 ===============
def lstm_forecast(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')

    # [新增] 在這裡印出讀取到的資料
    print(f"--- Processing file: {csv_path} ---")
    print("top 5 data (df.head()):")
    print(df.head())
    print("\nAll columns read(df.columns):")
    print(df.columns)
    print("--------------------------------------\n")

    df["date"] = pd.to_datetime(df["date"])

    # 平滑處理避免0影響
    df["smooth"] = df["total"].rolling(window=3, min_periods=1).mean()

    data = df[["smooth"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 建立模型
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    # 預測未來
    last_seq = scaled_data[-look_back:]
    preds = []
    for _ in range(future_days):
        X_pred = np.reshape(last_seq, (1, look_back, 1))
        pred = model.predict(X_pred, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], pred).reshape(look_back, 1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=future_days)

    return {
        "historical": df[["date", "total"]].assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "forecast": pd.DataFrame({
            "date": future_dates.strftime("%Y-%m-%d"),
            "predicted_total": preds
        }).to_dict(orient="records")
    }


file_list = [
    "powerBank_weight.csv",
    "powerBank_price.csv",
    "powerBank_design.csv",
    "powerBank_compatibility.csv",
    "powerBank_chargingSpeed.csv"
]


for file in file_list:
    name = file.split("_")[-1].replace(".csv", "")
    print(f"Processing {name} ...")
    output_all[name] = lstm_forecast(file)

# 最後輸出 JSON 檔
with open("lstm_forecast_all.json", "w") as f:
    json.dump(output_all, f, ensure_ascii=False, indent=4)
