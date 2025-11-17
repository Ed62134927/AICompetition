import pandas as pd
import numpy as np
import json
import glob
from autots import AutoTS

# =============== 設定參數 ===============
future_weeks = 24
output_all = {}

# =============== 定義函式：AutoTS 預測 ===============
def autots_forecast(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')

    print(f"--- Processing file: {csv_path} ---")
    print("top 5 data (df.head()):")
    print(df.head())
    print("\nAll columns read(df.columns):")
    print(df.columns)
    print("--------------------------------------\n")

    df["date"] = pd.to_datetime(df["date"])

    # --- 平滑處理：避免太多 0 或噪音 ---
    df["smooth"] = df["score"].rolling(window=3, min_periods=1).mean()

    # AutoTS 用平滑後的數據
    df_for_autots = df[["date", "smooth"]].rename(columns={
        "smooth": "score"
    })

    # 建立 AutoTS 模型
    model = AutoTS(
        forecast_length=future_weeks,
        frequency='W',
        prediction_interval=0.95,
        ensemble='simple',
        model_list='default',
        transformer_list='fast',
        max_generations=5,
        min_allowed_train_percent=0.3,
        verbose=0
    )

    try:
        model.fit(df_for_autots, date_col='date', value_col='score')
        prediction = model.predict()
        forecast_df = prediction.forecast.reset_index().rename(columns={
            'index': 'date',
            'score': 'predicted_score'
        })
        forecast_df['date'] = forecast_df['date'].dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Warning: model fitting failed for {csv_path}. Error: {e}")
        forecast_df = pd.DataFrame(columns=['date', 'predicted_score'])

    # --- 負值截斷 ---
    forecast_df['predicted_score'] = forecast_df['predicted_score'].clip(lower=0)

    # --- 歷史資料回傳原始資料最後 4 筆 ---
    historical_df = df[["date", "score"]].tail(4).assign(
        date=df["date"].tail(4).dt.strftime("%Y-%m-%d")
    )

    return {
        "historical": historical_df.to_dict(orient="records"),
        "forecast": forecast_df.to_dict(orient="records")
    }

#files
file_list = [
    "GaN.csv",
    "magsafe.csv",
    "pd.csv",
    "typeC.csv"
]

print("Starting AutoML forecasting...")

for file in file_list:
    name = file.split("_")[-1].replace(".csv", "")
    print(f"Processing {name} ...")
    output_all[name] = autots_forecast(file)

# 最後輸出 JSON 檔
with open("autots_forecast_TechTrends.json", "w") as f:
    json.dump(output_all, f, ensure_ascii=False, indent=4)

print("AutoML forecasting finished. Output saved to autots_forecast_TechTrends.json")
