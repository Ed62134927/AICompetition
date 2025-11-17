import pandas as pd
import numpy as np
import json
import glob
from autots import AutoTS

# =============== 設定參數 ===============
future_weeks = 8
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

    # 篩選給 AutoTS 的資料
    df_for_autots = df[["date", "total"]]

    # 建立 AutoTS 模型
    model = AutoTS(
        forecast_length=future_weeks,
        frequency='W',               # 週資料
        prediction_interval=0.95,
        ensemble='simple',
        model_list='default',        # 含深度模型（LSTM, NVAR等）
        transformer_list='fast',     
        max_generations=5,
        min_allowed_train_percent=0.3,
        verbose=0
    )

    try:
        model.fit(df_for_autots, date_col='date', value_col='total')
        prediction = model.predict()
        forecast_df = prediction.forecast.reset_index().rename(columns={
            'index': 'date',
            'total': 'predicted_total'
        })
        forecast_df['date'] = forecast_df['date'].dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Warning: model fitting failed for {csv_path}. Error: {e}")
        forecast_df = pd.DataFrame(columns=['date', 'predicted_total'])

    # 時間序列預測可能出現負值，此處採用出現負值皆取0
    forecast_df['predicted_total'] = forecast_df['predicted_total'].clip(lower=0)

    historical_df = df[["date", "total"]].tail(4).assign(
        date=df["date"].tail(4).dt.strftime("%Y-%m-%d")
    )

    return {
        "historical": historical_df.to_dict(orient="records"),
        "forecast": forecast_df.to_dict(orient="records")
    }

#files
file_list = [
    "week_powerBank_weight.csv",
    "week_powerBank_price.csv",
    "week_powerBank_design.csv",
    "week_powerBank_compatibility.csv",
    "week_powerBank_chargingSpeed.csv"
]

print("Starting AutoML forecasting...")

for file in file_list:
    name = file.split("_")[-1].replace(".csv", "")
    print(f"Processing {name} ...")
    output_all[name] = autots_forecast(file)

# 最後輸出 JSON 檔
with open("autots_forecast_weeks_UserRequirements.json", "w") as f:
    json.dump(output_all, f, ensure_ascii=False, indent=4)

print("AutoML forecasting finished. Output saved to autots_forecast_all_weeks.json")
