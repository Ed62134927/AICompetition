import requests
from bs4 import BeautifulSoup
import json
import re
import time
import pandas as pd
product_ids = []
product_names = []
product_prices = []
allresult=[]
for i in range(1,72):
   url = f"https://24h.pchome.com.tw/search/?q=行動電源&p={i}"
   resp = requests.get(url)
   soup = BeautifulSoup(resp.text, "html.parser")
   for script in soup.find_all("script", type="application/ld+json"):
     try:
        data = json.loads(script.string)
        # 有些 script 可能是 list
        if isinstance(data, list):
            for item in data:
                if item.get("@type") == "Product" and "url" in item:
                    name = item.get("name")
                    price = item.get("offers", {}).get("price")
                    product_names.append(name)
                    product_prices.append(price)
                    match = re.search(r'/prod/([A-Z0-9\\-]+)', item["url"])
                    if match:
                        product_ids.append(match.group(1))
        elif isinstance(data, dict):
            if data.get("@type") == "Product" and "url" in data:
                name = data.get("name")
                price = data.get("offers", {}).get("price")
                product_names.append(name)
                product_prices.append(price)
                match = re.search(r'/prod/([A-Z0-9\\-]+)', data["url"])
                if match:
                    product_ids.append(match.group(1))
     except Exception:
        continue

# --- 配置參數 ---
# 將 limit 設為最大值 50，以減少請求次數，加快速度
PAGE_LIMIT = 50
count=0
for i in product_ids:
    
    PRODUCT_ID = i
    BASE_URL = f"https://ecapi-cdn.pchome.com.tw/fsapi/reviews/{PRODUCT_ID}/comments"

    # 模擬瀏覽器行為的 Headers（與您上一個成功的版本一致）
    HEADERS = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      'Referer': f'https://24h.pchome.com.tw/prod/{PRODUCT_ID}' 
    }


    # --- 爬蟲主邏輯 ---
    def fetch_all_reviews():
       """迭代所有頁碼，從 PChome 評論 API 獲取所有評論。"""
    
       all_reviews_data = []
       total_pages = 1 # 初始假設至少有 1 頁
       page_num = 1
    
       print(f"--- 開始爬取產品 ID: {PRODUCT_ID} 的評論 ---")

       while page_num <= total_pages:
          # 構造帶有 page 和 limit 參數的完整 URL
          params = {
            "type": "all",
            "category": "best",
            "attachment": "",
            "page": page_num,
            "limit": PAGE_LIMIT
          }
        
          try:
            response = requests.get(BASE_URL, params=params, headers=HEADERS)
            response.raise_for_status() # 檢查 HTTP 錯誤
            data = response.json()
            
            # *** 【修正點 1】在第一次請求時，獲取總頁數 ***
            if page_num == 1:
                total_pages = data.get('TotalPages', 1)
                print(f"產品總評論數: {data.get('Total', '未知')}; 總頁數: {total_pages}")
                
            # *** 【修正點 2】數據提取路徑是 'Rows' ***
            comments_rows = data.get('Rows', [])

            if not comments_rows:
                # 如果 TotalPages 錯誤或某頁為空，則停止
                break
            
            # 提取用戶評論文本和評分
            for row in comments_rows:
                # 提取用戶評論文本 (路徑: Comments -> User)
                user_comment = row.get('Comments', {}).get('User', '').strip()
                # 提取評分 (路徑: QualityLikes)
                quality_likes = row.get('QualityLikes')
                
                # 僅保留有評論內容的數據
                if user_comment:
                    all_reviews_data.append({
                        'text': user_comment,
                        'rating': quality_likes,
                        'date': row.get('ReviewsDate')
                    })
            
            print(f"已成功爬取第 {page_num}/{total_pages} 頁，累積評論數：{len(all_reviews_data)} 條。")
            
            page_num += 1
            time.sleep(1) # 設置延遲

          except requests.exceptions.RequestException as e:
            print(f"請求錯誤，終止爬取：{e}")
            break
          except json.JSONDecodeError:
            print("無法解析 JSON 響應。")
            break

       return all_reviews_data

    # 執行函數
    collected_data = fetch_all_reviews()

    print("\n--- 爬取結果 ---")
    print(f"總共收集到 {len(collected_data)} 條有效評論和評分。")
    # 打印前 2 條評論作為範例
    print("前 2 條評論數據範例:")
    for i, item in enumerate(collected_data[:2]):
      print(f"[{i+1}] 評分: {item['rating']}, 評論: {item['text'][:50]}...")
    row={}
    row["product_name"]=product_names[count]
    row["product_price"]=product_prices[count]
    row['reviews']=collected_data
    allresult.append(row)
    count+=1
result_df=pd.DataFrame(allresult)
result_df.to_csv("pchome_reviews.csv",index=False,encoding="utf-8-sig")
print("\n✅ 爬蟲完成!")

