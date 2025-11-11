import os
import pandas as pd
#from crawlers.dcard_crawler_api import DcardCrawlerAPI
#from crawlers.ptt_crawler import PttCrawler
from analysis.analyzer import PowerbankAnalyzer
from crawlers.mobile_crawler import MobileCrawler

# === Step 0️⃣ 輸出資料夾 ===
output_dir = os.path.expanduser("~/AICompetition/crawlers_result")
os.makedirs(output_dir, exist_ok=True)
print(f"資料將輸出至：{output_dir}")

# === CSV 檢查函數 ===
def check_csv(csv_file):
    if not os.path.exists(csv_file):
        print(f"{csv_file} 不存在")
        return False
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            print(f"{csv_file} 內容為空")
            return False
        return True
    except pd.errors.EmptyDataError:
        print(f"{csv_file} 無法解析（空檔）")
        return False

# === Step 1️⃣ Dcard 爬蟲 ===
#dcard = DcardCrawlerAPI(keyword="行動電源", limit=20, delay=1)
#df_dcard = dcard.crawl()
#dcard_csv = os.path.join(output_dir, "data_dcard.csv")
#dcard.save_to_csv(dcard_csv)

# === Step 2️⃣ PTT Selenium 爬蟲 ===
#ptt = PttCrawler(board="MobileComm", keyword="行動電源", headless=True, delay=1)
#df_ptt = ptt.fetch_articles(pages=2)
#ptt_csv = os.path.join(output_dir, "data_ptt.csv")
#ptt.save_to_csv(ptt_csv)
#ptt.close()

# 使用 MobileCrawler

if __name__ == "__main__":
    crawler = MobileCrawler(
        keyword="行動電源",
        output_path=output_dir,
        headless=True
    )
    
    crawler.fetch_all_articles()  # 自動抓取所有頁數
    mobile_csv = os.path.join(output_dir, "data_mobile.csv")
    crawler.save_csv(mobile_csv)
    crawler.close()
"""
# === Step 3️⃣ 整合分析 ===
csv_files = []
#if check_csv(dcard_csv):
#    csv_files.append(dcard_csv)
#if check_csv(ptt_csv):
#    csv_files.append(ptt_csv)
if check_csv(mobile_csv):
    csv_files.append(mobile_csv)

if csv_files:
    analyzer = PowerbankAnalyzer(csv_files)
    analyzer.analyze_keywords()

    analyzed_csv = os.path.join(output_dir, "powerbank_analyzed.csv")
    chart_path = os.path.join(output_dir, "powerbank_summary.png")

    analyzer.save(analyzed_csv)
    analyzer.visualize(chart_path)
    print("\n專案完成！分析結果已存入：", output_dir)
else:
    print("沒有可分析的資料，流程結束")
"""