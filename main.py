
import os
import pandas as pd
#from crawlers.dcard_crawler_api import DcardCrawlerAPI
#from crawlers.ptt_crawler import PttCrawler
from crawlers.mobile_crawler import MobileCrawler
#from crawlers.pchome_crawler import PChomeCrawler
#from crawlers.momo_crawler import MomoCrawler

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



if __name__ == "__main__":
    
    # 使用 MobileCrawler
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
    # ---- momo ----
    momo = MomoCrawler(keyword="行動電源", timeout=20)
    print("開始爬取 momo...")
    df_momo = momo.crawl(pages=1, delay=0.8)  # 需要多頁可自行調整 pages
    momo_csv = os.path.join(output_dir, "data_momo.csv")
    # 兩種擇一：直接用內建 save_csv 或自行輸出 pandas
    momo.save_csv(momo_csv)
    # df_momo.to_csv(momo_csv, index=False, encoding="utf-8-sig")
    print(f"momo 已輸出 CSV：{momo_csv}")
    """
    """
    # 使用 PChomeCrawler
    
    pchome = PChomeCrawler(keyword="行動電源", timeout=20, debug=True)
    print("開始爬取 PChome...")
    df_pchome = pchome.crawl(pages=2, delay=1.0)
    pchome_csv = os.path.join(output_dir, "data_pchome.csv")
    pchome.save_csv(pchome_csv)
    print(f"PChome 已輸出 CSV：{pchome_csv}")
    """
