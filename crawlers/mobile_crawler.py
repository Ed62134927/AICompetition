import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

class MobileCrawler:
    def __init__(self, keyword, output_path, headless=True):
        self.keyword = keyword
        self.output_path = output_path
        self.articles = []
        self.base_url = "https://www.mobile01.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

    def fetch_comments(self, article_url, max_comments=20):
        comments = []
        try:
            response = requests.get(article_url, headers=self.headers)
            if response.status_code != 200:
                return comments

            soup = BeautifulSoup(response.text, "html.parser")
            # Mobile01 留言區塊
            comment_tags = soup.select("div.l-leaveMsg div.msgContent.c-summary__desc")

            for idx, tag in enumerate(comment_tags):
                if idx >= max_comments:
                    break
                text = tag.get_text(strip=True)
                if text:
                    comments.append(text)
        except Exception as e:
            print(f"抓取留言時發生錯誤：{e}")
        return comments


    def fetch_all_articles(self, max_pages=3):
        print(f"開始爬取 Mobile01，搜尋關鍵字：{self.keyword}")
        for page in range(1, max_pages + 1):
            url = f"https://www.mobile01.com/topiclist.php?f=738&p={page}"
            print(f"抓取第 {page} 頁：{url}")
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"無法存取 {url} (status {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            topics = soup.select("div.l-listTable__tr:not(.l-listTable__thead)")

            if not topics:
                print(f"第 {page} 頁沒有找到文章。")
                continue

            count = 0
            for topic in topics:
                title_tag = topic.select_one("a.c-link")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = self.base_url + "/" + title_tag["href"].lstrip("/")


                # ✅ 關鍵字過濾
                if self.keyword in title:
                    # 抓留言
                    comments = self.fetch_comments(link)
                    self.articles.append({
                        "title": title,
                        "url": link,
                        "comments": " || ".join(comments) if comments else ""
                    })
                    count += 1
            
            print(f"第 {page} 頁共找到 {count} 篇包含「{self.keyword}」的文章")
            time.sleep(1)

    def save_csv(self, filename):
        if not self.articles:
            print("沒有符合關鍵字的資料可儲存")
            return
        df = pd.DataFrame(self.articles)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"已儲存到：{filename}")

    def close(self):
        print("MobileCrawler 任務完成")
