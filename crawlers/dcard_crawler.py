import requests
import pandas as pd
import time

class DcardCrawler:
    def __init__(self, keyword="行動電源", limit=20, delay=1.0):
        """
        keyword : 搜尋關鍵字
        limit   : 抓取文章數量上限
        delay   : 每篇抓取內文延遲秒數
        """
        self.keyword = keyword
        self.limit = limit
        self.delay = delay
        self.results = []

        # 模擬瀏覽器 User-Agent
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
        }

    def fetch_posts(self):
        """透過搜尋 API 抓文章列表"""
        url = f"https://www.dcard.tw/service/api/v2/search/posts?query={self.keyword}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        posts = response.json()
        return posts[:self.limit]

    def fetch_post_content(self, post_id):
        """抓取單篇文章內文"""
        url = f"https://www.dcard.tw/service/api/v2/posts/{post_id}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        data = r.json()
        return data.get("content", "")

    def crawl(self):
        print(f"開始爬取 Dcard 搜尋關鍵字: {self.keyword}")
        posts = self.fetch_posts()
        for post in posts:
            content = self.fetch_post_content(post["id"])
            self.results.append({
                "title": post.get("title", ""),
                "link": f"https://www.dcard.tw/f/{post.get('forumAlias','')}/p/{post.get('id','')}",
                "createdAt": post.get("createdAt", ""),
                "content": content
            })
            time.sleep(self.delay)
        print(f"完成爬取 {len(self.results)} 篇文章")
        return pd.DataFrame(self.results)

    def save_to_csv(self, filename="dcard_api_data.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"已輸出 CSV: {filename}")
