
import time
import urllib.parse
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd


class MomoCrawler:
    BASE_SEARCH_URL = "https://www.momoshop.com.tw/search/searchShop.jsp"
    BASE_DETAIL_URL = "https://www.momoshop.com.tw/goods/GoodsDetail.jsp"

    def __init__(self, keyword: str, timeout: int = 20):
        self.keyword = keyword
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.results: List[Dict[str, str]] = []

    def fetch_search_page(self, page: int = 1) -> str:
        params = {
            "keyword": self.keyword,
            "curPage": str(page),
            "_isFuzzy": "0",
            "searchType": "1",
        }
        url = f"{self.BASE_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def parse_search_results(self, html: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, "lxml")
        cards = soup.select("ul#searchResults li") or soup.select("li.goodsItem")
        items = []

        for li in cards:
            title_el = li.select_one(".prdName") or li.select_one(".goodsTitle")
            title = title_el.get_text(strip=True) if title_el else None

            a_tag = li.select_one("a")
            href = a_tag.get("href") if a_tag else None
            if href and href.startswith("/"):
                href = urllib.parse.urljoin("https://www.momoshop.com.tw", href)

            if title and href:
                items.append({"title": title, "url": href})

        return items

    def fetch_comments(self, url: str) -> List[str]:
        # 嘗試從商品頁面擷取評論（若 API 可用）
        i_code = self.extract_icode(url)
        if not i_code:
            return []

        # 模擬 API 呼叫（實際 API 結構可能需調整）
        api_url = f"https://www.momoshop.com.tw/ajax/goods/GetReviewList.jsp?i_code={i_code}&page=1"
        try:
            resp = self.session.get(api_url, timeout=self.timeout)
            data = resp.json()
            comments = [r.get("content", "").strip() for r in data.get("rtnReviewList", [])]
            return comments
        except Exception:
            return []

    def extract_icode(self, url: str) -> Optional[str]:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        return qs.get("i_code", [None])[0]

    def crawl(self, pages: int = 1, delay: float = 1.0) -> pd.DataFrame:
        self.results.clear()
        for p in range(1, pages + 1):
            html = self.fetch_search_page(page=p)
            items = self.parse_search_results(html)
            for item in items:
                comments = self.fetch_comments(item["url"])
                self.results.append({
                    "title": item["title"],
                    "url": item["url"],
                    "comments": " | ".join(comments) if comments else "",
                })
            if p < pages:
                time.sleep(delay)

        return pd.DataFrame(self.results)

    def save_csv(self, path: str) -> None:
        df = pd.DataFrame(self.results or [], columns=["title", "url", "comments"])
        df.to_csv(path, index=False, encoding="utf-8-sig")
