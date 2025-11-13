# crawlers/momo_crawler.py
import time
import urllib.parse
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd


class MomoCrawler:
    BASE_SEARCH_URL = "https://www.momoshop.com.tw/search/searchShop.jsp"

    def __init__(self, keyword: str, timeout: int = 20, headers: Optional[Dict[str, str]] = None):
        self.keyword = keyword
        self.timeout = timeout
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self._rows: List[Dict[str, Any]] = []

    def fetch_list_html(self, page: int = 1) -> str:
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

    def parse_list(self, html: str) -> List[Dict[str, Any]]:
        """
        解析 momo 搜尋結果頁；不同版位 class 可能會變動，必要時調整 selector。
        """
        soup = BeautifulSoup(html, "lxml")
        items: List[Dict[str, Any]] = []

        # 常見列表容器備援 selector（依實際抓到的 HTML 調整）
        # 你也可以先把 HTML 存下來開 DevTools 對 selector 做微調
        cards = (
            soup.select("ul#searchResults li")
            or soup.select("div.listArea li")
            or soup.select("li.goodsItem")
        )

        for li in cards:
            # 標題
            title_el = (
                li.select_one(".prdName")
                or li.select_one(".goodsTitle")
                or li.select_one("h3, h4, .name")
            )
            title = title_el.get_text(strip=True) if title_el else None

            # 連結
            a = li.select_one("a")
            href = a.get("href") if a else None
            if href and href.startswith("/"):
                href = urllib.parse.urljoin("https://www.momoshop.com.tw", href)

            # 價格（可能在不同容器）
            price_el = (
                li.select_one(".price")
                or li.select_one(".priceArea .money")
                or li.select_one(".priceArea")
            )
            price = price_el.get_text(strip=True) if price_el else None

            # 圖片
            img_el = li.select_one("img")
            img = None
            if img_el:
                img = img_el.get("data-original") or img_el.get("src")
                if img and img.startswith("//"):
                    img = "https:" + img

            # 產品代碼（若連結上帶參數）
            goods_no = None
            if href:
                q = urllib.parse.urlparse(href).query
                qs = urllib.parse.parse_qs(q)
                for key in ("i_code", "i_code2", "prodNo", "goods_code", "g_code"):
                    if key in qs and qs[key]:
                        goods_no = qs[key][0]
                        break

            if title and href:
                items.append(
                    {
                        "source": "momo",
                        "keyword": self.keyword,
                        "title": title,
                        "price": price,
                        "url": href,
                        "image": img,
                        "goods_no": goods_no,
                    }
                )

        return items

    def crawl(self, pages: int = 1, delay: float = 0.8) -> pd.DataFrame:
        self._rows.clear()
        for p in range(1, pages + 1):
            html = self.fetch_list_html(page=p)
            items = self.parse_list(html)
            self._rows.extend(items)
            if p < pages and delay > 0:
                time.sleep(delay)

        df = pd.DataFrame(self._rows)
        return df

    def save_csv(self, csv_path: str) -> None:
        if not self._rows:
            # 若尚未呼叫 crawl()，輸出空表頭 CSV
            pd.DataFrame(self._rows or [], columns=["source", "keyword", "title", "price", "url", "image", "goods_no"]) \
              .to_csv(csv_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(self._rows).to_csv(csv_path, index=False, encoding="utf-8-sig")