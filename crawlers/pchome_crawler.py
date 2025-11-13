
import time
import requests
import pandas as pd
from typing import List, Dict


class PChomeCrawler:
    BASE_SEARCH_API = "https://ecshweb.pchome.com.tw/search/v3.3/all/results"

    def __init__(self, keyword: str, timeout: int = 20):
        self.keyword = keyword
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })
        self.results: List[Dict[str, str]] = []

    def fetch_search_results(self, page: int = 1) -> List[Dict[str, str]]:
        params = {
            "q": self.keyword,
            "page": str(page),
            "sort": "rnk/dc",
        }
        resp = self.session.get(self.BASE_SEARCH_API, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("prods", [])

    def crawl(self, pages: int = 1, delay: float = 1.0) -> pd.DataFrame:
        self.results.clear()
        for p in range(1, pages + 1):
            print(f"抓取第 {p} 頁...")
            products = self.fetch_search_results(page=p)
            print(f"商品數量：{len(products)}")
            for prod in products:
                title = prod.get("name")
                prod_id = prod.get("Id")
                url = f"https://24h.pchome.com.tw/prod/{prod_id}" if prod_id else ""
                comment = prod.get("describe", "") or prod.get("originName", "")
                if title and url:
                    self.results.append({
                        "title": title,
                        "url": url,
                        "comment": comment,
                    })
            if p < pages:
                time.sleep(delay)

        return pd.DataFrame(self.results)

    def save_csv(self, path: str) -> None:
        df = pd.DataFrame(self.results or [], columns=["title", "url", "comment"])
        df.to_csv(path, index=False, encoding="utf-8-sig")
