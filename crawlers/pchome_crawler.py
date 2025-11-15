import time
import requests
import pandas as pd
import urllib.parse
import html
from typing import List, Dict, Optional
from requests.exceptions import RequestException, Timeout, JSONDecodeError


class PChomeCrawler:
    """PChome 商品爬蟲 - 修正版"""
    
    BASE_SEARCH_API = "https://ecshweb.pchome.com.tw/search/v3.3/all/results"
    PRODUCT_URL_PREFIX = "https://24h.pchome.com.tw/prod/"
    IMG_URL_PREFIX = "https://ec1img.pchome.com.tw/"
    
    def __init__(self, keyword: str, timeout: int = 20, debug: bool = False):
        self.keyword = keyword
        self.timeout = timeout
        self.debug = debug
        self.session = requests.Session()
        
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-TW,zh;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://24h.pchome.com.tw/",
        })
        
        self.results: List[Dict[str, str]] = []
    
    def fetch_search_results(self, page: int = 1, min_price: str = "", max_price: str = "") -> List[Dict[str, str]]:
        """抓取搜尋結果
        
        Args:
            page: 頁碼
            min_price: 最低價格（選填）
            max_price: 最高價格（選填）
            
        Returns:
            商品列表
        """
        # 重要：使用 urllib.parse.quote 進行 URL 編碼
        encoded_keyword = urllib.parse.quote(self.keyword)
        
        params = {
            "q": encoded_keyword,  # 編碼後的關鍵字
            "page": str(page),
            "sort": "rnk/dc",
        }
        
        # 如果有價格範圍，加入參數
        if min_price or max_price:
            params["price"] = f"{min_price}-{max_price}"
        
        if self.debug:
            print(f"\n[DEBUG] 原始關鍵字: {self.keyword}")
            print(f"[DEBUG] 編碼後: {encoded_keyword}")
            print(f"[DEBUG] 請求 URL: {self.BASE_SEARCH_API}")
            print(f"[DEBUG] 參數: {params}")
        
        try:
            resp = self.session.get(
                self.BASE_SEARCH_API, 
                params=params, 
                timeout=self.timeout
            )
            
            if self.debug:
                print(f"[DEBUG] 狀態碼: {resp.status_code}")
                print(f"[DEBUG] 實際 URL: {resp.url}")
                print(f"[DEBUG] 回應大小: {len(resp.content)} bytes")
            
            resp.raise_for_status()
            
            # 設定編碼
            resp.encoding = 'UTF-8'
            
            try:
                data = resp.json()
                
                if self.debug:
                    print(f"[DEBUG] JSON Keys: {list(data.keys())}")
                    print(f"[DEBUG] totalRows: {data.get('totalRows', 0)}")
                
                # 檢查是否有商品
                prods = data.get("prods")
                
                if prods is None:
                    print(f"WARNING: 第 {page} 頁 API 回應中沒有 'prods' 欄位")
                    if self.debug:
                        print(f"[DEBUG] 完整回應: {data}")
                    return []
                
                if not prods:
                    total_rows = data.get('totalRows', 0)
                    print(f"INFO: 第 {page} 頁沒有商品 (totalRows: {total_rows})")
                    return []
                
                if self.debug:
                    print(f"[DEBUG] 商品數量: {len(prods)}")
                    if prods:
                        print(f"[DEBUG] 第一個商品: {prods[0]}")
                
                return prods
                
            except JSONDecodeError as e:
                print(f"ERROR: JSON 解析失敗: {e}")
                if self.debug:
                    print(f"[DEBUG] 回應內容: {resp.text[:500]}")
                return []
            
        except Timeout:
            print(f"ERROR: 第 {page} 頁請求超時")
            return []
        except RequestException as e:
            print(f"ERROR: 第 {page} 頁請求失敗: {e}")
            return []
    
    def _parse_product(self, prod: Dict) -> Optional[Dict[str, str]]:
        """解析單一商品資料"""
        try:
            # 使用 html.unescape 處理 HTML 實體
            title = html.unescape(prod.get("name", "")).strip()
            prod_id = prod.get("Id", "").strip()
            
            if not title or not prod_id:
                if self.debug:
                    print(f"[DEBUG] 跳過無效商品: title={title}, id={prod_id}")
                return None
            
            url = self.PRODUCT_URL_PREFIX + prod_id
            describe = html.unescape(prod.get("describe", "")).strip()
            price = prod.get("price", "")
            
            # 圖片 URL
            pic = prod.get("picB", "")
            img_url = self.IMG_URL_PREFIX + pic if pic else ""
            
            return {
                "title": title,
                "url": url,
                "describe": describe,
                "product_id": prod_id,
                "price": price,
                "img_url": img_url,
            }
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] 解析商品時發生錯誤: {e}")
            return None
    
    def crawl(self, pages: int = 1, delay: float = 1.0, 
              min_price: str = "", max_price: str = "") -> pd.DataFrame:
        """爬取多頁商品資料
        
        Args:
            pages: 要爬取的頁數
            delay: 每頁之間的延遲時間（秒）
            min_price: 最低價格（選填）
            max_price: 最高價格（選填）
            
        Returns:
            包含所有商品的 DataFrame
        """
        self.results.clear()
        
        print(f"\n{'='*60}")
        print(f"開始爬取 PChome")
        print(f"關鍵字: {self.keyword}")
        print(f"頁數: {pages}")
        if min_price or max_price:
            print(f"價格範圍: {min_price or '不限'} ~ {max_price or '不限'}")
        print(f"{'='*60}\n")
        
        for p in range(1, pages + 1):
            print(f"正在抓取第 {p}/{pages} 頁...")
            
            products = self.fetch_search_results(
                page=p, 
                min_price=min_price, 
                max_price=max_price
            )
            
            print(f"第 {p} 頁找到 {len(products)} 個商品")
            
            parsed_count = 0
            for prod in products:
                parsed = self._parse_product(prod)
                if parsed:
                    self.results.append(parsed)
                    parsed_count += 1
            
            print(f"成功解析 {parsed_count} 個商品")
            
            if p < pages:
                time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"爬取完成！共收集 {len(self.results)} 個商品")
        print(f"{'='*60}\n")
        
        return pd.DataFrame(self.results)
    
    def save_csv(self, path: str) -> None:
        """儲存結果為 CSV"""
        if not self.results:
            print("WARNING: 沒有資料可以儲存")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"已儲存 {len(df)} 筆資料至 {path}")
