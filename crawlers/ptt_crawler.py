# ptt_crawler.py
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@dataclass
class PttPost:
    作者: Optional[str]
    版面: Optional[str]
    標題: Optional[str]
    日期: Optional[str]
    內容: Optional[str]
    IP位置: Optional[str]
    連結: Optional[str]


class PttCrawler:
    def __init__(self, board: str = "MobileComm", keyword: str = "行動電源",
                 headless: bool = True, delay: float = 1.0, wait_sec: int = 10):
        self.board = board
        self.keyword = keyword
        self.delay = delay
        self.wait_sec = wait_sec
        self.results: List[Dict[str, Any]] = []

        # --- Chrome Options（較穩定的 headless 與旗標） ---
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1366,768")
        chrome_options.add_argument("--lang=zh-TW")

        # --- Driver with webdriver-manager ---
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        self.wait = WebDriverWait(self.driver, wait_sec)

        # --- PTT 18 禁 Cookie（避免每頁都要點） ---
        # 有些板會跳 18 禁，先設 cookie 再導向目標頁面
        self.driver.get("https://www.ptt.cc/ask/over18")
        self.driver.add_cookie({"name": "over18", "value": "1", "domain": "www.ptt.cc", "path": "/"})

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

    # 取得看板 index 頁 URL
    def _index_url(self, page_index: Optional[int] = None) -> str:
        # 預設 index.html；若需要由「上頁」一路往前就用預設；也可擴充 index page number 的模式
        return f"https://www.ptt.cc/bbs/{self.board}/index.html"

    def _click_over18_if_any(self):
        # 若頁面又跳 18 禁，點同意
        try:
            btn = self.wait.until(EC.presence_of_element_located((By.XPATH, "//button[text()='我同意']")))
            btn.click()
            time.sleep(self.delay)
        except Exception:
            # 沒跳就略過
            pass

    def _get_prev_page_url(self, page_html: str) -> Optional[str]:
        soup = BeautifulSoup(page_html, "html.parser")
        # 不能用 :contains，改用文字過濾
        for a in soup.select("a.btn.wide"):
            if a.get_text(strip=True) == "上頁":
                href = a.get("href")
                if href:
                    return "https://www.ptt.cc" + href
        return None

    def _extract_links_on_page(self, page_html: str) -> List[str]:
        soup = BeautifulSoup(page_html, "html.parser")
        return ["https://www.ptt.cc" + a["href"] for a in soup.select("div.title a") if a.has_attr("href")]

    def _parse_article(self, html: str) -> PttPost:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("div", id="main-content")
        # meta 值
        author = board = title = date = None
        try:
            metas = main.find_all("span", class_=re.compile(r"article-meta-value"))
            if len(metas) >= 4:
                author, board, title, date = [m.get_text(strip=True) for m in metas[:4]]
        except Exception:
            # 容錯
            pass

        # 內容（去掉 push 與簽名檔後的內容）
        content_text = None
        if main:
            # 移除 meta 與推文區（避免混雜）
            for tag in main.find_all(["div", "span"], class_=re.compile(r"(article-meta|push)")):
                tag.decompose()
            content_text = main.get_text("\n", strip=True)
            # 切掉 -- 後方簽名
            content_text = content_text.split("--")[0].strip()

        # IP 位置：優先找「※ 發信站」段落，若沒有再 regex fallback
        ip_location = None
        try:
            ip_line = soup.find(string=re.compile(r"※ 發信站"))
            if ip_line and "來自:" in ip_line:
                ip_location = ip_line.split("來自:")[-1].strip()
        except Exception:
            pass
        if not ip_location:
            m = re.search(r"來自:\s*([^\s]+)", soup.get_text(" ", strip=True))
            ip_location = m.group(1) if m else None

        return PttPost(
            作者=author, 版面=board, 標題=title, 日期=date,
            內容=content_text, IP位置=ip_location, 連結=None
        )

    def fetch_articles(self, pages: int = 2) -> pd.DataFrame:
        """
        從目前 index 開始，向「上頁」方向抓指定頁數；只保留標題包含 keyword 的文章。
        """
        print(f"開始爬取 PTT {self.board} 板，關鍵字：{self.keyword}，頁數：{pages}")
        try:
            current_url = self._index_url()
            for i in range(pages):
                self.driver.get(current_url)
                self._click_over18_if_any()
                # 等待列表載入（以 title 連結當條件）
                try:
                    self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.title a")))
                except Exception:
                    time.sleep(self.delay)

                page_html = self.driver.page_source
                links = self._extract_links_on_page(page_html)

                for link in links:
                    try:
                        self.driver.get(link)
                        self._click_over18_if_any()
                        # 等待主體載入
                        try:
                            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#main-content")))
                        except Exception:
                            time.sleep(self.delay)

                        article_html = self.driver.page_source
                        post = self._parse_article(article_html)
                        post.連結 = link

                        # 標題關鍵字過濾（None 保護）
                        title = post.標題 or ""
                        if self.keyword and (self.keyword not in title):
                            continue

                        self.results.append({
                            "作者": post.作者,
                            "版面": post.版面,
                            "標題": post.標題,
                            "日期": post.日期,
                            "內容": post.內容,
                            "IP 位置": post.IP位置,
                            "連結": post.連結,
                        })

                    except Exception as e:
                        print(f"無法讀取 {link}：{e}")
                        continue

                print(f"第 {i+1}/{pages} 頁完成，共累計 {len(self.results)} 篇")

                # 由當前頁的 HTML 取得「上頁」網址
                prev_url = self._get_prev_page_url(page_html)
                if not prev_url:
                    print("找不到上頁，提前結束")
                    break
                current_url = prev_url

                # 節流
                time.sleep(self.delay)

            print(f"完成爬取 {len(self.results)} 篇文章")
            return pd.DataFrame(self.results)
        except Exception as e:
            print("抓取過程發生例外：", e)
            return pd.DataFrame(self.results)
        finally:
            # 交給呼叫端決定是否 close()；如要自動關閉可啟用
            # self.close()
            pass

    def save_to_csv(self, filename: str):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"已輸出 CSV: {filename}")