import pandas as pd
import re, time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

class PttCrawler:
    def __init__(self, board="MobileComm", keyword="行動電源", headless=True, delay=1):
        self.board = board
        self.keyword = keyword
        self.delay = delay
        self.results = []

        # 設定 headless 瀏覽器
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(options=chrome_options)

    def close(self):
        self.driver.quit()

    def fetch_articles(self, pages=2):
        """爬取指定看板關鍵字文章"""
        base_url = f"https://www.ptt.cc/bbs/{self.board}/index.html"
        print(f"開始爬取 PTT {self.board} 板，關鍵字：{self.keyword}")

        for _ in range(pages):
            self.driver.get(base_url)
            time.sleep(self.delay)

            # 如果有滿18歲確認按鈕
            try:
                enter_button = self.driver.find_element(By.XPATH, "//button[text()='我同意']")
                enter_button.click()
                time.sleep(self.delay)
            except:
                pass

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            links = [a['href'] for a in soup.select("div.title a")]  # 先抓所有文章

            # 拜訪每篇文章
            for link in links:
                try:
                    full_url = "https://www.ptt.cc" + link
                    self.driver.get(full_url)
                    time.sleep(self.delay)
                    article_soup = BeautifulSoup(self.driver.page_source, "html.parser")
                    main_tag = article_soup.find('div', id='main-content')

                    meta_tags = main_tag.find_all('span', re.compile('article-meta-value'))
                    author = meta_tags[0].text
                    board = meta_tags[1].text
                    title = meta_tags[2].text
                    date = meta_tags[3].text

                    content_text = main_tag.get_text(separator='\n').split('--')[0].strip()
                    ip_match = re.search(r'來自: (.+)', str(article_soup))
                    ip_location = ip_match.group(1) if ip_match else ''

                    self.results.append({
                        '作者': author,
                        '版面': board,
                        '標題': title,
                        '日期': date,
                        '內容': content_text,
                        'IP 位置': ip_location
                    })
                except Exception as e:
                    print(f"無法讀取 {link}：{e}")
                    continue

            # 找上一頁
            try:
                prev_link = soup.select_one("a.btn.wide:contains('上頁')")['href']
                base_url = "https://www.ptt.cc" + prev_link
            except:
                break

        print(f"完成爬取 {len(self.results)} 篇文章")
        return pd.DataFrame(self.results)

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"已輸出 CSV: {filename}")
