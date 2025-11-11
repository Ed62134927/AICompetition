import pandas as pd
import matplotlib.pyplot as plt

class PowerbankAnalyzer:
    def __init__(self, csv_files):
        # 支援合併多平台資料
        dfs = [pd.read_csv(f) for f in csv_files]
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"已讀取 {len(self.df)} 筆資料")

        self.keywords = {
            "重量體積": ["重量", "體積", "厚度", "攜帶", "輕", "重"],
            "充電速度": ["充電", "快充", "PD", "速度", "效率"],
            "接口兼容性": ["Type-C", "USB", "接口", "相容", "輸出", "輸入"],
            "外觀方便性": ["外觀", "設計", "顏值", "質感", "手感","方便"],
            "價格": ["價格", "便宜", "貴", "CP值", "划算", "優惠"]
        }

    def analyze_keywords(self):
        for category, words in self.keywords.items():
            pattern = "|".join(words)
            self.df[category] = self.df["content"].astype(str).str.count(pattern)
        print("已完成關鍵字分析")
        return self.df

    def visualize(self, output="keyword_summary.png"):
        summary = self.df[list(self.keywords.keys())].sum().sort_values(ascending=False)
        summary.plot(kind="bar", figsize=(8, 5))
        plt.title("行動電源五大面向關注度")
        plt.ylabel("出現次數")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output)
        print(f"圖表已輸出：{output}")

    def save(self, filename="analyzed_data.csv"):
        self.df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"分析後資料已輸出至 {filename}")
