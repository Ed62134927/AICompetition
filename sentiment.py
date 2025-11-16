# PowerPulse 改進版情感分析系統
# 解決負面偏見問題,提升準確性

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import umap
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import jieba
import re

class ImprovedABSA:
    """
    改進版方面級情感分析
    修正負面偏見,提升準確性
    """
    
    def __init__(self, model_name='ckiplab/bert-base-chinese'):
        print("📥 載入改進版情感分析模型...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 定義產品方面與更精確的關鍵詞權重
        self.aspects = {
            '重量體積': {
                'keywords': ['輕', '重', '薄', '厚', '小', '大', '體積', '重量', '尺寸', '便攜', '攜帶', '輕薄', '輕巧'],
                'positive_terms': {
                    '輕': 1.0, '薄': 1.0, '小': 0.8, '便攜': 1.0, '輕巧': 1.0, '袖珍': 0.9,
                    '迷你': 0.8, '超薄': 1.2, '超輕': 1.2, '不佔空間': 1.0, '體積小': 0.9,
                    '重量輕': 1.0, '手感輕': 0.8, '易攜': 0.9, '方便攜帶': 1.0, '隨身': 0.8,
                    '收納方便': 0.9, '體積適中': 0.6, '輕薄': 1.0
                },
                'negative_terms': {
                    '重': 0.8, '厚': 0.8, '大': 0.6, '笨重': 1.2, '佔空間': 1.0, '超大': 1.0,
                    '超重': 1.2, '超厚': 1.0, '攜帶不便': 1.0, '難攜': 0.9, '體積大': 0.8,
                    '重量重': 0.9, '手感重': 0.7, '收納困難': 0.9, '不方便攜帶': 1.0
                }
            },
            '充電速度': {
                'keywords': ['快充', '慢', '快', '速度', '充電', 'PD', 'QC', '瓦數', 'W', '快速', '閃充'],
                'positive_terms': {
                    '快': 0.9, '快速': 1.0, '急速': 1.1, '秒充': 1.2, '閃充': 1.1, '充電快': 1.0,
                    '充電很快': 1.1, '充電超快': 1.2, '充電迅速': 1.0, '充電不用等': 1.0,
                    '快充': 1.0, '快充支援': 0.9, '極速充電': 1.2, '超快充': 1.2, '給力': 0.8
                },
                'negative_terms': {
                    '慢': 0.8, '龜速': 1.2, '久': 0.6, '等很久': 1.0, '充電慢': 0.9,
                    '充電很慢': 1.0, '充電超慢': 1.1, '充電緩慢': 0.9, '充電等待': 0.7,
                    '慢充': 0.8, '充電拖延': 1.0, '充電不穩': 1.0, '充電卡頓': 1.1
                }
            },
            '接口相容性': {
                'keywords': ['Type-C', 'Lightning', 'USB', '接口', '孔', '線', '相容', '通用', '萬用', '多口'],
                'positive_terms': {
                    '通用': 1.0, '相容': 1.0, '萬用': 1.0, '多口': 0.9, '齊全': 1.0, '支援': 0.8,
                    '支援多種': 1.0, '跨平台': 0.9, '適用': 0.8, '兼容': 0.9, '支援Type-C': 0.8,
                    '支援PD': 0.8, '支援QC': 0.8, '多功能': 0.9, '全面': 0.9
                },
                'negative_terms': {
                    '不相容': 1.2, '沒有': 0.9, '缺少': 1.0, '只有': 0.6, '不支援': 1.1,
                    '不兼容': 1.2, '不適用': 1.0, '插頭缺少': 1.0, '不支援Type-C': 1.0,
                    '不支援PD': 0.9, '單一': 0.7
                }
            },
            '外觀材質': {
                'keywords': ['外觀', '質感', '材質', '設計', '顏色', '美', '醜', '好看', '塑膠', '金屬'],
                'positive_terms': {
                    '質感': 1.0, '高級': 1.1, '精緻': 1.0, '好看': 0.9, '美': 1.0, '時尚': 0.9,
                    '漂亮': 0.9, '大方': 0.8, '簡約': 0.8, '現代': 0.7, '有質感': 1.0,
                    '金屬感': 0.9, '磨砂': 0.7, '手感好': 0.9, '手感舒適': 0.9, '細膩': 0.8
                },
                'negative_terms': {
                    '廉價': 1.2, '醜': 1.1, '塑膠感': 1.0, '粗糙': 1.0, '手感差': 1.0,
                    '難看': 1.0, '老氣': 0.9, '普通': 0.5, '單調': 0.7, '不美觀': 0.9
                }
            },
            '價格': {
                'keywords': ['價格', '價錢', '價值', '貴', '便宜', '划算', '超值', 'CP', '性價比'],
                'positive_terms': {
                    '便宜': 0.9, '划算': 1.0, '超值': 1.1, '值得': 1.0, 'CP值高': 1.1,
                    '優惠': 0.8, '特價': 0.7, '性價比高': 1.1, '價格合理': 0.9, '價格親民': 0.9,
                    '物超所值': 1.1, '物有所值': 0.9, '平價': 0.8, '實惠': 0.9
                },
                'negative_terms': {
                    '貴': 0.8, '昂貴': 1.0, '不值': 1.1, 'CP值低': 1.1, '坑錢': 1.3,
                    '價格偏高': 0.9, '價格過高': 1.0, '太貴': 1.0, '不划算': 1.0, '太高': 0.8
                }
            }
        }
        
        # 使用多個情感分類器進行集成
        self.sentiment_classifiers = []
        try:
            # 中文情感分類器1
            self.sentiment_classifiers.append(
                pipeline('sentiment-analysis', 
                        model='uer/roberta-base-finetuned-jd-binary-chinese',
                        device=0 if torch.cuda.is_available() else -1)
            )
        except:
            print("警告: 主要情感分類器載入失敗")
        
        # 情感詞典
        self.load_sentiment_lexicon()
    
    def load_sentiment_lexicon(self):
        """載入中文情感詞典"""
        self.positive_words = set([
            '好', '棒', '讚', '優', '佳', '妙', '棒', '贊', '很棒', '非常好', '超棒',
            '不錯', '滿意', '喜歡', '推薦', '值得', '完美', '優秀', '出色', '卓越',
            '給力', '實用', '方便', '舒服', '舒適', '順暢', '流暢', '穩定'
        ])
        
        self.negative_words = set([
            '差', '爛', '糟', '壞', '劣', '垃圾', '失望', '後悔', '不好', '很差',
            '不滿', '討厭', '難用', '不推薦', '不值', '缺點', '問題', '故障',
            '損壞', '破', '斷', '壞掉', '不穩', '卡頓', '延遲', '漏', '漏電'
        ])
        
        self.negation_words = set(['不', '沒', '無', '非', '未', '別', '莫', '勿'])
    
    def _ensure_text(self, text) -> str:
        """確保輸入是字串"""
        if pd.isna(text):
            return ''
        return str(text) if not isinstance(text, str) else text
    
    def analyze_with_lexicon(self, text: str) -> float:
        """基於詞典的情感分析(輔助方法)"""
        words = list(jieba.cut(text))
        
        pos_count = 0
        neg_count = 0
        
        for i, word in enumerate(words):
            # 檢查否定詞
            is_negated = i > 0 and words[i-1] in self.negation_words
            
            if word in self.positive_words:
                if is_negated:
                    neg_count += 1
                else:
                    pos_count += 1
            elif word in self.negative_words:
                if is_negated:
                    pos_count += 1
                else:
                    neg_count += 1
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def calculate_aspect_score(self, text: str, aspect_info: Dict) -> float:
        """
        計算方面情感分數(改進版)
        使用多種方法的加權平均
        """
        scores = []
        weights = []
        
        # 方法1: 基於詞典的關鍵詞匹配(權重較高)
        pos_score = 0
        neg_score = 0
        
        for term, weight in aspect_info['positive_terms'].items():
            count = text.count(term)
            pos_score += count * weight
        
        for term, weight in aspect_info['negative_terms'].items():
            count = text.count(term)
            neg_score += count * weight
        
        # 檢查否定詞
        negation_pattern = r'(不|沒|無)' + r'(' + '|'.join(aspect_info['positive_terms'].keys()) + r')'
        negation_matches = len(re.findall(negation_pattern, text))
        
        # 調整分數
        pos_score -= negation_matches * 0.8
        neg_score += negation_matches * 0.8
        
        total = pos_score + neg_score
        if total > 0:
            keyword_score = (pos_score - neg_score) / total
            scores.append(keyword_score)
            weights.append(0.6)  # 關鍵詞匹配權重60%
        
        # 方法2: 通用詞典情感分析
        lexicon_score = self.analyze_with_lexicon(text)
        if abs(lexicon_score) > 0.1:
            scores.append(lexicon_score)
            weights.append(0.2)  # 通用詞典權重20%
        
        # 方法3: 預訓練模型(如果可用)
        if self.sentiment_classifiers:
            try:
                result = self.sentiment_classifiers[0](text[:512])[0]
                model_score = result['score'] if result['label'] == 'positive' else -result['score']
                scores.append(model_score)
                weights.append(0.2)  # 模型權重20%
            except:
                pass
        
        # 加權平均
        if not scores:
            return 0.0
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # 歸一化
        
        final_score = np.average(scores, weights=weights)
        
        # 平滑處理,避免極端值
        final_score = np.tanh(final_score * 1.5)  # 使用tanh壓縮到[-1,1]
        
        return float(final_score)
    
    def extract_aspect_mentions(self, text: str) -> Dict[str, List[str]]:
        """提取文本中提及的產品方面"""
        mentions = {}
        text = self._ensure_text(text)
        
        for aspect, info in self.aspects.items():
            sentences = text.split('。')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in info['keywords']):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                mentions[aspect] = relevant_sentences
        
        return mentions
    
    def analyze_aspect_sentiment(self, text: str, aspect: str) -> Dict:
        """
        分析特定方面的情感(改進版)
        """
        aspect_info = self.aspects.get(aspect)
        if not aspect_info:
            return None
        
        text = self._ensure_text(text)
        
        # 提取相關句子
        sentences = text.split('。')
        relevant_sentences = [
            s for s in sentences 
            if any(keyword in s for keyword in aspect_info['keywords'])
        ]
        
        if not relevant_sentences:
            return {
                'mentioned': False,
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'evidence': []
            }
        
        combined_text = '。'.join(relevant_sentences)
        
        # 使用改進的評分方法
        score = self.calculate_aspect_score(combined_text, aspect_info)
        
        # 調整後的閾值(更對稱)
        POSITIVE_THRESHOLD = 0.25
        NEGATIVE_THRESHOLD = -0.25
        
        # 計算置信度
        confidence = min(abs(score) * 1.5, 1.0)
        
        # 判定情感類別
        if score > POSITIVE_THRESHOLD:
            sentiment = 'positive'
        elif score < NEGATIVE_THRESHOLD:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'mentioned': True,
            'sentiment': sentiment,
            'score': float(score),
            'confidence': float(confidence),
            'evidence': relevant_sentences[:3],
            'pos_mentions': sum(combined_text.count(t) for t in aspect_info['positive_terms'].keys()),
            'neg_mentions': sum(combined_text.count(t) for t in aspect_info['negative_terms'].keys())
        }
    
    def analyze_full_review(self, text: str) -> Dict:
        """分析完整評論的所有方面"""
        results = {}
        
        for aspect in self.aspects.keys():
            results[aspect] = self.analyze_aspect_sentiment(text, aspect)
        
        return results
    
    def batch_analyze(self, texts: List[str], show_progress=True) -> pd.DataFrame:
        """批次分析多篇評論"""
        from tqdm import tqdm
        
        all_results = []
        
        iterator = tqdm(texts, desc="分析中") if show_progress else texts
        
        for text in iterator:
            safe_text = self._ensure_text(text)
            analysis = self.analyze_full_review(safe_text)
            
            row = {'text': safe_text[:200]}
            
            for aspect, result in analysis.items():
                row[f'{aspect}_mentioned'] = result['mentioned']
                row[f'{aspect}_score'] = result['score'] if result['mentioned'] else None
                row[f'{aspect}_sentiment'] = result['sentiment'] if result['mentioned'] else None
                row[f'{aspect}_confidence'] = result['confidence'] if result['mentioned'] else None
            
            all_results.append(row)
        
        return pd.DataFrame(all_results)
    
    def get_sentiment_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """獲取情感分佈統計"""
        aspects = ['重量體積', '充電速度', '接口相容性', '外觀材質', '價格']
        
        stats = []
        for aspect in aspects:
            sentiment_col = f'{aspect}_sentiment'
            score_col = f'{aspect}_score'
            
            if sentiment_col in df.columns:
                mentioned = df[f'{aspect}_mentioned'].sum()
                
                if mentioned > 0:
                    sentiment_counts = df[df[f'{aspect}_mentioned']][sentiment_col].value_counts()
                    avg_score = df[df[f'{aspect}_mentioned']][score_col].mean()
                    
                    stats.append({
                        '方面': aspect,
                        '提及次數': mentioned,
                        '正面': sentiment_counts.get('positive', 0),
                        '中性': sentiment_counts.get('neutral', 0),
                        '負面': sentiment_counts.get('negative', 0),
                        '平均分數': round(avg_score, 3),
                        '正面比例': f"{sentiment_counts.get('positive', 0) / mentioned * 100:.1f}%",
                        '負面比例': f"{sentiment_counts.get('negative', 0) / mentioned * 100:.1f}%"
                    })
        
        return pd.DataFrame(stats)


# 使用範例
if __name__ == "__main__":
    # 測試數據
    test_texts = [
        "這個行動電源超輕薄,充電速度很快,支援PD快充,質感也不錯,價格合理",
        "充電寶太重了,而且充電很慢,不支援Type-C很不方便",
        "GaN技術真的不錯,充電超快,就是價格有點貴",
        "這款尿袋很輕便,但容量太小了,外觀設計很美",
        "重量還可以,充電速度正常,接口齊全很方便,外觀一般般,價格偏高",
        "產品很好用,質感不錯,充電也挺快的,輕巧方便攜帶",
        "還行吧,沒什麼特別的,充電速度普通,外觀也普通",
        "充電不快但也不算慢,重量適中,質感還可以"
    ]
    data = pd.read_csv('./AICompetition/crawlers_result/data_mobile.csv')
    data1=list(data['title'])
    data2=list(data['comments'])
    data3=data1+data2
    data3=[x for x in data3 if str(x).lower() not in ('nan', 'none')]
    
    print("🚀 初始化改進版情感分析系統...")
    analyzer = ImprovedABSA()
    
    print("\n📊 執行批次分析...")
    results_df = analyzer.batch_analyze(data3)
    
    print("\n📈 情感分佈統計:")
    stats_df = analyzer.get_sentiment_distribution(results_df)
    print(stats_df.to_string(index=False))
    
    print("\n💾 保存結果...")
    results_df.to_csv('improved_absa_results.csv', index=False, encoding='utf-8-sig')
    stats_df.to_csv('sentiment_distribution.csv', index=False, encoding='utf-8-sig')
    
    print("\n✅ 分析完成!")
    
    # 顯示部分詳細結果
    print("\n📋 部分詳細結果:")
    for idx, row in results_df.head(3).iterrows():
        print(f"\n文本 {idx+1}: {row['text']}")
        print("各方面情感:")
        for aspect in ['重量體積', '充電速度', '價格']:
            if row[f'{aspect}_mentioned']:
                print(f"  - {aspect}: {row[f'{aspect}_sentiment']} ({row[f'{aspect}_score']:.2f})")
