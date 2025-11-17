# PowerPulse 進階 NLP 系統
# 包含：ABSA、詞嵌入、語義聚類、視覺化、零樣本學習

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

# =======================================
# 1. 方面級情感分析 (ABSA)
# =======================================

class AspectBasedSentimentAnalyzer:
    """
    進階方面級情感分析系統
    使用中文 RoBERTa 進行細粒度的產品特徵情感分析
    """
    
    def __init__(self, model_name='ckiplab/bert-base-chinese'):
        print("📥 載入 RoBERTa 模型...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 定義產品方面與相關關鍵詞
        self.aspects = {
            '重量體積': {
                'keywords': [
                    '輕', '重', '薄', '厚', '小', '大', '體積', '重量', '尺寸', '便攜', '攜帶', '輕薄', '輕巧', '袖珍',
                    '迷你', '微型', '超薄', '超輕', '超大', '超重', '超小', '超厚', '攜帶方便', '攜帶不便', '不佔空間', '佔空間',
                    '體積小', '體積大', '重量輕', '重量重', '手感重', '手感輕', '易攜', '難攜', '方便攜帶', '不方便攜帶',
                    '口袋', '包包', '隨身', '隨身攜帶', '收納', '收納方便', '收納困難', '體積適中', '體積適合', '體積剛好'
                ],
                'positive_terms': [
                    '輕', '薄', '小', '便攜', '輕巧', '袖珍', '迷你', '超薄', '超輕', '不佔空間', '體積小', '重量輕',
                    '易攜', '方便攜帶', '隨身', '收納方便', '體積適中', '體積適合', '體積剛好'
                ],
                'negative_terms': [
                    '重', '厚', '大', '笨重', '佔空間', '超大', '超重', '超厚', '攜帶不便', '難攜', '不方便攜帶',
                    '體積大', '重量重', '手感重', '收納困難'
                ]
            },
            '充電速度': {
                'keywords': [
                    '快充', '慢', '快', '速度', '充電', 'PD', 'QC', '瓦數', 'W', '快速', '閃充', '急速', '充滿',
                    '充電快', '充電慢', '充電速度', '充電效率', '充電時間', '充電很快', '充電很慢', '充電超快', '充電超慢',
                    '充電迅速', '充電緩慢', '充電等待', '充電等很久', '充電不用等', '充電馬上', '充電即時', '充電即刻',
                    '充電體驗', '充電表現', '充電過程', '充電時長', '充電時效', '充電功率', '充電支援', '快充支援',
                    '快充功能', '快充效果', '快充表現', '快充體驗', '快充協議', '快充技術', '快充標準', '快充速度',
                    '閃充技術', '閃充速度', '閃充效果', '閃充體驗', '閃充表現', '急速充電', '極速充電', '超快充',
                    '超慢充', '慢充', '慢速充電', '慢速', '充電緩慢', '充電拖延', '充電拖很久', '充電不穩', '充電不順',
                    '充電卡頓', '充電卡住', '充電不良', '充電異常', '充電問題', '充電困難', '充電障礙'
                ],
                'positive_terms': [
                    '快', '快速', '急速', '秒充', '閃充', '充電快', '充電很快', '充電超快', '充電迅速', '充電不用等',
                    '充電馬上', '充電即時', '充電即刻', '快充', '快充支援', '快充功能', '快充效果', '快充表現',
                    '快充體驗', '快充協議', '快充技術', '快充標準', '快充速度', '閃充技術', '閃充速度', '閃充效果',
                    '閃充體驗', '閃充表現', '急速充電', '極速充電', '超快充'
                ],
                'negative_terms': [
                    '慢', '龜速', '久', '等很久', '充電慢', '充電很慢', '充電超慢', '充電緩慢', '充電等待',
                    '充電等很久', '慢充', '慢速充電', '慢速', '充電拖延', '充電拖很久', '充電不穩', '充電不順',
                    '充電卡頓', '充電卡住', '充電不良', '充電異常', '充電問題', '充電困難', '充電障礙'
                ]
            },
            '接口相容性': {
                'keywords': [
                    'Type-C', 'Lightning', 'USB', '接口', '孔', '線', '相容', '通用', '萬用', '多口', 'Micro USB',
                    'USB-C', 'USB-A', 'USB3.0', 'USB2.0', 'PD', 'QC', '快充協議', '充電協議', '支援', '不支援',
                    '支援多種', '支援多口', '支援多設備', '支援多協議', '多設備', '多協議', '多裝置', '多平台', '跨平台',
                    '蘋果', '安卓', 'iPhone', 'Android', 'iPad', 'Mac', 'Windows', '筆電', '手機', '平板',
                    '轉接頭', '轉接線', '轉接', '轉換', '轉換頭', '轉換線', '轉換器', '轉接器', '插頭', '插孔',
                    '插座', '插槽', '插入', '插拔', '插合', '插接', '插配', '插合性', '插配性', '插接性',
                    '相容性', '不相容', '兼容', '不兼容', '兼容性', '不兼容性', '適用', '不適用', '適配', '不適配',
                    '支援Type-C', '支援Lightning', '支援USB', '支援Micro USB', '支援USB-C', '支援USB-A',
                    '支援PD', '支援QC', '支援快充', '支援充電協議', '支援多種協議', '支援多種設備', '支援多種平台'
                ],
                'positive_terms': [
                    '通用', '相容', '萬用', '多口', '齊全', '支援', '支援多種', '支援多口', '支援多設備', '支援多協議',
                    '多設備', '多協議', '多裝置', '多平台', '跨平台', '適用', '適配', '兼容', '兼容性', '支援Type-C',
                    '支援Lightning', '支援USB', '支援Micro USB', '支援USB-C', '支援USB-A', '支援PD', '支援QC',
                    '支援快充', '支援充電協議', '支援多種協議', '支援多種設備', '支援多種平台', '插頭齊全', '插孔齊全'
                ],
                'negative_terms': [
                    '不相容', '沒有', '缺少', '只有', '不支援', '不兼容', '不兼容性', '不適用', '不適配', '不支援Type-C',
                    '不支援Lightning', '不支援USB', '不支援Micro USB', '不支援USB-C', '不支援USB-A', '不支援PD',
                    '不支援QC', '不支援快充', '不支援充電協議', '不支援多種協議', '不支援多種設備', '不支援多種平台',
                    '插頭缺少', '插孔缺少', '插頭不齊', '插孔不齊', '插頭不合', '插孔不合', '插頭不配', '插孔不配'
                ]
            },
            '外觀材質': {
                'keywords': [
                    '外觀', '質感', '材質', '設計', '顏色', '美', '醜', '好看', '塑膠', '金屬', '鋁合金', '霧面',
                    '時尚', '流行', '外型', '外表', '外觀設計', '外觀造型', '外觀顏色', '外觀質感', '外觀材質', '外觀精緻',
                    '外觀高級', '外觀漂亮', '外觀美觀', '外觀時尚', '外觀流行', '外觀大方', '外觀簡約', '外觀現代',
                    '外觀新穎', '外觀獨特', '外觀有型', '外觀有設計感', '外觀有質感', '外觀有特色', '外觀有亮點',
                    '手感', '手感好', '手感差', '手感舒適', '手感粗糙', '手感細膩', '手感滑順', '手感扎實',
                    '顏色漂亮', '顏色好看', '顏色醜', '顏色單調', '顏色豐富', '顏色多樣', '顏色時尚', '顏色流行',
                    '顏色新穎', '顏色獨特', '顏色有質感', '顏色有設計感', '顏色有特色', '顏色有亮點',
                    '金屬感', '金屬質感', '金屬外觀', '金屬設計', '鋁合金外觀', '鋁合金設計', '鋁合金質感',
                    '塑膠感', '塑膠外觀', '塑膠設計', '塑膠質感', '霧面外觀', '霧面設計', '霧面質感',
                    '亮面', '亮面外觀', '亮面設計', '亮面質感', '磨砂', '磨砂外觀', '磨砂設計', '磨砂質感'
                ],
                'positive_terms': [
                    '質感', '高級', '精緻', '好看', '美', '有質感', '時尚', '流行', '漂亮', '大方', '簡約', '現代',
                    '新穎', '獨特', '有型', '有設計感', '有特色', '有亮點', '手感好', '手感舒適', '手感細膩',
                    '手感滑順', '手感扎實', '顏色漂亮', '顏色好看', '顏色豐富', '顏色多樣', '顏色時尚', '顏色流行',
                    '顏色新穎', '顏色獨特', '顏色有質感', '顏色有設計感', '顏色有特色', '顏色有亮點', '金屬感',
                    '金屬質感', '金屬外觀', '金屬設計', '鋁合金外觀', '鋁合金設計', '鋁合金質感', '霧面外觀',
                    '霧面設計', '霧面質感', '亮面外觀', '亮面設計', '亮面質感', '磨砂外觀', '磨砂設計', '磨砂質感'
                ],
                'negative_terms': [
                    '廉價', '醜', '塑膠感', '粗糙', '手感差', '顏色醜', '顏色單調', '塑膠外觀', '塑膠設計',
                    '塑膠質感', '手感粗糙', '手感不佳', '手感滑膩', '手感鬆散', '手感不舒服', '手感不順',
                    '外觀單調', '外觀老氣', '外觀普通', '外觀無特色', '外觀無亮點', '外觀不美觀', '外觀不時尚',
                    '外觀不流行', '外觀不大方', '外觀不簡約', '外觀不現代', '外觀不新穎', '外觀不獨特',
                    '外觀沒設計感', '外觀沒質感', '外觀沒特色', '外觀沒亮點', '金屬感差', '金屬質感差',
                    '金屬外觀差', '金屬設計差', '鋁合金外觀差', '鋁合金設計差', '鋁合金質感差', '霧面外觀差',
                    '霧面設計差', '霧面質感差', '亮面外觀差', '亮面設計差', '亮面質感差', '磨砂外觀差',
                    '磨砂設計差', '磨砂質感差'
                ]
            },
            '價格': {
                'keywords': [
                    '價格', '價錢', '價值', '價位', '售價', '標價', '定價', '市價', '元', '塊', '金額',
                    '貴', '便宜', '划算', '超值', '平價', '優惠', '折扣', '促銷', '特價', '性價比', 'CP', 'CP值',
                    '值得', '不值', '昂貴', '便宜貨', '高價', '低價', '價格合理', '價格不合理', '價格偏高', '價格偏低',
                    '坑錢', '花費', '花錢', '花太多', '花很少', '划不來', '划得來', '物超所值', '物有所值', '價格實惠', '價格親民'
                ],
                'positive_terms': [
                    '便宜', '划算', '超值', '值得', 'CP值高', '優惠', '折扣', '促銷', '特價', '性價比高',
                    '價格合理', '價格親民', '價格實惠', '物超所值', '物有所值', '平價', '划得來'
                ],
                'negative_terms': [
                    '貴', '昂貴', '不值', 'CP值低', '坑錢', '高價', '價格偏高', '價格不合理', '花太多', '划不來',
                    '價格過高', '價格太貴', '價格太高', '價格不親民', '價格不實惠', '價格太離譜', '不划算'
                ]
            }
        }
        
        # 建立情感分類器（簡化版，實際應該微調）
        self.sentiment_classifier = pipeline(
            'sentiment-analysis',
            model='uer/roberta-base-finetuned-jd-binary-chinese',
            device=0 if torch.cuda.is_available() else -1
        )

    def _ensure_text(self, text) -> str:
        """確保輸入是字串；若為 NaN/None 則回傳空字串，其他類型轉為 str。"""
        try:
            # pandas 的 NaN 是 float 且 pandas.isna 可檢測
            if pd.isna(text):
                return ''
        except Exception:
            # pd.isna 可能在某些非標準物件上丟例外，保守處理
            pass

        if isinstance(text, str):
            return text
        return str(text)
    
    def extract_aspect_mentions(self, text: str) -> Dict[str, List[str]]:
        """提取文本中提及的產品方面"""
        mentions = {}
        # 確保 text 為字串，避免 NaN/float 等型別造成 .split 錯誤
        text = self._ensure_text(text)

        for aspect, info in self.aspects.items():
            # 找到包含該方面關鍵詞的句子
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
        分析特定方面的情感
        返回：情感極性、分數、相關文本片段
        """
        aspect_info = self.aspects.get(aspect)
        if not aspect_info:
            return None

        # 確保 text 為字串，避免傳入 NaN/float 時呼叫 split 失敗
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
        
        # 使用 RoBERTa 進行情感分析
        combined_text = '。'.join(relevant_sentences)
        
        try:
            result = self.sentiment_classifier(combined_text[:512])[0]
            
            # 轉換為 -1 到 1 的分數
            if result['label'] == 'positive':
                score = result['score']
            else:
                score = -result['score']
            
            # 進階：基於正面/負面詞彙進行微調
            pos_count = sum(1 for term in aspect_info['positive_terms'] if term in combined_text)
            neg_count = sum(1 for term in aspect_info['negative_terms'] if term in combined_text)
            
            # 調整分數
            if pos_count > neg_count:
                score = max(score, 0.2)
            elif neg_count > pos_count:
                score = min(score, -0.2)
            
            # 提高正負面門檻，減少偏頗
            return {
                'mentioned': True,
                'sentiment': 'positive' if score > 0.2 else ('negative' if score < -0.4 else 'neutral'),
                'score': float(score),
                'confidence': float(result['score']),
                'evidence': relevant_sentences[:3],  # 最多保留3個例證
                'pos_mentions': pos_count,
                'neg_mentions': neg_count
            }
        
        except Exception as e:
            print(f"情感分析錯誤: {e}")
            return {
                'mentioned': True,
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'evidence': relevant_sentences[:3]
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
        
        iterator = tqdm(texts) if show_progress else texts
        
        for text in iterator:
            # 確保文本為字串再分析
            safe_text = self._ensure_text(text)
            analysis = self.analyze_full_review(safe_text)

            row = {'text': safe_text[:200]}  # 保留前200字
            
            for aspect, result in analysis.items():
                row[f'{aspect}_mentioned'] = result['mentioned']
                row[f'{aspect}_score'] = result['score'] if result['mentioned'] else None
                row[f'{aspect}_sentiment'] = result['sentiment'] if result['mentioned'] else None
            
            all_results.append(row)
        
        return pd.DataFrame(all_results)


# =======================================
# 2. 詞嵌入與語義增強
# =======================================

class SemanticEmbeddingAnalyzer:
    """
    利用 RoBERTa 隱藏層向量生成詞嵌入
    實現語義相似度、多語區詞彙對應、語義聚類
    """
    
    def __init__(self, model_name='ckiplab/bert-base-chinese'):
        print("📥 載入詞嵌入模型...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 詞彙庫
        self.vocabulary = {}
        self.embeddings_cache = {}
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """
        獲取單詞的上下文感知詞嵌入
        使用 RoBERTa 隱藏層向量
        """
        if word in self.embeddings_cache:
            return self.embeddings_cache[word]
        
        # 將詞彙放入簡單句子中以獲得上下文
        text = f"這個產品的{word}很好"
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用最後一層的隱藏狀態
            hidden_states = outputs.last_hidden_state
            
            # 找到目標詞的位置（簡化處理）
            word_embedding = hidden_states[0].mean(dim=0).cpu().numpy()
        
        self.embeddings_cache[word] = word_embedding
        return word_embedding
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """獲取整段文本的嵌入向量"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 [CLS] token 的嵌入作為文本表示
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
        return cls_embedding
    
    def calculate_semantic_similarity(self, word1: str, word2: str) -> float:
        """
        計算兩個詞彙的語義相似度
        應用一：多語區詞彙對應
        """
        emb1 = self.get_word_embedding(word1)
        emb2 = self.get_word_embedding(word2)
        
        # 計算餘弦相似度
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        return float(similarity)
    
    def find_similar_terms(self, target_word: str, candidate_words: List[str], 
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        找到與目標詞語義相似的詞彙
        可用於發現同義詞、地區用語差異
        """
        target_emb = self.get_word_embedding(target_word)
        
        similarities = []
        for word in candidate_words:
            if word == target_word:
                continue
            word_emb = self.get_word_embedding(word)
            sim = cosine_similarity([target_emb], [word_emb])[0][0]
            similarities.append((word, float(sim)))
        
        # 排序並返回 top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def semantic_clustering(self, words: List[str], n_clusters: int = 5) -> Dict:
        """
        語義聚類：將詞彙按語義自動分組
        例如：「行動電源」、「充電寶」、「尿袋」會聚成一類
        """
        print(f"🔄 對 {len(words)} 個詞彙進行語義聚類...")
        
        # 獲取所有詞彙的嵌入
        embeddings = []
        valid_words = []
        
        for word in words:
            try:
                emb = self.get_word_embedding(word)
                embeddings.append(emb)
                valid_words.append(word)
            except:
                continue
        
        embeddings = np.array(embeddings)
        
        # K-means 聚類
        kmeans = KMeans(n_clusters=min(n_clusters, len(valid_words)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # 組織結果
        cluster_dict = {}
        for i in range(n_clusters):
            cluster_words = [valid_words[j] for j in range(len(valid_words)) if clusters[j] == i]
            if cluster_words:
                cluster_dict[f'群集_{i+1}'] = cluster_words
        
        return cluster_dict
    
    def visualize_semantic_space(self, texts: List[str], labels: List[str] = None,
                                 method: str = 'umap') -> go.Figure:
        """
        應用二：語義可視化
        使用 UMAP 或 t-SNE 降維並視覺化詞向量空間
        """
        print(f"🎨 使用 {method.upper()} 進行語義空間視覺化...")
        
        # 獲取文本嵌入
        embeddings = []
        for text in texts:
            emb = self.get_text_embedding(text)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # 降維
        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:  # t-SNE
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        
        # 創建視覺化
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'text': [t[:50] + '...' for t in texts],
            'label': labels if labels else ['未分類'] * len(texts)
        })
        
        fig = px.scatter(
            df, x='x', y='y', color='label',
            hover_data=['text'],
            title=f'語義空間視覺化 ({method.upper()})',
            width=800, height=600
        )
        
        fig.update_layout(
            xaxis_title=f'{method.upper()} 維度 1',
            yaxis_title=f'{method.upper()} 維度 2'
        )
        
        return fig


# =======================================
# 3. 零樣本學習技術偵測
# =======================================

class ZeroShotTechDetector:
    """
    使用零樣本學習即時識別新興技術詞
    無需預先標註，自動偵測 GaN、Qi2 等技術關鍵詞
    """
    
    def __init__(self):
        print("📥 載入零樣本學習模型...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 定義技術類別（可動態擴展）
        self.tech_categories = {
            '充電技術': ['快充', 'PD', 'QC', '閃充', '無線充電', 'Qi', 'Qi2'],
            '材料技術': ['GaN', '氮化鎵', '矽材料', '石墨烯'],
            '電池技術': ['鋰電池', '固態電池', '磷酸鐵鋰', '三元鋰'],
            '接口標準': ['Type-C', 'Lightning', 'USB-A', 'Micro USB'],
            '安全技術': ['過充保護', '溫控', '短路保護', 'BMS']
        }
    
    def detect_technology(self, text: str, threshold: float = 0.5) -> List[Dict]:
        """
        零樣本偵測文本中的技術類別
        """
        categories = list(self.tech_categories.keys())
        
        result = self.classifier(
            text,
            candidate_labels=categories,
            multi_label=True
        )
        
        detected_techs = []
        for label, score in zip(result['labels'], result['scores']):
            if score > threshold:
                detected_techs.append({
                    'category': label,
                    'confidence': float(score),
                    'keywords': self.tech_categories[label]
                })
        
        return detected_techs
    
    def extract_emerging_keywords(self, texts: List[str], min_frequency: int = 3) -> Dict:
        """
        從大量文本中提取新興技術關鍵詞
        """
        # 分詞並統計詞頻
        all_words = []
        for text in texts:
            words = list(jieba.cut(text))
            all_words.extend(words)
        
        from collections import Counter
        word_freq = Counter(all_words)
        
        # 篩選可能的技術詞（2-4個字，出現頻率適中）
        tech_candidates = [
            word for word, freq in word_freq.items()
            if 2 <= len(word) <= 4 and min_frequency <= freq <= len(texts) * 0.3
        ]
        
        # 使用零樣本分類驗證
        emerging_techs = {}
        
        for word in tech_candidates[:50]:  # 限制數量避免太慢
            sample_text = f"這個產品使用了{word}技術"
            detection = self.detect_technology(sample_text, threshold=0.4)
            
            if detection:
                category = detection[0]['category']
                if category not in emerging_techs:
                    emerging_techs[category] = []
                emerging_techs[category].append({
                    'keyword': word,
                    'frequency': word_freq[word],
                    'confidence': detection[0]['confidence']
                })
        
        return emerging_techs
    
    def track_tech_trends(self, df: pd.DataFrame, date_col: str = 'date',
                         text_col: str = 'text') -> pd.DataFrame:
        """
        追蹤技術詞的時間趨勢
        返回各技術類別隨時間的討論熱度
        """
        df[date_col] = pd.to_datetime(df[date_col])
        df['month'] = df[date_col].dt.to_period('M')
        
        trends = []
        
        for month in df['month'].unique():
            month_data = df[df['month'] == month]
            month_texts = ' '.join(month_data[text_col].tolist())
            
            # 偵測該月的技術分佈
            for category, keywords in self.tech_categories.items():
                mention_count = sum(
                    text.count(keyword) 
                    for text in month_data[text_col] 
                    for keyword in keywords
                )
                
                trends.append({
                    'month': str(month),
                    'category': category,
                    'mentions': mention_count,
                    'docs': len(month_data)
                })
        
        trends_df = pd.DataFrame(trends)
        trends_df['mention_rate'] = trends_df['mentions'] / trends_df['docs']
        
        return trends_df


# =======================================
# 4. 整合分析引擎
# =======================================

class AdvancedNLPEngine:
    """
    PowerPulse 進階 NLP 引擎整合
    結合 ABSA、詞嵌入、零樣本學習
    """
    
    def __init__(self):
        print("🚀 初始化 PowerPulse 進階 NLP 引擎...")
        self.absa = AspectBasedSentimentAnalyzer()
        self.semantic = SemanticEmbeddingAnalyzer()
        self.tech_detector = ZeroShotTechDetector()
        print("✅ NLP 引擎就緒！")
    
    def full_analysis(self, texts: List[str], dates: List[str] = None) -> Dict:
        """
        執行完整的 NLP 分析管線
        """
        print("\n" + "="*50)
        print("開始完整 NLP 分析...")
        print("="*50 + "\n")
        
        results = {}
        
        # 1. 方面級情感分析
        print("📊 執行方面級情感分析...")
        absa_results = self.absa.batch_analyze(texts)
        results['sentiment_analysis'] = absa_results
        
        # 2. 語義聚類分析
        print("\n🔄 執行語義聚類...")
        # 提取高頻詞進行聚類
        all_words = []
        for text in texts:
            words = list(jieba.cut(text))
            all_words.extend([w for w in words if len(w) >= 2])
        
        from collections import Counter
        top_words = [word for word, _ in Counter(all_words).most_common(50)]
        
        clusters = self.semantic.semantic_clustering(top_words, n_clusters=5)
        results['semantic_clusters'] = clusters
        
        # 3. 語義空間視覺化
        print("\n🎨 生成語義空間視覺化...")
        semantic_viz = self.semantic.visualize_semantic_space(
            texts[:100],  # 限制數量
            method='umap'
        )
        results['semantic_visualization'] = semantic_viz
        
        # 4. 零樣本技術偵測
        print("\n🔍 執行零樣本技術偵測...")
        emerging_techs = self.tech_detector.extract_emerging_keywords(texts)
        results['emerging_technologies'] = emerging_techs
        
        # 5. 技術趨勢追蹤（如果有日期）
        if dates:
            print("\n📈 追蹤技術趨勢...")
            df = pd.DataFrame({'text': texts, 'date': dates})
            tech_trends = self.tech_detector.track_tech_trends(df)
            results['tech_trends'] = tech_trends
        
        print("\n✅ 完整 NLP 分析完成！")
        return results
    
    def generate_insights_report(self, results: Dict) -> str:
        """生成洞察報告"""
        report = []
        report.append("="*60)
        report.append("PowerPulse AI - 進階 NLP 分析報告")
        report.append("="*60)
        
        # 情感分析摘要
        sentiment_df = results['sentiment_analysis']
        report.append("\n【方面級情感分析】")
        
        aspects = ['重量體積', '充電速度', '接口相容性', '外觀材質', '價格']
        for aspect in aspects:
            mentioned = sentiment_df[f'{aspect}_mentioned'].sum()
            if mentioned > 0:
                avg_score = sentiment_df[sentiment_df[f'{aspect}_mentioned']][f'{aspect}_score'].mean()
                report.append(f"  • {aspect}: 提及 {mentioned} 次, 平均情感 {avg_score:.2f}")
        
        # 語義聚類摘要
        report.append("\n【語義聚類發現】")
        clusters = results['semantic_clusters']
        for cluster_name, words in list(clusters.items())[:3]:
            report.append(f"  • {cluster_name}: {', '.join(words[:5])}")
        
        # 新興技術
        report.append("\n【新興技術偵測】")
        emerging = results['emerging_technologies']
        for category, tech_list in emerging.items():
            top_tech = tech_list[0] if tech_list else None
            if top_tech:
                report.append(f"  • {category}: {top_tech['keyword']} (提及 {top_tech['frequency']} 次)")
        
        report.append("\n" + "="*60)
        
        return '\n'.join(report)


# =======================================
# 使用範例
# =======================================

if __name__ == "__main__":
    # 初始化引擎
    nlp_engine = AdvancedNLPEngine()

    # 修正資料夾名稱: crawelers_result -> crawlers_result
    data = pd.read_csv('./AICompetition/crawlers_result/data_mobile.csv')
    data1=list(data['title'])
    data2=list(data['comments'])
    data3=data1+data2
    data3=[x for x in data3 if str(x).lower() not in ('nan', 'none')]
    # 測試數據
    sample_texts = [
        "這個行動電源超輕薄，充電速度很快，支援PD快充，質感也不錯",
        "充電寶太重了，而且充電很慢，不支援Type-C很不方便",
        "GaN技術真的不錯，充電超快，就是價格有點貴",
        "這款尿袋很輕便，但容量太小了，外觀設計很美",
        "支援Qi2無線充電很方便，不過價格偏高"
    ]
    
    dates = ['2025-11-15'] * len(data3)
    
    # 執行完整分析
    results = nlp_engine.full_analysis(data3, dates)
    
    # 生成報告
    report = nlp_engine.generate_insights_report(results)
    print(report)
    
    # 保存結果
    results['sentiment_analysis'].to_csv('absa_results.csv', index=False, encoding='utf-8-sig')
    print("\n💾 結果已保存至 absa_results.csv")