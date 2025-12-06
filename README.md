# NLP-Retrieval-Augmented-Generation-with-LangChain
此為清大自然語言處理課程的 HW4: Retrieval-Augmented Generation with LangChain

**Platform:** Local  
**Python version:** 3.10.18  
**OS:** Linux 6.8.0-65-generic  
**CPU:** x86_64 (24 cores)  
**GPU:** NVIDIA GeForce RTX 3090  

---

## 1. RAG 系統實作內容

本次作業建構了完整的 Retrieval-Augmented Generation (RAG) 系統，包含以下核心模組：

### **LLM 模型**
- Llama-3.2-1B（Ollama 本地部署）

### **Embedding 模型**
- jinaai/jina-embeddings-v2-base-en

### **向量資料庫**
- ChromaDB

### **Two-Stage Retrieval（兩階段檢索）**
1. **Base Retrieval**：向量相似度搜尋 Top-20  
2. **Reranking**：使用 cross-encoder/ms-marco-MiniLM-L-6-v2 重新排序，取 Top-3

### **System Prompt**
```
You are a helpful assistant.

Answer the question based ONLY on the following context.
Keep your answer concise and to the point.

Examples:
Q: How much of a day do cats spend sleeping?
A: Two thirds

Q: What is a group of cats called?
A: Clowder

Context:
{context}
```


### **相較於 baseline 的改進**

| 項目 | baseline | 我的版本 |
|------|----------|-----------|
| Retrieval | Top-k 直接使用 | Two-stage + Reranking |
| Prompt | 無 few-shot | Few-shot（更貼近任務格式） |
| 文件品質 | 無重排序 | Reranker 保證前 3 準確性 |

---

## 2. Prompt 對 RAG 效能的影響

| Prompt Strategy | EM | Recall@1 | Recall@5 |
|------------------|-------|------------|------------|
| Baseline Prompt | 54.00% | 67.33% | 70.67% |
| Strict Keyword Prompt | 17.33% | 67.33% | 70.67% |
| Few-Shot Prompt | **58.00%** | 67.33% | 70.67% |

### **結論**
- Few-shot > Instruction-based  
- 過度嚴格的抽取式 prompt 反而降低 EM  
- 少樣本 prompt 有助於保持固定格式輸出  

---

## 3. Retriever 輸入格式比較

| Data Format | EM | Recall@1 | Recall@5 |
|-------------|--------|--------------|--------------|
| Line-by-Line | 58.00% | 67.33% | 70.67% |
| Chunk-5 | **60.00%** | 66.67% | 70.67% |

### **結論**
- Chunking 讓文件更完整，有助於 Reranker  
- Recall 幾乎不變 → embedding model 才是真瓶頸  

---

## 4. 文件順序對生成結果的影響

| Order | EM | Recall@1 | Recall@5 |
|--------|--------|--------------|--------------|
| Best-first | **60.00%** | 66.67% | 70.67% |
| Worst-first | 50.67% | 66.67% | 70.67% |

### **結論**
- 小模型（1B）對 context 排序高度敏感  
- 最重要資訊必須放在最前面  
- Reranker 的存在同時改善「內容」與「順序」  

---

## 5. Counterfactual（反事實資訊）實驗

| Case | Fake Info | Model Behavior | Interpretation |
|------|-----------|----------------|----------------|
| 睡眠時間 | "cats never sleep" | 模型拒絕錯誤資訊 | 常識防護佳 |
| 群體名稱 | "cats group = gaggle" | 模型被騙 | 名詞定義最脆弱 |
| 甜食偏好 | "cats love sweets" | 混合 internal knowledge | 有初步衝突辨識 |

### **結論**
模型對不同類型資訊的脆弱度不同：  
- 常識 → 比較安全  
- 定義類 → 極易受錯誤 context 影響  

---

## 6. 其他補充分析

### **(1) 檢索瓶頸**
Recall@5 永遠停在 70.67%  
→ 代表 30% 答案根本無法被檢索到  

### **(2) Exact Match 過於嚴格**
如：  
- "2/3" vs "Two thirds" → 被判錯  

可改進：semantic matching、LLM-as-a-judge

### **(3) Cross-Encoder 延遲成本**
| Model | Speed | Accuracy |
|--------|---------|------------|
| Bi-Encoder | 快 | 中等 |
| Cross-Encoder | 慢 | 高 |

視應用需求權衡準確與速度。

---


