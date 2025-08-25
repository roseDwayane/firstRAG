import csv, numpy as np, faiss, os
from textwrap import dedent

# ==== 0) 開關：要用哪個向量模型來「生 embedding」？ ====
USE_OLLAMA = False   # True=沿用你的 Ollama bge-base-zh；False=改用 sentence-transformers

if USE_OLLAMA:
    import ollama
    EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-zh-v1.5-gguf'  # 你原本的
else:
    from sentence_transformers import SentenceTransformer
    # 建議中文/多語模型（all-MiniLM-L6-v2 偏英文）
    EMBEDDING_MODEL = 'BAAI/bge-m3'  # 或 'paraphrase-multilingual-MiniLM-L12-v2'
    st_model = SentenceTransformer(EMBEDDING_MODEL)

# ==== 1) 讀取表格 ====
ROWS = []
with open('records.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        pay  = (r.get('InvoicePayment', '') or '').strip()
        cat  = (r.get('AccountingCategory', '') or '').strip()
        summ = (r.get('Summary', '') or '').strip()
        if not pay and not cat and not summ:
            continue
        ROWS.append({'InvoicePayment': pay, 'AccountingCategory': cat, 'Summary': summ})

print(f'Loaded {len(ROWS)} rows')

# ==== 2) 準備要嵌入的文字（把多欄位組成一段） ====
DOCS = []
for row in ROWS:
    text_for_embedding = (
        f"InvoicePayment: {row['InvoicePayment']}. "
        f"AccountingCategory: {row['AccountingCategory']}. "
        f"Summary: {row['Summary']}"
    )
    DOCS.append(text_for_embedding)

# ==== 3) 產生向量 ====
def embed_texts(texts):
    if USE_OLLAMA:
        # Ollama 一次一段；也可自行批次呼叫以降低 overhead
        vecs = []
        for t in texts:
            emb = ollama.embed(model=EMBEDDING_MODEL, input=t)['embeddings'][0]
            vecs.append(emb)
        arr = np.array(vecs, dtype='float32')
    else:
        # sentence-transformers 可一次批量編碼，normalize_embeddings=True → 直接用 cosine
        arr = st_model.encode(texts, normalize_embeddings=True).astype('float32')
    # 若用 Ollama，建議自行做 L2 normalize 以便用 IP 當 cosine
    #（用 sbert 已經正規化過則不會改變值）
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr

EMB = embed_texts(DOCS).astype('float32')
d = EMB.shape[1]

# ==== 4) 建 FAISS 索引（用內積，等價 cosine；並保留 id 對應） ====
index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
ids = np.arange(len(DOCS)).astype('int64')
index.add_with_ids(EMB, ids)
print("index.ntotal =", index.ntotal)

# 為了能回傳原欄位，做一個 id → metadata 的對照表
ID2META = {i: ROWS[i] for i in range(len(ROWS))}
ID2TEXT = {i: DOCS[i] for i in range(len(DOCS))}

# ==== 5) 查詢函式（輸入問題字串，回傳 Top-k + 分數 + 原欄位） ====
def search(query_text, k=5):
    q = embed_texts([query_text]).astype('float32')
    D, I = index.search(q, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:  # 當 k 超過資料量時，FAISS 可能回 -1
            continue
        results.append({
            'score': float(score),            # cosine 相似度
            'text': ID2TEXT[int(idx)],
            'meta': ID2META[int(idx)]
        })
    return results

# ==== 5) 對話 ====
while(1):
    input_query = input('Ask me a question: ')
    payment = "paid" # 這個要改

    rst = search(input_query+payment, k=3)
    for item in rst:
        cat = item['meta']['AccountingCategory']
        summ = item['meta']['Summary']
        pay = item['meta']['InvoicePayment']
        if pay == payment:
            print(f" - (similarity: {item['score']:.2f}) {cat} {summ} {pay}")
            # return
        else:
            continue

    # ----------------------
    # retrieved = retrieve(input_query)

    # print('Retrieved rows:')
    # for item, sim in retrieved:
    #     cat = item['meta']['AccountingCategory']
    #     summ = item['meta']['Summary']
    #     pay = item['meta']['InvoicePayment']
    #     print(f' - (similarity: {sim:.2f}) [{cat}] {summ} {pay}')

    # # 把「要給模型看的上下文」做成可讀格式（不使用 f-string 內嵌反斜線的寫法）
    # context_lines = [f"[{item['meta']['AccountingCategory']}] {item['meta']['Summary']} | InvoicePayment: {item['meta']['InvoicePayment']}"
    # for item, _ in retrieved]
    # context = "\n".join(context_lines)

    # instruction_prompt = dedent(f"""
    #     你是一個嚴格的單一標籤分類器。任務：根據「使用者輸入」中的兩個欄位
    #     - Summary
    #     - InvoicePayment
    #     以及我提供的「範例資料(context)」，推斷並輸出對應的 AccountingCategory。

    #     【輸入與範例】
    #     - 使用者輸入只會包含兩個資訊：
    #     Summary: <使用者的摘要文字>
    #     InvoicePayment: <received|paid 或其他值>
    #     - 我將提供多行範例資料作為 context。每行都是一筆已知標註，格式為：
    #     [AccountingCategory] <Summary 文本> | InvoicePayment: <值>

    #     【判斷規則（依序）】
    #     1) 只能使用提供的 context 做比對與歸納，不得編造新知識。
    #     2) 將「使用者輸入的 (Summary, InvoicePayment)」與 context 各行的 (Summary, InvoicePayment) 做語意相似度比對。
    #     3) 以最相近的類別為答案；如需多例參考，採 k=5 多數決；若票數相同，取相似度最高者所屬類別。
    #     4) 只允許輸出「context 中出現過」的類別名稱（大小寫與標點需與 context 完全一致）。
    #     5) 若無法從 context 合理判斷，輸出：Unknown。

    #     【輸出格式（極嚴格）】
    #     - 只輸出最終的 AccountingCategory。
    #     - 僅一行，無任何前後綴、說明、引號、標點或額外字元。
    #     - 範例：如果判斷為 留抵稅額，就只輸出 留抵稅額

    #     以下是最相關的資料：
    #     {context}
    #     （僅能依上述內容做判斷）
    #     """).strip()

    # stream = ollama.chat(
    #     model=LANGUAGE_MODEL,
    #     messages=[
    #         {'role': 'system', 'content': instruction_prompt},
    #         {'role': 'user',   'content': input_query},
    #     ],
    #     stream=True,
    # )

    # print('Chatbot response:')
    # for chunk in stream:
    #     print(chunk['message']['content'], end='', flush=True)
    
    print("\n------------\n")
