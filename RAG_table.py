import csv
import ollama
from textwrap import dedent

# ==== 1) 模型設定（建議用多語/中文 embedding）====
# 你目前用的是英文 bge-base-en，中文效果會差；換成多語或中文向量模型更好
#EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-zh-v1.5-gguf'  # 你期望的
LANGUAGE_MODEL  = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# ==== 2) 讀取表格 ====
# 先把你的試算表匯出成 CSV：records.csv，表頭至少有 InvoicePayment, AccountingCategory, Summary
ROWS = []
with open('records.csv', newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        pay  = str(r.get('InvoicePayment', '')).strip()
        cat = str(r.get('AccountingCategory', '')).strip()
        summ = (r.get('Summary', '') or '').strip()
        if not pay and not cat and not summ:
            continue
        ROWS.append({
            'InvoicePayment': pay,
            'AccountingCategory': cat,
            'Summary': summ
        })

print(f'Loaded {len(ROWS)} rows')

# ==== 3) 建索引（把每列組成一段文字去做向量，並保留欄位做 metadata）====
# VECTOR_DB 裡每筆是 dict：{"text": 索引文字, "embedding": 向量, "meta": 原欄位}
VECTOR_DB = []

def add_row_to_database(row):
    # 你要拿哪些欄位檢索，就組進來；加入提示詞能讓純數字類型（如 2455）更好被模型理解
    text_for_embedding = (
        f"InvoicePayment: {row['InvoicePayment']}"
        f"AccountingCategory is {row['AccountingCategory']}. "
        f"Summary: {row['Summary']}"
    )
    emb = ollama.embed(model=EMBEDDING_MODEL, input=text_for_embedding)['embeddings'][0]
    VECTOR_DB.append({'text': text_for_embedding, 'embedding': emb, 'meta': row})

for i, row in enumerate(ROWS, start=1):
    add_row_to_database(row)
    if i % 50 == 0 or i == len(ROWS):
        print(f'Indexed {i}/{len(ROWS)} rows')

# ==== 4) 檢索 ====
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(x*x for x in b) ** 0.5
    return dot / (na * nb)

def retrieve(query, top_n=5):
    q_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    scored = []
    for item in VECTOR_DB:
        sim = cosine_similarity(q_emb, item['embedding'])
        scored.append((item, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# ==== 5) 對話 ====
while(1):
    input_query = input('Ask me a question: ')
    retrieved = retrieve(input_query)

    print('Retrieved rows:')
    for item, sim in retrieved:
        cat = item['meta']['AccountingCategory']
        summ = item['meta']['Summary']
        pay = item['meta']['InvoicePayment']
        print(f' - (similarity: {sim:.2f}) [{cat}] {summ} {pay}')

    # 把「要給模型看的上下文」做成可讀格式（不使用 f-string 內嵌反斜線的寫法）
    
    context_lines = [f"[{item['meta']['AccountingCategory']}] {item['meta']['Summary']} | InvoicePayment: {item['meta']['InvoicePayment']}"
    for item, _ in retrieved]
    context = "\n".join(context_lines)

    instruction_prompt = dedent(f"""
        你是一個嚴格的單一標籤分類器。任務：根據「使用者輸入」中的兩個欄位
        - Summary
        - InvoicePayment
        以及我提供的「範例資料(context)」，推斷並輸出對應的 AccountingCategory。

        【輸入與範例】
        - 使用者輸入只會包含兩個資訊：
        Summary: <使用者的摘要文字>
        InvoicePayment: <received|paid 或其他值>
        - 我將提供多行範例資料作為 context。每行都是一筆已知標註，格式為：
        [AccountingCategory] <Summary 文本> | InvoicePayment: <值>

        【判斷規則（依序）】
        1) 只能使用提供的 context 做比對與歸納，不得編造新知識。
        2) 將「使用者輸入的 (Summary, InvoicePayment)」與 context 各行的 (Summary, InvoicePayment) 做語意相似度比對。
        3) 以最相近的類別為答案；如需多例參考，採 k=5 多數決；若票數相同，取相似度最高者所屬類別。
        4) 只允許輸出「context 中出現過」的類別名稱（大小寫與標點需與 context 完全一致）。
        5) 若無法從 context 合理判斷，輸出：Unknown。

        【輸出格式（極嚴格）】
        - 只輸出最終的 AccountingCategory。
        - 僅一行，無任何前後綴、說明、引號、標點或額外字元。
        - 範例：如果判斷為 留抵稅額，就只輸出 留抵稅額

        以下是最相關的資料：
        {context}
        （僅能依上述內容做判斷）
        """).strip()

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user',   'content': input_query},
        ],
        stream=True,
    )

    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    
    print("\n------------\n")
