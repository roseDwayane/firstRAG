import csv
import ollama

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
    context_lines = [f"[{item['meta']['AccountingCategory']}] {item['meta']['Summary']}"
                    for item, _ in retrieved]
    context = "\n".join(context_lines)

    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context (each line is one record) to answer the question. "
        "Don't make up any new information:\n"
        f"{context}"
    )

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
