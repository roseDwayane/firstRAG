# pip install faiss-cpu sentence-transformers
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

docs = ["段落1...", "段落2...", "段落3..."]
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(docs, normalize_embeddings=True).astype("float32")  # cosine→先正規化
d = emb.shape[1]

index = faiss.IndexIDMap(faiss.IndexFlatIP(d))  # 用內積=cosine
ids = np.arange(len(docs)).astype("int64")
index.add_with_ids(emb, ids)

q = model.encode(["我的問題是..."], normalize_embeddings=True).astype("float32")
D, I = index.search(q, k=3)  # 取 Top-3
print([docs[i] for i in I[0]])