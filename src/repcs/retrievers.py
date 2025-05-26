from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from .config import DEVICE, TOP_K

class SparseRetriever:
    """Lexical BM25 retriever."""
    def __init__(self, docs: List[str]):
        self.docs = docs
        tokenized = [d.split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(query.split())
        idx = np.argsort(scores)[::-1][:TOP_K]
        return [(self.docs[i], float(scores[i])) for i in idx]

class DenseRetriever:
    """Cosine‑similarity dense retriever backed by sentence‑transformers & FAISS."""
    def __init__(self, docs: List[str], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 32):
        self.docs = docs
        self.encoder = SentenceTransformer(model_name, device=DEVICE)
        self.batch_size = batch_size
        doc_embs = self.encoder.encode(docs, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        self.dim = doc_embs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(doc_embs.astype('float32'))

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        q = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims, idx = self.index.search(q.astype('float32'), TOP_K)
        return [(self.docs[i], float(sims[0][j])) for j,i in enumerate(idx[0])]
