import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pickle
import os

# Model configuration
MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
MODELS_DIR = os.path.join(os.getcwd(), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'bge-small-zh-v1.5')

# Load model lazily
_model = None
def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        
        if os.path.exists(MODEL_PATH):
            print(f"Loading embedding model from local: {MODEL_PATH}")
            _model = SentenceTransformer(MODEL_PATH, device=device)
        else:
            print(f"Downloading {MODEL_NAME} from HuggingFace...")
            _model = SentenceTransformer(MODEL_NAME, device=device)
            os.makedirs(MODELS_DIR, exist_ok=True)
            _model.save(MODEL_PATH)
            print(f"Model saved to: {MODEL_PATH}")
            
    return _model

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += (chunk_size - overlap)
    return chunks

class VectorStore:
    def __init__(self):
        # Maps doc_id -> list of chunk embeddings (numpy arrays)
        self.doc_embeddings: Dict[str, np.ndarray] = {}
        # Maps doc_id -> list of chunk texts
        self.doc_chunks: Dict[str, List[str]] = {}
        # Maps doc_id -> metadata (like title) for Phase 3 fast access
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}

    def add_document(self, id: str, text: str, title: str = ""):
        chunks = chunk_text(text)
        if not chunks:
            return
        self.doc_chunks[id] = chunks
        self.doc_metadata[id] = {"title": title}
        
        # Batch encode chunks
        model = get_model()
        # SentenceTransformers encode returns numpy arrays by default
        embeddings = model.encode(chunks, normalize_embeddings=True)
        self.doc_embeddings[id] = embeddings

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            data = {
                "embeddings": self.doc_embeddings,
                "chunks": self.doc_chunks,
                "metadata": self.doc_metadata
            }
            pickle.dump(data, f)

    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.doc_embeddings = data.get("embeddings", {})
                self.doc_chunks = data.get("chunks", {})
                self.doc_metadata = data.get("metadata", {})
            return True
        except Exception:
            return False

    def search(self, query: str, top_k: int = 3) -> Dict[str, float]:
        """Returns map of doc_id -> mean similarity score of top-K chunks"""
        if not self.doc_embeddings:
            return {}
            
        model = get_model()
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        
        results = {}
        for doc_id, chunk_embs in self.doc_embeddings.items():
            # compute dot product for all chunks
            similarities = np.dot(chunk_embs, query_emb)
            
            # Top-K mean pooling
            if len(similarities) <= top_k:
                score = np.mean(similarities)
            else:
                # get top-k elements using partition (faster than full sort)
                top_k_idx = np.argpartition(similarities, -top_k)[-top_k:]
                top_k_sims = similarities[top_k_idx]
                score = np.mean(top_k_sims)
                
            results[doc_id] = float(score)
        return results

vec_store = VectorStore()
