from services.embedding import get_model as get_embed
from services.reranker import get_model as get_rerank
import os

print("Starting model pre-loading/downloading...")
os.makedirs("models", exist_ok=True)

print("Pre-loading Embedding Model...")
get_embed()

print("Pre-loading Reranker Model...")
get_rerank()

print("All models successfully stored in ./models/")
