from services.embedding import vec_store
import os

# Load the store
vec_store.load("data/vector_store.pkl")

# Print first 10 chunks of sop-001
doc_id = "sop-001"
if doc_id in vec_store.doc_chunks:
    print(f"\nChunks for {doc_id} (New Logical Chunking) total {len(vec_store.doc_chunks[doc_id])} chunks:")
    for i, chunk in enumerate(vec_store.doc_chunks[doc_id][:10]):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print("-" * 20)
else:
    print(f"Doc {doc_id} not found in store.")
