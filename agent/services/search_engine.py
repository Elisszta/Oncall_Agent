from typing import List, Dict
from services.document_store import store

def search_v1(query: str) -> List[Dict]:
    results = []
    if not query:
        return results
        
    for doc in store.get_all():
        text_lower = doc.text.lower()
        query_lower = query.lower()
        if query_lower in text_lower:
            # count occurrences for basic scoring
            score = text_lower.count(query_lower)
            # generate snippet
            idx = text_lower.find(query_lower)
            start = max(0, idx - 30)
            end = min(len(doc.text), idx + len(query) + 30)
            snippet = doc.text[start:end]
            if start > 0: snippet = "..." + snippet
            if end < len(doc.text): snippet += "..."
            
            results.append({
                "id": doc.id,
                "title": doc.title,
                "snippet": snippet,
                "score": float(score)
            })
            
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def search_v2(query: str, use_rrf: bool = True, vector_weight: float = 0.8) -> List[Dict]:
    from services.embedding import vec_store
    from services.reranker import rerank
    
    if not query:
        return []
        
    keyword_results = search_v1(query)
    semantic_scores = vec_store.search(query)
    
    all_docs = store.get_all()
    
    # rank 1-indexed dictionaries
    kw_ranks = {res["id"]: rank for rank, res in enumerate(keyword_results, 1)}
    
    # sort semantic scores descending to get rank
    sem_sorted = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
    sem_ranks = {doc_id: rank for rank, (doc_id, score) in enumerate(sem_sorted, 1)}
    
    final_candidates = []
    k = 60
    
    keyword_weight = 1.0 - vector_weight
    
    for doc in all_docs:
        kw_rank = kw_ranks.get(doc.id, None)
        sem_rank = sem_ranks.get(doc.id, None)
        
        kw_score = (1.0 / (k + kw_rank)) if kw_rank else 0.0
        sem_score = (1.0 / (k + sem_rank)) if sem_rank else 0.0
        
        if use_rrf:
            combined_score = (keyword_weight * kw_score) + (vector_weight * sem_score)
        else:
            combined_score = sem_score
            
        if combined_score > 0:
            # generate basic snippet
            text_lower = doc.text.lower()
            idx = text_lower.find(query.lower())
            if idx == -1: idx = 0
            start = max(0, idx - 30)
            end = min(len(doc.text), idx + len(query) + 30)
            snippet = doc.text[start:end]
            if start > 0: snippet = "..." + snippet
            if end < len(doc.text): snippet += "..."
            
            final_candidates.append({
                "id": doc.id,
                "title": doc.title,
                "text": doc.text,  # passing text for reranker
                "snippet": snippet,
                "score": combined_score,
                "rrf_score": combined_score
            })
            
    # Sort by RRF score
    final_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Extract top 5 for cross-encoder reranking
    reranked = rerank(query, final_candidates, top_k=5)
    
    # Remove internal fields to save bandwidth and align with API spec
    for r in reranked:
        r.pop("text", None)
        r.pop("rrf_score", None)
        
    return reranked
