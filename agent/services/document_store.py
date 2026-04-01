import os
from pydantic import BaseModel
from typing import List, Dict
from services.html_parser import parse_html

from services.embedding import vec_store

class Document(BaseModel):
    id: str
    title: str
    text: str

class DocumentStore:
    def __init__(self):
        self.documents: Dict[str, Document] = {}

    def add_document(self, id: str, html_content: str, skip_embedding: bool = False) -> Document:
        parsed = parse_html(html_content)
        doc = Document(id=id, title=parsed["title"], text=parsed["text"])
        self.documents[id] = doc
        
        # Populate Phase 2 embedding store if needed
        if not skip_embedding:
            vec_store.add_document(id, parsed["text"], doc.title)
        
        return doc

        
    def get_all(self) -> List[Document]:
        return list(self.documents.values())

# Global store instance
store = DocumentStore()

def load_initial_data(data_dir: str = "data"):
    if not os.path.exists(data_dir):
        return
        
    pkl_path = os.path.join(data_dir, "vector_store.pkl")
    loaded_from_disk = vec_store.load(pkl_path)
    
    new_docs_added = False
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".html"):
            doc_id = filename.replace(".html", "")
            
            # If vectors are already loaded for this doc_id, skip embedding generation
            skip_embedding = loaded_from_disk and (doc_id in vec_store.doc_embeddings)
            
            if not skip_embedding:
                new_docs_added = True
                
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                store.add_document(doc_id, f.read(), skip_embedding=skip_embedding)
                
    if new_docs_added:
        vec_store.save(pkl_path)
