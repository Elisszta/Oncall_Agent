from bs4 import BeautifulSoup
from typing import List

def parse_html(html_content: str) -> dict:
    soup = BeautifulSoup(html_content, "html.parser")
    # Extract title
    title_tag = soup.find(["h1", "title"])
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"
    
    # Remove script and style tags
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
        
    text = soup.get_text(separator=" ", strip=True)
    return {
        "title": title,
        "text": text
    }

def get_logical_chunks(html_content: str, max_chunk_size: int = 1200) -> List[str]:
    """
    Extract text segments from HTML by respecting structural boundaries (headers, paragraphs, lists).
    Improved version with better boilerplate removal and boundary detection.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 1. Boilerplate Removal (Deep Cleaning)
    # We remove things that are usually not part of the core SOP content
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
        for matching_tag in soup.find_all(tag.name):
            matching_tag.decompose()

    # 2. Extract block-level content
    # We'll use a more comprehensive list of block elements
    block_elements = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "section", "article", "pre", "blockquote", "table"]
    
    chunks = []
    current_chunk = []
    current_size = 0

    # find all block elements that are NOT nested within other block elements we're tracking
    def is_top_block(el):
        if el.name not in block_elements:
            return False
        # Check if any parent is also a block element
        for p in el.parents:
            if p.name in block_elements:
                return False
        return True

    top_blocks = [el for el in soup.find_all(recursive=True) if is_top_block(el)]
    
    for element in top_blocks:
        text = element.get_text(separator=" ", strip=True)
        if not text or len(text) < 2:
            continue
            
        is_header = element.name.startswith('h')
        
        # New chunk if:
        # 1. It's a header (H1, H2, H3) and we already have content - start fresh for new sections
        # 2. Or adding this element exceeds our max size
        if (is_header and current_chunk and element.name in ['h1', 'h2', 'h3']) or \
           (current_size + len(text) > max_chunk_size and current_chunk):
            chunks.append("\n".join(current_chunk))
            current_chunk = [text]
            current_size = len(text)
        else:
            current_chunk.append(text)
            current_size += len(text)
            
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    # 3. Handle documents with no standard block structure (fallback)
    if not chunks:
        # Just take all text and split it
        raw_text = soup.get_text(separator="\n", strip=True)
        if raw_text:
            # Simple recursive character split as absolute fallback
            import re
            return [t.strip() for t in re.split(r'\n+', raw_text) if len(t.strip()) > 20]
            
    return chunks
