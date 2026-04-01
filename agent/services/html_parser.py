from bs4 import BeautifulSoup

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
