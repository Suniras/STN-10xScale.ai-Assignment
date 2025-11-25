def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Simple text chunking function
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    # For this assignment, the documents are small, so we can return as-is
    # But here's a simple chunking implementation if needed
    
    if len(text) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for period, exclamation, or question mark
            for i in range(end, start + overlap, -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks
