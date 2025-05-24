"""
RAG-Specific Utilities

This module contains utility functions specifically designed for RAG (Retrieval-Augmented Generation)
systems, focusing on content chunking and specialized RAG operations.
"""

from typing import List

from .async_helpers import run_in_executor

async def chunk_content_async(content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split content into overlapping chunks in a non-blocking way
    
    Args:
        content: Text content to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    return await run_in_executor(
        lambda: chunk_content(content, chunk_size, overlap)
    )

def chunk_content(content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split content into overlapping chunks
    
    Args:
        content: Text content to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(content):
        end = min(start + chunk_size, len(content))
        
        # Try to find a natural breaking point (paragraph)
        if end < len(content):
            # Look for paragraph breaks
            paragraph_break = content.rfind("\n\n", start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            else:
                # Look for newlines
                newline = content.rfind("\n", start, end)
                if newline != -1 and newline > start + chunk_size // 2:
                    end = newline + 1
                else:
                    # Look for sentence breaks
                    sentence_break = content.rfind(". ", start, end)
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2
        
        # Add the chunk
        chunks.append(content[start:end])
        
        # Move start position with overlap
        start = max(start, end - overlap)
        
        # If we didn't make progress, force an increment
        if start >= len(content) - 1:
            break
    
    return chunks