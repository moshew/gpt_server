"""
Text Processing Utilities

This module contains general-purpose text processing utilities that can be used
across the entire application, not just in RAG-specific contexts.
"""

import json
import re
from typing import Dict, List, Any, Optional
from functools import partial

from .async_helpers import run_in_executor

async def parse_json_string(json_string: str) -> Dict:
    """
    Parse a JSON string to a dictionary (CPU-bound operation)
    
    Args:
        json_string: JSON string to parse
        
    Returns:
        Parsed JSON as dictionary
    """
    return await run_in_executor(json.loads, json_string)

async def serialize_to_json(obj: Any, **kwargs) -> str:
    """
    Serialize an object to a JSON string (CPU-bound operation)
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    return await run_in_executor(
        partial(json.dumps, **kwargs),
        obj
    )

async def run_regex_search(pattern: str, text: str) -> Optional[re.Match]:
    """
    Run a regex search in a thread pool (CPU-bound operation)
    
    Args:
        pattern: Regex pattern
        text: Text to search
        
    Returns:
        Match object or None
    """
    return await run_in_executor(
        lambda: re.search(pattern, text)
    )

async def run_regex_findall(pattern: str, text: str) -> List[str]:
    """
    Run a regex findall in a thread pool (CPU-bound operation)
    
    Args:
        pattern: Regex pattern
        text: Text to search
        
    Returns:
        List of matches
    """
    return await run_in_executor(
        lambda: re.findall(pattern, text)
    )

async def extract_code_from_markdown(markdown_text: str) -> str:
    """
    Extract code from markdown text, removing language tags
    
    Args:
        markdown_text: Markdown text containing code blocks
        
    Returns:
        Clean code without markdown or language tags
    """
    # Run in thread pool as regex can be CPU-intensive
    return await run_in_executor(
        _extract_code_sync, markdown_text
    )

def _extract_code_sync(markdown_text: str) -> str:
    """
    Synchronous version of code extraction for thread pool
    
    Args:
        markdown_text: Markdown text containing code blocks
        
    Returns:
        Clean code without markdown or language tags
    """
    # If there are code blocks, extract them
    if "```" in markdown_text:
        # Find all code blocks
        code_block_pattern = r"```(?:[a-zA-Z]*\n)?([\s\S]*?)```"
        code_blocks = re.findall(code_block_pattern, markdown_text)
        
        if code_blocks:
            # Join code blocks with newlines
            return "\n\n".join(code_blocks)
    
    # If no code blocks found, return original text
    return markdown_text