"""
Code query processing module

This module:
1. Handles code-specific queries
2. Collects code files from the code directory
3. Prioritizes code files over configuration files
4. Formats and returns code context for LLM processing
"""

import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Define code file extensions with priority (higher priority = more important)
FILE_PRIORITIES = {
    # High priority - Main programming languages
    '.py': 10,
    '.js': 10,
    '.jsx': 10,
    '.ts': 10,
    '.tsx': 10,
    '.java': 10,
    '.cpp': 10,
    '.c': 10,
    '.h': 10,
    '.hpp': 10,
    '.cs': 10,
    '.php': 10,
    '.rb': 10,
    '.go': 10,
    '.rs': 10,
    '.swift': 10,
    '.kt': 10,
    '.scala': 10,
    
    # Medium priority - Less common languages and special files
    '.sh': 8,
    '.bash': 8,
    '.sql': 8,
    '.r': 8,
    '.m': 8,
    '.dart': 8,
    '.vue': 8,
    '.svelte': 8,
    '.elm': 8,
    '.clj': 8,
    '.lua': 8,
    '.pl': 8,
    '.pm': 8,
    '.ex': 8,
    '.exs': 8,
    '.hs': 8,
    '.proto': 8,
    '.graphql': 8,
    '.prisma': 8,
    
    # Lower priority - Frontend and styling
    '.html': 6,
    '.css': 6,
    '.scss': 6,
    '.sass': 6,
    '.less': 6,
    
    # Lowest priority - Configuration files
    '.json': 4,
    '.xml': 4,
    '.yml': 4,
    '.yaml': 4,
    '.toml': 4,
    '.conf': 4,
    '.ini': 4,
    '.properties': 4,
    '.cfg': 4,
    '.env': 4,
    '.gradle': 4,
    '.cmake': 4,
    
    # Documentation
    '.md': 3,
    '.txt': 3,
    
    # Build and ignore files
    '.dockerfile': 2,
    '.dockerignore': 2,
    '.gitignore': 2,
}

# Special files without extensions
SPECIAL_FILES = {
    'makefile': 8,
    'dockerfile': 2,
    'jenkinsfile': 8,
    'rakefile': 8
}

MAX_FILES = 35  # Limit to avoid token limits


def get_file_priority(file_name: str) -> Optional[int]:
    """
    Get the priority of a file based on its extension or name
    
    Args:
        file_name: Name of the file
        
    Returns:
        Priority value or None if not a recognized code file
    """
    file_ext = os.path.splitext(file_name)[1].lower()
    base_name = file_name.lower()
    
    # Check extension priority
    if file_ext in FILE_PRIORITIES:
        return FILE_PRIORITIES[file_ext]
    
    # Check special files without extensions
    if not file_ext and base_name in SPECIAL_FILES:
        return SPECIAL_FILES[base_name]
    
    return None


def is_text_file(file_path: str) -> bool:
    """
    Check if a file is a text file (not binary)
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if text file, False if binary or unreadable
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to read the first line to check if it's text
            f.readline()
        return True
    except (UnicodeDecodeError, IOError):
        return False


def collect_code_files(code_dir: str) -> List[Tuple[int, str, str]]:
    """
    Collect all code files from a directory and its subdirectories
    
    Args:
        code_dir: Base directory to search for code files
        
    Returns:
        List of tuples (priority, relative_path, absolute_path)
    """
    code_files = []
    
    if not os.path.exists(code_dir):
        logger.info(f"Code directory does not exist: {code_dir}")
        return code_files
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(code_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            # Get priority for the file
            priority = get_file_priority(file_name)
            if priority is None:
                continue  # Skip files not in our priority lists
            
            # Check if it's not a binary file
            if not is_text_file(file_path):
                logger.info(f"Skipping binary or unreadable file: {file_path}")
                continue
            
            # Add to the list
            relative_path = os.path.relpath(file_path, code_dir)
            code_files.append((priority, relative_path, file_path))
    
    # Sort files by priority (descending) and then by path (for stable ordering)
    code_files.sort(key=lambda x: (-x[0], x[1]))
    
    # Limit to MAX_FILES
    if len(code_files) > MAX_FILES:
        logger.warning(f"Found {len(code_files)} code files, limiting to {MAX_FILES} highest priority files")
        code_files = code_files[:MAX_FILES]
    
    return code_files


def format_code_context(code_files: List[Tuple[int, str, str]]) -> str:
    """
    Format code files into a context string for the LLM
    
    Args:
        code_files: List of tuples (priority, relative_path, absolute_path)
        
    Returns:
        Formatted code context string
    """
    if not code_files:
        return ""
    
    code_context = "\n\n=== Code Files for Context ===\n"
    
    for priority, relative_path, file_path in code_files:
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add file content to context
            code_context += f"\n--- File: {relative_path} ---\n"
            code_context += content
            code_context += "\n--- End of {relative_path} ---\n"
            
        except Exception as e:
            logger.error(f"Error reading code file {relative_path}: {e}")
            code_context += f"\n--- File: {relative_path} (Error reading file: {e}) ---\n"
    
    code_context += "\n=== End of Code Files ===\n"
    return code_context


def get_code_context(chat_id: int) -> str:
    """
    Get the code context for a specific chat
    
    Args:
        chat_id: Chat identifier
        
    Returns:
        Formatted code context string
    """
    code_dir = os.path.join("code", f"chat_{chat_id}")
    
    try:
        code_files = collect_code_files(code_dir)
        
        if code_files:
            logger.info(f"Added {len(code_files)} code files to context for chat {chat_id}")
            return format_code_context(code_files)
        else:
            logger.info(f"No code files found for chat {chat_id}")
            return ""
            
    except Exception as e:
        logger.error(f"Error gathering code files for chat {chat_id}: {e}")
        return ""