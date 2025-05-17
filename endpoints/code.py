"""
Code Management and Analysis Module

This module integrates code file management and analysis capabilities by combining:
1. Code file management endpoints (uploading, listing, indexing)
2. Code analysis endpoints (documentation, structure analysis)
3. Code file operations (downloading, internal documentation)

The module provides a comprehensive interface for code-related functionality
with async support and optimized performance handling.
"""

import os
import time
import tempfile
import asyncio
from typing import Dict, Any, List

import aiofiles
from aiofiles.os import path as aiopath

from fastapi import Depends, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app_init import app
from database import Chat, Message, User
from auth import verify_chat_owner
from db_manager import get_db, save_message

# =================================
# Code Documentation and Analysis
# =================================

@app.get("/document_code/{chat_id}")
async def document_code(
    chat_id: int, 
    file_name: str,
    doc_style: str = "standard",
    db: AsyncSession = Depends(get_db),
    _: User = Depends(verify_chat_owner())
):
    """
    Generate documentation for a specific code file
    
    Args:
        chat_id: Chat ID
        file_name: Name of the file to document
        doc_style: Documentation style ("standard", "detailed", "minimal")
        db: Database session
        
    Returns:
        Generated documentation
    """
    # Measure performance
    start_time = time.time()
    
    # Check if chat exists
    chat_folder = os.path.join("chats", f"chat_{chat_id}")
    if not await aiopath.exists(chat_folder):
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get code_rag instance
    code_rag = app.state.code_analyzer._get_code_rag(str(chat_id))
    
    # If file is not indexed, try to index it first
    if file_name not in code_rag.metadata["files"]:
        file_path = os.path.join(chat_folder, file_name)
        if not await aiopath.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {file_name} not found")
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Index the file
            await code_rag.index_code_file(file_path, content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error indexing file: {str(e)}")
    
    # Process the request using regular documentation
    result = await code_rag.document_file(file_name, doc_style)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Save the documentation as a system message
    if "documentation" in result:
        system_message = f"Documentation generated for {file_name}:\n\n{result['documentation']}"
        
        # Run in background task to avoid blocking
        task = asyncio.create_task(save_message(db, chat_id, "assistant", system_message))
        app.state.task_manager.add_task(chat_id, task)

        elapsed_time = time.time() - start_time
        print(f"Documentation generated in {elapsed_time:.2f} seconds")
        
        return JSONResponse(content=result)
    
    return result

@app.get("/internal_documentation/{chat_id}")
async def internal_documentation(
    chat_id: int, 
    file_name: str,
    doc_style: str = "standard",
    _: User = Depends(verify_chat_owner())
):
    """
    Generate internal documentation (docstrings/comments) for a specific code file
    
    Args:
        chat_id: Chat ID
        file_name: Name of the file to document
        doc_style: Documentation style ("standard", "detailed", "minimal")
        
    Returns:
        Internal documentation for the file
    """
    # Measure performance
    start_time = time.time()
    
    # Check if chat exists
    chat_folder = os.path.join("chats", f"chat_{chat_id}")
    if not await aiopath.exists(chat_folder):
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get code_rag instance
    code_rag = app.state.code_analyzer._get_code_rag(str(chat_id))
    
    # If file is not indexed, try to index it first
    if file_name not in code_rag.metadata["files"]:
        file_path = os.path.join(chat_folder, file_name)
        if not await aiopath.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {file_name} not found")
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Index the file
            await code_rag.index_code_file(file_path, content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error indexing file: {str(e)}")
    
    # Process the request using internal documentation
    result = await code_rag.generate_internal_documentation(file_name, doc_style)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    elapsed_time = time.time() - start_time
    print(f"Internal documentation generated in {elapsed_time:.2f} seconds")
    
    # Return only the documented code without additional explanations
    return {
        "file_name": file_name,
        "language": result["language"],
        "documented_code": result["documented_code"]
    }

# =================================
# Advanced Code Analysis Endpoints
# =================================

@app.get("/analyze_code_structure/{chat_id}")
async def analyze_code_structure_endpoint(
    chat_id: int,
    _: User = Depends(verify_chat_owner())
):
    """
    Analyze the structure of code files in a chat
    
    Args:
        chat_id: Chat ID
        
    Returns:
        Code structure analysis
    """
    # Check if chat exists
    chat_folder = os.path.join("chats", f"chat_{chat_id}")
    if not await aiopath.exists(chat_folder):
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get the code_rag instance
    code_rag = app.state.code_analyzer._get_code_rag(str(chat_id))
    
    # Analyze code structure
    analysis = await code_rag.analyze_code_structure()
    
    if "error" in analysis:
        raise HTTPException(status_code=500, detail=analysis["error"])
    
    return analysis

@app.get("/analyze_documentation/{chat_id}")
async def analyze_documentation_endpoint(
    chat_id: int,
    file_name: str,
    _: User = Depends(verify_chat_owner())
):
    """
    Analyze documentation coverage for a specific file
    
    Args:
        chat_id: Chat ID
        file_name: Name of the file to analyze
        
    Returns:
        Documentation coverage analysis
    """
    # Get the code_rag instance from app state
    code_rag = app.state.code_analyzer._get_code_rag(str(chat_id))
    
    # Analyze documentation coverage
    result = await code_rag.analyze_documentation_coverage(file_name)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result

@app.post("/document_multiple_files/{chat_id}")
async def document_multiple_files_endpoint(
    chat_id: int,
    file_names: list[str],
    doc_style: str = "standard",
    _: User = Depends(verify_chat_owner())
):
    """
    Generate documentation for multiple files
    
    Args:
        chat_id: Chat ID
        file_names: List of files to document
        doc_style: Documentation style ("standard", "detailed", "minimal")
        
    Returns:
        Documentation for all files
    """
    # Get the code_rag instance from app state
    code_rag = app.state.code_analyzer._get_code_rag(str(chat_id))
    
    # Document multiple files
    start_time = time.time()
    result = await code_rag.document_multiple_files(file_names, doc_style)
    
    elapsed_time = time.time() - start_time
    print(f"Multiple files documented in {elapsed_time:.2f} seconds")
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

# =================================
# Status and Monitoring Endpoints
# =================================

@app.get("/code_processing_status/{chat_id}")
async def code_processing_status_endpoint(
    chat_id: int,
    _: User = Depends(verify_chat_owner())
):
    """
    Get the status of code processing for a chat
    
    Args:
        chat_id: Chat ID
        
    Returns:
        Processing status information
    """
    try:
        # Get code files information
        code_files_info = await app.state.code_analyzer._list_code_files(chat_id)
        
        # Check if there are any active indexing locks
        # To be implemented if needed
        
        return {
            "chat_id": chat_id,
            "code_files": code_files_info.get("files", []),
            "indexed_files": code_files_info.get("indexed_files", [])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking code processing status: {str(e)}"
        )