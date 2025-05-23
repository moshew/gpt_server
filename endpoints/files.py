"""
Files Handler for both Document and Code files

This module:
1. Provides a unified interface for uploading both document and code files
2. Leverages both the DocumentRAG and CodeRAG systems
3. Handles file format detection and routing
4. Implements a shared file registry with type tracking
5. Supports archive extraction for both systems
"""

import os
import asyncio
import json
import shutil
from typing import Dict, List, Any, Optional, Tuple, Set
import mimetypes
from datetime import datetime
import aiofiles.os

from pydantic import BaseModel
from fastapi import Depends, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import User, File as DBFile, SessionLocal

from app_init import app
from auth import verify_chat_owner
from rag_documents import get_document_rag
from utils.async_helpers import run_in_executor

# Semaphore to limit concurrent extraction processes
extraction_semaphore = asyncio.Semaphore(4)

@app.post("/upload_files/{chat_id}")
async def upload_files(
    chat_id: int,
    files: List[UploadFile] = File(...),
    file_type: str = Form("doc"),  # "doc" or "code"
    _: User = Depends(verify_chat_owner())
):
    """
    File upload endpoint for both document and code files
    
    Args:
        chat_id: Chat ID
        files: List of uploaded files
        file_type: Type of files - "doc" or "code"
        
    Returns:
        Upload result information including a list of all files in the chat
    """
    async with SessionLocal() as db:
        try:
            # Get handlers
            doc_rag = get_document_rag(str(chat_id))
            
            # Determine the destination folder based on file_type
            if file_type == "doc":
                destination_folder = os.path.join("chats", f"chat_{chat_id}")
            else:  # file_type == "code"
                destination_folder = os.path.join("code", f"chat_{chat_id}")

            # Ensure chat folder exists
            os.makedirs(destination_folder, exist_ok=True)

            # Track upload results
            results = {
                "uploaded": [], 
                "extracted": [],
                "errors": []
            }
            
            # Keep track of all archive processing tasks
            archive_tasks = []
            
            # Process files sequentially to avoid race conditions in file saving
            for file in files:
                try:
                    # Read file content
                    content = await file.read()
                    
                    # Save the file to the chat folder
                    file_path = os.path.join(destination_folder, file.filename)
                    
                    # Save the file to disk using run_in_executor
                    await run_in_executor(lambda: open(file_path, "wb").write(content))

                    # Check for archive files first by extension
                    file_ext = os.path.splitext(file.filename)[1].lower()
                    archive_extensions = ['.zip', '.tar', '.tar.gz', '.tgz', '.gz', '.rar']
                    if file_ext in archive_extensions:
                        # Create an archive processing task but don't start it yet
                        task = process_archive(
                            chat_id,
                            file_path, 
                            file.filename, 
                            destination_folder, 
                            results,
                            file_type
                        )
                        # Add to our list of tasks to wait for
                        archive_tasks.append(task)
                    else:
                        # For non-archive files (code or documents), add to database with the file_type
                        db_file = DBFile(
                            chat_id=chat_id,
                            file_type=file_type,
                            file_name=file.filename
                        )
                        db.add(db_file)
                        
                        # Update results
                        results["uploaded"].append(file.filename)
                except Exception as e:
                    results["errors"].append(f"Error processing {file.filename}: {str(e)}")
                    continue
            
            # Commit database changes for non-archive files
            await db.commit()
            
            # Now start all archive processing tasks in parallel
            if archive_tasks:
                # Wait for all archive extraction tasks to complete
                await asyncio.gather(*archive_tasks)
                
            # Query the database to get ALL files associated with this chat
            result = await db.execute(select(DBFile).filter(DBFile.chat_id == chat_id))
            all_files = result.scalars().all()
            
            # Split files by type
            doc_files = [{"id": file.id, "file_name": file.file_name} for file in all_files if file.file_type == "doc"]
            code_files = [{"id": file.id, "file_name": file.file_name} for file in all_files if file.file_type == "code"]
            
            return {
                "message": f"Uploaded {len(results['uploaded'])} file(s) and extracted {len(results['extracted'])} file(s)",
                "results": results,
                "doc_files": doc_files,
                "code_files": code_files
            }
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading files: {str(e)}"
            )

async def process_archive(
    chat_id: int,
    archive_path: str, 
    archive_filename: str, 
    destination_folder: str, 
    results: Dict[str, Any],
    file_type: str = "doc",
):
    """
    Process an archive file with semaphore control
    
    Args:
        chat_id: Chat ID
        archive_path: Path to the archive file
        archive_filename: Original filename of the archive
        destination_folder: Where to extract the files
        results: Results dictionary to update
        file_type: Type of files - "doc" or "code"
    """
    # Acquire semaphore to limit concurrent extractions
    async with extraction_semaphore:
        try:
            # Extract the archive
            extract_result = await extract_archive(archive_path, destination_folder)
            
            if extract_result["success"]:
                # Create a new session for this background task
                async with SessionLocal() as db:
                    try:
                        # Process each extracted file in batches to avoid overwhelming the DB
                        extracted_files = extract_result["extracted_files"]
                        batch_size = 50  # Reduced batch size for better performance
                        
                        for i in range(0, len(extracted_files), batch_size):
                            # Get the current batch of files
                            batch = extracted_files[i:i+batch_size]
                            batch_results = []
                            
                            # Process batch files asynchronously
                            for extracted_file in batch:
                                file_path = os.path.join(destination_folder, extracted_file)
                                
                                # Use async file system checks
                                file_exists = await aiofiles.ospath.isfile(file_path)
                                if not file_exists:
                                    continue
                                
                                try:
                                    # Add file to database with file_type
                                    db_file = DBFile(
                                        chat_id=chat_id,
                                        file_type=file_type,
                                        file_name=extracted_file
                                    )
                                    async_sessdbion.add(db_file)
                                    batch_results.append(extracted_file)
                                    
                                except Exception as e:
                                    results["errors"].append(f"Error processing extracted file {extracted_file}: {str(e)}")
                            
                            # Commit each batch to avoid large transactions
                            try:
                                await db.commit()
                                # Only add to results after successful commit
                                results["extracted"].extend(batch_results)
                            except Exception as e:
                                await db.rollback()
                                results["errors"].append(f"Database commit error for batch: {str(e)}")
                                break
                            
                            # Allow other coroutines to run between batches
                            await asyncio.sleep(0.01)
                    
                    except Exception as e:
                        await db.rollback()
                        results["errors"].append(f"Database error during extraction: {str(e)}")
                # Async file deletion
                await aiofiles.os.remove(archive_path)
            else:
                results["errors"].append(f"Failed to extract {archive_filename}: {extract_result.get('error', 'Unknown error')}")
        except Exception as e:
            results["errors"].append(f"Error processing archive {archive_filename}: {str(e)}")

async def extract_archive(archive_path: str, destination_folder: str) -> Dict[str, Any]:
    """
    Extract zip or tar archive to the destination folder, preserving subdirectories
    
    Args:
        archive_path: Path to the archive file
        destination_folder: Folder to extract to
        
    Returns:
        Dict with extraction results
    """
    file_ext = os.path.splitext(archive_path)[1].lower()
    
    # Get a reference to the event loop
    loop = asyncio.get_event_loop()
    
    try:
        if file_ext == '.zip':
            # Define a function to extract zip files
            def extract_zip():
                import zipfile
                extracted_files = []
                
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # Get list of files in the archive
                    file_list = zip_ref.namelist()
                    
                    # Create all needed directories first
                    for item in file_list:
                        item_path = os.path.join(destination_folder, item)
                        # Create directories if needed
                        dirname = os.path.dirname(item_path)
                        if dirname and not os.path.exists(dirname):
                            os.makedirs(dirname, exist_ok=True)
                    
                    # Extract all files
                    zip_ref.extractall(destination_folder)
                    
                    # Return only the file names (including subdirectory paths), not directory entries
                    extracted_files = [f for f in file_list if not f.endswith('/') and not os.path.isdir(os.path.join(destination_folder, f))]
                    return extracted_files
            
            # Run in dedicated executor and await completion
            extracted_files = await loop.run_in_executor(None, extract_zip)
            
        elif file_ext in ['.tar', '.tar.gz', '.tgz', '.gz']:
            # Define a function to extract tar files
            def extract_tar():
                import tarfile
                extracted = []
                
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    # Create all needed directories first
                    for member in tar_ref.getmembers():
                        if member.isdir():
                            directory_path = os.path.join(destination_folder, member.name)
                            os.makedirs(directory_path, exist_ok=True)
                    
                    # Extract all files
                    tar_ref.extractall(destination_folder)
                    
                    # Return file paths including subdirectories
                    for member in tar_ref.getmembers():
                        if member.isfile():
                            extracted.append(member.name)
                            
                return extracted
            
            # Run in dedicated executor and await completion
            extracted_files = await loop.run_in_executor(None, extract_tar)
            
        elif file_ext == '.rar':
            # Define a function to extract rar files
            def extract_rar():
                try:
                    import rarfile
                    extracted = []
                    
                    with rarfile.RarFile(archive_path) as rar_ref:
                        # Get list of files
                        file_list = rar_ref.namelist()
                        
                        # Create all needed directories first
                        for item in file_list:
                            item_path = os.path.join(destination_folder, item)
                            # Create directories if needed
                            dirname = os.path.dirname(item_path)
                            if dirname and not os.path.exists(dirname):
                                os.makedirs(dirname, exist_ok=True)
                        
                        # Extract all files
                        rar_ref.extractall(destination_folder)
                        
                        # Return only files, not directories
                        extracted = [f for f in file_list if not f.endswith('/') and not os.path.isdir(os.path.join(destination_folder, f))]
                        return extracted
                except ImportError:
                    raise Exception("RAR extraction requires the 'rarfile' package. Please install it with 'pip install rarfile'")
            
            # Run in dedicated executor and await completion
            extracted_files = await loop.run_in_executor(None, extract_rar)
        else:
            return {
                "success": False,
                "error": f"Unsupported archive format: {file_ext}",
                "extracted_files": []
            }
        
        return {
            "success": True,
            "extracted_files": extracted_files
        }
        
    except Exception as e:
        print(f"Error extracting archive {archive_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "extracted_files": []
        }


@app.post("/index_files/{chat_id}")
async def index_files(
    chat_id: int,
    file_type: str = Form("doc"),  # "doc" or "code"
    _: User = Depends(verify_chat_owner())
):
    """
    Index files for a specific chat by type
    
    Args:
        chat_id: Chat ID
        file_type: "doc" or "code"
        
    Returns:
        Indexing results (only after all operations are fully complete)
    """
    results = {}
    print(f"Starting indexing for chat {chat_id}, file_type: {file_type}")
    
    # Index documents if requested
    if file_type == "doc":
        try:
            print(f"Initializing document RAG for chat {chat_id}")
            doc_rag = get_document_rag(str(chat_id))
            
            print(f"Starting document indexing for chat {chat_id}")
            # WAIT for complete indexing before returning
            doc_results = await doc_rag.index_documents(str(chat_id))
            print(f"Document indexing completed for chat {chat_id}: {doc_results}")
            
            results["documents"] = doc_results
        except Exception as e:
            error_msg = f"Error indexing documents: {str(e)}"
            print(f"Indexing error for chat {chat_id}: {error_msg}")
            results["documents"] = {"error": error_msg}
    
    print(f"All indexing operations completed for chat {chat_id}")
    return {
        "message": "Indexing complete",
        "results": results
    }