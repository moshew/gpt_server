"""
Chat management endpoints module

This module:
1. Handles chat creation and management
2. Provides endpoints for retrieving chat history
3. Focuses exclusively on chat operations (not query processing)
"""

import asyncio
import json
import datetime
from typing import Optional
from pydantic import BaseModel

from fastapi import  HTTPException, Request
from sqlalchemy.future import select
from ..database import Chat, Message, File

from ..app_init import app
from ..auth import get_current_user
from ..database import SessionLocal
from ..utils import run_in_executor, call_llm
from ..utils.llm_helpers_azure import get_rate_limit_status

# Request deduplication cache
_active_chat_requests = {}

# Class for chat creation request
class ChatRequest(BaseModel):
    message: str

# Class for chat name update request
class ChatNameRequest(BaseModel):
    chat_id: int
    message: str

# Function to generate chat title from message
async def generate_chat_title(message: str) -> str:
    """
    Generate a chat title based on a user message
    
    Args:
        message: The user message to base the title on
        
    Returns:
        Generated chat title
    """
    try:
        chat_title_prompt = f"Generate a short chat title (3-6 words) IN ENGLISH based on the user's message: {message}"
        # Use the generic LLM helper with a precise configuration
        chat_title = await call_llm(
            prompt=chat_title_prompt,
            model_config="precise"
        )
        # Clean up the title
        return chat_title.replace('"', '')
        
    except Exception as e:
        print(f"Error generating chat title: {e}")
        return ""

# Get user chats endpoint
@app.get("/chats/")
async def get_user_chats(
    request: Request,
):
    """
    Get the 10 most recent chats for a user
    
    Args:
        user: User object from authentication dependency
        
    Returns:
        List of user's 10 most recent chats
    """
    try:
        # Query with limit to get only the 10 most recent chats
        async with SessionLocal() as db:
            user = await get_current_user(request, db)
            result = await db.execute(
                select(Chat)
                .filter(Chat.user_id == user.id)
                .order_by(Chat.created_at.desc())
                .limit(10)
            )
            chats = result.scalars().all()
        
        return {
            "chats": [
                {"id": chat.id, "name": chat.chat_name, "created_at": chat.created_at}
                for chat in chats
            ],
            "count": len(chats)
        }
    except Exception as e:
        print(f"Error getting user chats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error while fetching chats: {str(e)}"
        )

# Get chat data endpoint
@app.get("/chat_data/{chat_id}")
async def get_chat_data(
    chat_id: int,
    request: Request,
):
    """
    Get all messages and files for a specific chat, separating document and code files,
    ensuring the chat belongs to the requesting user
    
    Args:
        chat_id: ID of the chat to get data from
        user: User object from authentication dependency
        
    Returns:
        Dict containing chat information, messages, document files, and code files
    """
    import time
    import logging
    
    logger = logging.getLogger("chat_data")
    start_time = time.time()
    
    # Request deduplication - check if same request is already in progress
    request_key = f"chat_data_{chat_id}"
    
    if request_key in _active_chat_requests:
        try:
            # Wait for the existing request to complete (max 15 seconds)
            result = await asyncio.wait_for(_active_chat_requests[request_key], timeout=15.0)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Duplicate request for chat {chat_id} timed out, proceeding with new request")
            _active_chat_requests.pop(request_key, None)
        except Exception as e:
            logger.error(f"Error waiting for existing request for chat {chat_id}: {e}")
            _active_chat_requests.pop(request_key, None)
    
    # Create a new task for this request
    async def execute_request():
        try:
            # Add timeout for database operations
            timeout_duration = 30  # 30 seconds timeout
            
            async def db_operations():                
                async with SessionLocal() as db:
                    # Check chat ownership
                    user = await get_current_user(request, db)
                    
                    result = await db.execute(
                        select(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id)
                    )
                    chat = result.scalars().first()
                
                    if not chat:
                        raise HTTPException(status_code=404, detail="Chat not found or access denied")
                
                    # Query all messages for this chat ordered by timestamp
                    result = await db.execute(
                        select(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp)
                    )
                    messages = result.scalars().all()
                
                    # Query all files for this chat
                    result = await db.execute(
                        select(File).filter(File.chat_id == chat_id)
                    )
                    files = result.scalars().all()
                    
                    return chat, messages, files
            
            # Use asyncio.wait_for to add timeout
            try:
                chat, messages, files = await asyncio.wait_for(db_operations(), timeout=timeout_duration)
                    
            except asyncio.TimeoutError:
                logger.error(f"Database operations timed out after {timeout_duration}s for chat {chat_id}")
                raise HTTPException(status_code=504, detail=f"Request timed out after {timeout_duration} seconds")
            
            # Separate files by type
            doc_files = [file for file in files if file.file_type == "doc"]
            code_files = [file for file in files if file.file_type == "code"]

            def format_message_content(context):
                try:
                    return json.loads(context)
                except json.JSONDecodeError:
                    pass
                return context
            
            response = {
                "chat_id": chat_id,
                "chat_name": chat.chat_name,
                "messages": [
                    {
                        "id": message.id,
                        "sender": message.sender,
                        "content": format_message_content(message.content),
                        "timestamp": message.timestamp
                    }
                    for message in messages
                ],
                "docs": {
                    "keep_original_files": chat.keep_original_files,
                    "files": [
                        {
                            "id": file.id,
                            "file_name": file.file_name
                        }
                        for file in doc_files
                    ]
                },
                "code": [
                    {
                        "id": file.id,
                        "file_name": file.file_name
                    }
                    for file in code_files
                ],
                "message_count": len(messages),
                "doc_count": len(doc_files),
                "code_count": len(code_files)
            }
            
            total_time = time.time() - start_time
            if total_time > 1.0:  # Only log slow requests
                logger.info(f"Chat_data request for chat {chat_id} took {total_time:.2f}s")
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error retrieving chat data for chat {chat_id} after {total_time:.2f}s: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error while fetching chat data: {str(e)}"
            )
    
    # Create and store the task
    task = asyncio.create_task(execute_request())
    _active_chat_requests[request_key] = task
    
    try:
        result = await task
        return result
    finally:
        # Clean up the request from cache
        _active_chat_requests.pop(request_key, None)

# Create new chat with default name endpoint
@app.post("/new_chat/")
async def new_chat(
    request: Request,
):
    """
    Create a new chat with a default name (fast creation without waiting for name generation)
    
    Args:
        user: User object from authentication dependency
        
    Returns:
        Chat information with default name
    """
    async with SessionLocal() as db:
        try:
            # Create default chat name using current timestamp
            default_chat_name = f"New Chat"
        
            # Create new chat in database
            user = await get_current_user(request, db)
            chat = Chat(user_id=user.id, chat_name=default_chat_name)
            db.add(chat)
            await db.commit()
            await db.refresh(chat)
            
            return {
                "id": chat.id, 
                "name": chat.chat_name
            }
            
        except Exception as e:
            await db.rollback()
            print(f"Error creating new chat: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating new chat: {str(e)}")

# Update chat name based on message
@app.post("/update_chat_name/")
async def update_chat_name(
    chat_name_request: ChatNameRequest,
    request: Request,
):
    """
    Update chat name based on a provided message
    
    Args:
        chat_name_request: Request with chat ID and message for generating name
        user: User object from authentication dependency
        
    Returns:
        Updated chat information with the generated name
    """
    async with SessionLocal() as db:
        try:
            # Get the chat ID from the request
            chat_id = chat_name_request.chat_id

            user = await get_current_user(request, db)
            result = await db.execute(
                select(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id)
            )
            chat = result.scalars().first()

            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found or access denied")
            
            # Generate chat name based on the message
            chat_name = await generate_chat_title(chat_name_request.message)
            
            # If generation failed or returned empty, use a fallback name
            if not chat_name:
                chat_name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Update chat name in database
            chat.chat_name = chat_name
            await db.commit()
            await db.refresh(chat)
            
            return {
                "id": chat.id, 
                "name": chat.chat_name,
                "status": "updated"
            }
            
        except Exception as e:
            await db.rollback()
            print(f"Error updating chat name: {e}")
            raise HTTPException(status_code=500, detail=f"Error updating chat name: {str(e)}")

# Create new chat with automatically generated name based on first message
@app.post("/new_named_chat/")
async def new_named_chat(
    chat_request: ChatRequest, 
    request: Request,
):
    """
    Create a new chat with the first question, generating a chat name synchronously
    
    Args:
        chat_request: Request with the first message
        user: User object from authentication dependency
        
    Returns:
        Chat information with the generated name
    """
    async with SessionLocal() as db:
        try:
            # First create the chat with a temporary name
            temp_chat_name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create new chat in database
            user = await get_current_user(request, db)
            chat = Chat(user_id=user.id, chat_name=temp_chat_name)
            db.add(chat)
            await db.commit()
            await db.refresh(chat)
            
            # Generate chat name based on the message
            chat_name = await generate_chat_title(chat_request.message)
            
            # If generation failed or returned empty, keep the temporary name
            if not chat_name:
                chat_name = temp_chat_name
            
            # Update chat with the generated name
            chat.chat_name = chat_name
            await db.commit()
            await db.refresh(chat)
            
            return {
                "id": chat.id, 
                "name": chat.chat_name,
                "status": "created"
            }
            
        except Exception as e:
            await db.rollback()
            print(f"Error creating new chat: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating new chat: {str(e)}")

# Debug endpoint for rate limiting status
@app.get("/debug/rate_limits")
async def get_rate_limits():
    """
    Debug endpoint to check rate limiting status
    
    Returns:
        Current rate limiting information
    """
    try:
        status = get_rate_limit_status()
        return {
            "rate_limiting": status,
            "active_chat_requests": len(_active_chat_requests),
            "active_requests": list(_active_chat_requests.keys())
        }
    except Exception as e:
        return {"error": f"Could not get rate limit status: {e}"}
