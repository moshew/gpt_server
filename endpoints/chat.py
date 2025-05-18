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

from fastapi import Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import Chat, Message, File

from app_init import app
from auth import get_current_user
from db_manager import get_db, get_new_db_session, safe_close_session
from database import SessionLocal
from utils import run_in_executor, call_llm

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
    user = Depends(get_current_user)
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
    user = Depends(get_current_user)
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
    try:
        async with SessionLocal() as db:
        # Check chat ownership
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
        
        # Separate files by type
        doc_files = [file for file in files if file.file_type == "doc"]
        code_files = [file for file in files if file.file_type == "code"]

        def format_message_content(context):
            try:
                return json.loads(context)
            except json.JSONDecodeError:
                pass
            return context
        
        return {
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
            "docs": [
                {
                    "id": file.id,
                    "file_name": file.file_name
                }
                for file in doc_files
            ],
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
    except Exception as e:
        print(f"Error retrieving chat data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error while fetching chat data: {str(e)}"
        )

# Create new chat with default name endpoint
@app.post("/new_chat/")
async def new_chat(
    user = Depends(get_current_user)
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
    user = Depends(get_current_user)
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
    user = Depends(get_current_user)
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
