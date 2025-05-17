"""
Query processing endpoints module

This module:
1. Handles processing of user queries
2. Manages active queries and their cancellation
3. Provides endpoints for both general and code-specific queries
4. Implements streaming responses for query results
5. Enhances queries with document context when available
6. Supports image generation and variation requests
7. Enhanced with Confluence knowledge base integration
8. Supports adding images to queries for multimodal LLM processing
"""
import uuid
import asyncio
import time
import os
import json
import logging
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from fastapi import Depends, BackgroundTasks, HTTPException, Query, File, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import User, Chat, Message
from langchain.schema import HumanMessage, SystemMessage
from contextlib import asynccontextmanager

from app_init import app
from auth import get_user_from_token, verify_chat_owner, verify_chat_ownership
from db_manager import get_db, get_new_db_session, safe_close_session, load_memory, save_message
from utils import process_langchain_messages
from utils.sse import stream_text_as_sse, stream_generator_as_sse
from rag_documents import get_document_rag, DocumentRAG
from image_service import get_image_service
from query_web import web_search_handler
from query_code import get_code_context

# Dictionary to track active queries, with chat_id as key and task/event objects as values
active_queries = {}
pending_queries = {}

# Define path for knowledge bases
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "knowledgebase")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("query")

# Helper function to register an active query
def register_active_query(chat_id, task=None, cancel_event=None):
    """
    Register an active query for potential cancellation
    
    Args:
        chat_id: Chat ID as a unique identifier
        task: Optional asyncio task object
        cancel_event: Optional cancellation event
    """
    global active_queries
    
    # Create a cancellation event if none was provided
    if cancel_event is None:
        cancel_event = asyncio.Event()
    
    active_queries[str(chat_id)] = {
        "task": task,
        "cancel_event": cancel_event,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    return cancel_event

# Helper function to check if a query should be cancelled
def should_cancel_query(chat_id):
    """
    Check if a query has been flagged for cancellation
    
    Args:
        chat_id: Chat ID to check
        
    Returns:
        True if the query should be cancelled, False otherwise
    """
    global active_queries
    
    chat_id_str = str(chat_id)
    if chat_id_str in active_queries and active_queries[chat_id_str].get("cancel_event"):
        return active_queries[chat_id_str]["cancel_event"].is_set()
    
    return False

# Helper function to clean up after a query completes or is cancelled
def unregister_active_query(chat_id):
    """
    Remove a query from the active queries dictionary
    
    Args:
        chat_id: Chat ID to unregister
    """
    global active_queries
    
    chat_id_str = str(chat_id)
    if chat_id_str in active_queries:
        del active_queries[chat_id_str]

# Function to clean up stale active queries
async def cleanup_stale_active_queries():
    """
    Periodically clean up stale active queries
    
    This function removes queries that have been active for too long,
    which could indicate they were never properly unregistered.
    """
    global active_queries
    
    try:
        current_time = asyncio.get_event_loop().time()
        stale_threshold = 3600  # 1 hour in seconds
        
        # Find queries that have been active for too long
        stale_queries = []
        for chat_id, query_info in active_queries.items():
            if current_time - query_info.get("timestamp", 0) > stale_threshold:
                stale_queries.append(chat_id)
        
        # Remove stale queries
        for chat_id in stale_queries:
            logger.info(f"Cleaning up stale query for chat {chat_id}")
            del active_queries[chat_id]
            
        logger.info(f"Cleaned up {len(stale_queries)} stale queries. {len(active_queries)} active queries remaining.")
    except Exception as e:
        logger.error(f"Error during active query cleanup: {e}")

# Schedule periodic cleanup at application startup
@app.on_event("startup")
async def schedule_active_query_cleanup():
    """Schedule periodic cleanup of stale active queries"""
    async def cleanup_task():
        while True:
            await cleanup_stale_active_queries()
            # Run every 30 minutes
            await asyncio.sleep(1800)
    
    # Start the cleanup task in the background
    asyncio.create_task(cleanup_task())

# Endpoint to stop an active query
@app.post("/stop_query/{chat_id}")
async def stop_query(
    chat_id: int,
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Stop an active query for a specific chat
    
    Args:
        chat_id: ID of the chat with the query to stop
        token
        db: Database session
        
    Returns:
        Status message indicating whether the query was stopped
    """
    user = await get_user_from_token(token, db)
    await verify_chat_ownership(chat_id, user.id, db)
    
    try:
        # Check if there's an active query for this chat
        chat_id_str = str(chat_id)
        if chat_id_str not in active_queries:
            return {"message": "No active query found for this chat"}
        
        # Set the cancellation event
        if "cancel_event" in active_queries[chat_id_str]:
            active_queries[chat_id_str]["cancel_event"].set()
            
            # If there's a task, try to cancel it
            if "task" in active_queries[chat_id_str] and active_queries[chat_id_str]["task"]:
                try:
                    active_queries[chat_id_str]["task"].cancel()
                except Exception as e:
                    logger.error(f"Error cancelling task: {e}")
            
            return {"message": "Query stop signal sent successfully"}
        else:
            return {"message": "Query found but no cancellation mechanism available"}
            
    except Exception as e:
        logger.error(f"Error stopping query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error stopping query: {str(e)}"
        )

# --- Shared helper functions for queries ---

async def setup_query(
    chat_id: int,
    query: str,
    token: str,
    db: AsyncSession
) -> tuple:
    """
    Setup common resources for a query
    
    Args:
        chat_id: Chat identifier
        query: User's question
        token: Authentication token
        db: Database session
        
    Returns:
        A tuple of (cancel_event, memory, system_prompt, task_db)
    """
    user = await get_user_from_token(token, db)
    await verify_chat_ownership(chat_id, user.id, db)
    
    # Create a cancellation event for this query
    cancel_event = register_active_query(chat_id)
    
    # Create a new independent DB session for saving messages
    task_db = await get_new_db_session()
    
    # Start saving the user message in background without waiting for it
    async def save_message_task():
        try:
            await save_message(task_db, chat_id, "user", query)
        except Exception as e:
            logger.error(f"Error in background save for user message: {e}")
            # Don't close the session here, it will be used for the assistant message
            
    save_task = asyncio.create_task(save_message_task())
    
    # Load conversation history
    memory = await load_memory(db, chat_id)
    
    # Get system instructions
    system_prompt = app.state.system_prompt
    
    # Return everything without waiting for the save to complete
    # Include the task_db so we can use it for saving assistant response
    return (cancel_event, memory, system_prompt, task_db, save_task)

async def create_save_response_task(chat_id: int, full_response: str, task_db: AsyncSession = None, user_msg_task: asyncio.Task = None):
    """
    Create a task to save the assistant's response to the database
    
    Args:
        chat_id: Chat identifier
        full_response: The complete response to save
        task_db: Database session already created for this chat (to reuse)
        user_msg_task: Task that saves the user message (to ensure proper message order)
    """
    # Define an async function to save the response
    async def save_assistant_message():
        try:
            # First, make sure user message save task has completed
            if user_msg_task is not None:
                try:
                    # Wait for user message to be saved with a timeout
                    await asyncio.wait_for(user_msg_task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for user message save in chat {chat_id}")
                except Exception as e:
                    logger.error(f"Error waiting for user message save in chat {chat_id}: {e}")
            
            # Use the provided DB session or create a new one if none
            session_to_use = task_db
            should_close = False
            
            if session_to_use is None or not hasattr(session_to_use, 'is_active') or not session_to_use.is_active:
                session_to_use = await get_new_db_session()
                should_close = True
                
            try:
                await save_message(session_to_use, chat_id, "assistant", full_response)
            finally:
                # Only close the session if we created it here
                if should_close:
                    await safe_close_session(session_to_use)
                else:
                    # If using an existing session, we close it here since this is the end of its use
                    await safe_close_session(session_to_use)
        except Exception as e:
            logger.error(f"Error in background save for assistant message: {e}")
    
    # Create task and register with task_manager
    task = asyncio.create_task(save_assistant_message())
    app.state.task_manager.add_task(chat_id, task)

@app.post("/start_query_session/{chat_id}")
async def start_query_session(
    chat_id: int,
    query: str = Form(...),
    images: List[UploadFile] = File(None),
    _: User = Depends(verify_chat_owner()),
    db: AsyncSession = Depends(get_db)
):
    """
    Start a new query session by uploading a long query and optional images.
    
    Args:
        chat_id: Chat identifier
        query: User's question
        images: Optional list of image files to include with the query
        db: Database session
        
    Returns:
        Session ID to use with query_chat
    """
    
    session_id = str(uuid.uuid4())
    session_data = {
        "chat_id": chat_id,
        "query": query,
        "created_at": time.time(),
        "images": []
    }
    
    # Process and store images in session
    if images:
        for img in images:
            if img is not None:
                try:
                    # Read image content
                    image_data = await img.read()
                    
                    # Convert to base64
                    import base64
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    
                    # Get proper extension from content type
                    extension = ".jpg"  # Default extension
                    if img.content_type:
                        if img.content_type == "image/png":
                            extension = ".png"
                        elif img.content_type == "image/gif":
                            extension = ".gif"
                        elif img.content_type == "image/webp":
                            extension = ".webp"
                        elif img.content_type == "image/svg+xml":
                            extension = ".svg"
                    
                    # Store image metadata in session for later use in query_chat
                    session_data["images"].append({
                        "filename": img.filename or f"image_{uuid.uuid4()}{extension}",
                        "content_type": img.content_type,
                        "base64": base64_image
                    })
                    
                    # Reset file pointer
                    await img.seek(0)
                except Exception as e:
                    logger.error(f"Error processing image in session: {e}")
    
    pending_queries[session_id] = session_data
    return {"session_id": session_id}

@app.get("/query/")
async def query_chat(
    chat_id: int,
    token: str,
    web_search: bool = False,
    query: Optional[str] = None,
    session_id: Optional[str] = None,
    kb_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Process general user queries and stream responses
    
    This endpoint requires either a direct query or a session_id from start_query_session.
    For queries with images, first call start_query_session and then use the returned
    session_id with this endpoint.
    
    Args:
        chat_id: Chat identifier
        token: Authentication token
        web_search: Whether to perform web search for up-to-date information
        query: User's question (required if session_id is not provided)
        session_id: Optional session ID from start_query_session (required for image queries)
        kb_name: Name of knowledge base to query (if None, use document context)
        db: Database session
        
    Returns:
        Streaming response in SSE (Server-Sent Events) format
    """
    if False:
        ### DEBUG
        await asyncio.sleep(3)
        async def stream_response1():
            error_text = "תשובה תשובה תשובה\nשורה 2 שורה 2 שורה 2\nשורה 3 שורה 3 שורה 3"
            async for chunk in stream_text_as_sse(error_text):
                yield chunk
        return StreamingResponse(
            stream_response1(), 
            media_type="text/event-stream",
            background=BackgroundTasks()
        )

    # Verify user authorization for GET request
    user = await get_user_from_token(token, db)
    await verify_chat_ownership(chat_id, user.id, db)

                
    # Process a user query, either directly or by session_id.
    if session_id:
        session = pending_queries.pop(session_id, None)
        if not session:
            raise HTTPException(status_code=400, detail="Invalid or expired session_id")
        if session["chat_id"] != chat_id:
            raise HTTPException(status_code=400, detail="chat_id mismatch with session_id")
        query = session["query"]
        # Get images from session if they exist
        image_contents = session.get("images", [])
    else:
        if not query:
            raise HTTPException(status_code=400, detail="Missing query")
        image_contents = []

    
    # Create a cancellation event for this query
    cancel_event = register_active_query(chat_id)
    
    # Save images to DB if they were included in the session
    if image_contents:
        for img in image_contents:
            try:
                # Get image format from content type
                img_format = "png"  # Default format
                if img["content_type"]:
                    if "jpeg" in img["content_type"] or "jpg" in img["content_type"]:
                        img_format = "jpg"
                    elif "png" in img["content_type"]:
                        img_format = "png"
                    elif "gif" in img["content_type"]:
                        img_format = "gif"
                    elif "webp" in img["content_type"]:
                        img_format = "webp"
                    elif "svg" in img["content_type"]:
                        img_format = "svg"
                
                # Create JSON structure for image
                img_json = json.dumps({
                    "type": "image",
                    "data": img["base64"],
                    "format": img_format
                })
                
                # Save image as a separate user message
                await save_message(db, chat_id, "user", img_json)
            except Exception as e:
                logger.error(f"Error saving image message from session: {e}")
    
    # Now save the text query as a user message
    await save_message(db, chat_id, "user", query)
    
    # Load conversation history (now includes image messages)
    memory = await load_memory(db, chat_id)
    
    # Get system instructions
    system_prompt = app.state.system_prompt
    
    # Prepare document context if available
    document_context = ""
    try:
        if kb_name:
            # Use specified knowledge base
            logger.info(f"Using knowledge base: {kb_name} for chat {chat_id}")
            kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
            if os.path.exists(kb_path):
                kb_rag = DocumentRAG(docs_dir=KNOWLEDGE_BASE_DIR, rag_storage_dir=KNOWLEDGE_BASE_DIR)
                # Use kb_name as the chat_id for the knowledge base
                document_context = await kb_rag.get_document_context(kb_name, query, top_k=5)
                
                if document_context:
                    # Prefix to indicate the source of the context
                    document_context = f"Information from knowledge base '{kb_name}':\n{document_context}"
                    logger.info(f"Retrieved context from knowledge base {kb_name} for chat {chat_id}")
                else:
                    logger.warning(f"No relevant context found in knowledge base {kb_name} for chat {chat_id}")
            else:
                logger.warning(f"Knowledge base {kb_name} not found")
        else:
            # Use document context from chat's documents
            doc_rag = get_document_rag(str(chat_id))
            document_context = await doc_rag.get_document_context(str(chat_id), query, top_k=3)
            
            if document_context:
                logger.info(f"Retrieved context from chat documents for chat {chat_id}")
                
        # If web search is requested, get additional context from web
        if web_search:
            logger.info(f"Web search requested for chat {chat_id}")
            # Import here to avoid circular imports
            from query_web import web_search_handler
            
            # Pass the database session to allow access to chat history
            web_context = await web_search_handler.get_document_context(
                str(chat_id), 
                query, 
                top_k=5,
                db=db  # Pass database session for history access
            )
            
            if web_context:
                # Add web search context either on top of document context or as the only context
                if document_context:
                    document_context += f"\n\n{web_context}"
                else:
                    document_context = web_context
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
    
    async def stream_response():
        """Internal function for streaming the response"""
        full_response = ""
        start_time = time.time()
        
        try:
            # Create message chain to send to the LLM
            conversation = [SystemMessage(content=system_prompt)] + memory.chat_memory.messages
            
            # Process image files if provided in the session
            # (No need to read files as they're already processed and stored in the session)
            
            # Add document context to the query if available
            enhanced_query = query
            if document_context:
                enhanced_query = f"{query}\n{document_context}"
                
            # Create the message content with text and images if available
            message_content = enhanced_query
            
            # If we have images from the session, create a multimodal message content
            if image_contents:
                message_content = [
                    {"type": "text", "text": enhanced_query}
                ]
                # Add each image as content part
                for img in image_contents:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['content_type']};base64,{img['base64']}"
                        }
                    })
            
            # Add user message with text and images
            conversation.append(HumanMessage(content=message_content))
            
            # Use llm_helpers to process langchain messages
            async def message_generator():
                async for content in await process_langchain_messages(
                    messages=conversation,
                    model_config="default",  # Use default model config as base
                    stream=True
                ):
                    # Check if the query has been cancelled
                    if should_cancel_query(chat_id):
                        logger.info(f"Query for chat {chat_id} was cancelled")
                        break
                    
                    nonlocal full_response
                    full_response += content
                    yield content
            
            # Use the SSE utility to stream the response
            async for chunk in stream_generator_as_sse(message_generator()):
                yield chunk
            
            # After receiving the complete response
            elapsed_time = time.time() - start_time
            logger.info(f"Query processed in {elapsed_time:.2f} seconds")
            
            # Skip saving if the query was cancelled or empty response
            if full_response:
                await create_save_response_task(chat_id, full_response, task_db, save_task)
            
        except Exception as e:
            # Handle errors and send them to the user
            logger.error(f"Error in stream_response for chat {chat_id}: {e}")
            error_text = f"[ERROR] {str(e)}"
            async for chunk in stream_text_as_sse(error_text):
                yield chunk
        finally:
            # Unregister the query from the active queries
            unregister_active_query(chat_id)
    
    # Return StreamingResponse object that will stream the response to the user
    return StreamingResponse(
        stream_response(), 
        media_type="text/event-stream",
        background=BackgroundTasks()
    )

@app.get("/query_code/")
async def query_code(
    chat_id: int, 
    query: str, 
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Process code-specific queries by including all code files in the prompt
    
    Args:
        chat_id: Chat identifier
        query: User's question
        token: Authentication token
        db: Database session
        
    Returns:
        Streaming response in SSE (Server-Sent Events) format
    """
    # Setup common resources
    cancel_event, memory, system_prompt, task_db, user_msg_task = await setup_query(chat_id, query, token, db)
    
    # Get code context from the external module
    code_context = get_code_context(chat_id)
    
    async def stream_response():
        """Internal function for streaming the response"""
        full_response = ""
        start_time = time.time()
        
        try:
            # Create message chain to send to the LLM
            conversation = [SystemMessage(content=system_prompt)] + memory.chat_memory.messages
            
            # Add user query with code context
            if code_context:
                enhanced_query = f"{query}\n{code_context}"
                conversation.append(HumanMessage(content=enhanced_query))
            else:
                conversation.append(HumanMessage(content=query))

            # Use llm_helpers to process langchain messages with code-specific configuration
            async def message_generator():
                async for content in await process_langchain_messages(
                    messages=conversation,
                    model_config="code",  # Use code-specific model configuration
                    stream=True
                ):
                    # Check if the query has been cancelled
                    if should_cancel_query(chat_id):
                        print(f"Code query for chat {chat_id} was cancelled")
                        break
                    
                    nonlocal full_response
                    full_response += content
                    yield content
            
            # Use the SSE utility to stream the response
            async for chunk in stream_generator_as_sse(message_generator()):
                yield chunk
            
            # After receiving the complete response
            elapsed_time = time.time() - start_time
            print(f"Code query processed in {elapsed_time:.2f} seconds")
            
            # Skip saving if the query was cancelled or empty response
            if full_response:
                await create_save_response_task(chat_id, full_response, task_db, user_msg_task)
            
        except Exception as e:
            # Handle errors and send them to the user
            print(f"Error in stream_response for chat {chat_id}: {e}")
            error_text = f"[ERROR] {str(e)}"
            async for chunk in stream_text_as_sse(error_text):
                yield chunk
        finally:
            # Unregister the query from the active queries
            unregister_active_query(chat_id)
    
    # Return StreamingResponse object that will stream the response to the user
    return StreamingResponse(
        stream_response(), 
        media_type="text/event-stream",
        background=BackgroundTasks()
    )

@app.post("/query_image/{chat_id}")
async def query_image(
    request: Request,
    chat_id: int,
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    size: str = Form("1024x1024"),
    quality: str = Form("standard"),
    style: str = Form("natural"),
    token: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Optimized endpoint for generating images within a chat
    
    Args:
        request: FastAPI Request object to extract base URL
        chat_id: Chat ID
        prompt: Text prompt for image generation or variation guidance
        image: Optional image file for variation (if None, creates new image)
        size: Image size
        quality: Image quality
        style: Image style
        token: Authentication token
        db: Database session
        
    Returns:
        Image metadata and chat message information
    """
    try:
        # Authentication and authorization
        user = await get_user_from_token(token, db)
        await verify_chat_ownership(chat_id, user.id, db)
        
        # Register this query for potential cancellation
        cancel_event = register_active_query(chat_id)
        
        # Save the user's message using the injected db session
        await save_message(db, chat_id, "user", prompt)
        
        # Extract base URL from the request
        base_url = str(request.base_url).rstrip('/')
        
        # Create a task for image generation
        async def generate_image_task():
            result = await get_image_service().generate_image(
                prompt=prompt,
                uploaded_image=image,
                size=size,
                quality=quality,
                style=style
            )
            
            # Modify the URL to be absolute if it's not already
            if result and "url" in result and result["url"].startswith("/"):
                # Convert relative URL to absolute using the request's base URL
                result["url"] = f"{base_url}{result['url']}"
                
            return result
        
        # Start the image generation as a task
        generation_task = asyncio.create_task(generate_image_task())

        # Update the active query with the task
        active_queries[str(chat_id)]["task"] = generation_task
        
        # Wait for either task completion or cancellation
        done = False
        result = None
        
        while True:
            # Check if the operation should be cancelled
            if should_cancel_query(chat_id):
                # Try to cancel the task
                if not generation_task.done():
                    generation_task.cancel()
                result = {"error": "Image generation was cancelled"}
                break
            
            # Check if the task is complete
            if generation_task.done():
                try:
                    result = await generation_task
                except asyncio.CancelledError:
                    result = {"error": "Image generation was cancelled"}
                except Exception as e:
                    result = {"error": f"Error in image generation task: {e}"}
                finally:
                    break
            
            await asyncio.sleep(0.1)

        # Unregister the query since it's complete
        unregister_active_query(chat_id)
        
        # Define a function to save the assistant message
        async def save_assistant_message():
            try:
                # Create a new DB session for the background task
                task_db = await get_new_db_session()
                try:
                    # Save the message
                    if "error" not in result:
                        message_context = json.dumps({"type": "image", "filename": result['filename'], "url": result['url'], "created": result['created']})
                    else:
                         message_context = result["error"]
                    await save_message(task_db, chat_id, "assistant",  message_context)
                finally:
                    # Always close the session
                    await safe_close_session(task_db)
            except Exception as e:
                logger.error(f"Error in background save for image response: {e}")
        
        # Create task and register with task_manager
        task = asyncio.create_task(save_assistant_message())
        app.state.task_manager.add_task(chat_id, task)

        # Return response
        return result
        
    except Exception as e:
        # Make sure to unregister the query if there was an error
        try:
            unregister_active_query(chat_id)
        except:
            pass
            
        error_message = f"Error processing image for chat: {str(e)}"
        return JSONResponse(
            status_code=500,
            content={"error": error_message}
        )