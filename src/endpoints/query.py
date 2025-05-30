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
import base64

from typing import Optional, List
from fastapi import Depends, BackgroundTasks, HTTPException, Query, File, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ..database import User, Message, Chat
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from ..app_init import app
from ..auth import get_user_from_token, verify_chat_owner, verify_chat_ownership
from ..database import SessionLocal
from ..utils import process_langchain_messages
from ..utils.sse import stream_text_as_sse, stream_generator_as_sse
from ..query_docs import get_document_context, get_document_rag
from ..query_images import get_image_service
from ..query_web import web_search_handler
from ..query_code import get_code_context

# Dictionary to track active queries, with chat_id as key and task/event objects as values
active_queries = {}
pending_queries = {}

# Session cleanup settings
SESSION_EXPIRY_MINUTES = 30  # Sessions expire after 30 minutes

CONTENT_TYPE_TO_EXT = {
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/svg+xml": "svg"
}

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

async def load_memory(db: AsyncSession, chat_id: int):
    """
    Load conversation history from the database
    
    This function loads the last 20 messages (10 user-assistant pairs) and processes them:
    - User text messages: Added to memory as-is
    - User image messages: Converted to multimodal format for LLM
    - User multimodal messages: Added as-is (contains text + images)
    - Assistant text messages: Added to memory as-is  
    - Assistant image messages: Skipped (generated images not included in context)
    
    Args:
        db: Database session
        chat_id: Chat ID to load history for
        
    Returns:
        LangChain memory object with loaded messages
    """
    memory = InMemoryChatMessageHistory()
    
    try:
        # Query messages with async syntax, limit to last 20 messages (10 pairs)
        result = await db.execute(
            select(Message).filter(Message.chat_id == chat_id)
            .order_by(Message.timestamp.desc())
            .limit(20)
        )
        messages = list(reversed(result.scalars().all()))  # Reverse to get chronological order
        
        for msg in messages:
            if msg.sender == "user":
                # Check if this is an image message
                try:
                    content_json = json.loads(msg.content)
                    if isinstance(content_json, dict) and content_json.get("type") == "image":
                        # This is a user image message - add it to memory
                        image_data = content_json.get("data", "")
                        image_format = content_json.get("format", "image/jpeg")
                        
                        # Create multimodal message content for image
                        multimodal_content = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_format};base64,{image_data}"
                                }
                            }
                        ]
                        
                        memory.add_user_message(multimodal_content)
                        logger.info(f"Added user image message to memory for chat {chat_id}")
                    else:
                        # Regular text message or other JSON content
                        memory.add_user_message(msg.content)
                except (json.JSONDecodeError, TypeError):
                    # Check if content might be a multimodal message (list)
                    try:
                        if isinstance(msg.content, str) and msg.content.startswith('['):
                            # Try to parse as list
                            content_list = json.loads(msg.content)
                            if isinstance(content_list, list):
                                # This might be a multimodal message - keep it as is
                                memory.add_user_message(content_list)
                                logger.info(f"Added multimodal user message to memory for chat {chat_id}")
                            else:
                                # Regular text message
                                memory.add_user_message(msg.content)
                        else:
                            # Regular text message
                            memory.add_user_message(msg.content)
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON - treat as regular text message
                        memory.add_user_message(msg.content)
            else:
                # Assistant message
                try:
                    content_json = json.loads(msg.content)
                    if isinstance(content_json, dict) and content_json.get("type") == "image":
                        # Skip assistant image messages (like generated images)
                        logger.info(f"Skipping assistant image message from memory for chat {chat_id}")
                        continue
                    else:
                        # Regular assistant text message
                        memory.add_ai_message(msg.content)
                except (json.JSONDecodeError, TypeError):
                    # Not JSON or invalid JSON - treat as regular text message
                    memory.add_ai_message(msg.content)
    except Exception as e:
        logger.error(f"Error loading memory for chat {chat_id}: {e}")
        # Continue with empty memory if there's an error
    
    return memory

async def save_message(db: AsyncSession, chat_id: int, sender: str, content: str):
    """
    Save a message to the database with retry capability and session safety
    
    Args:
        db: Database session
        chat_id: Chat ID to save message in
        sender: Message sender ("user" or "assistant")
        content: Message content
    """
    try:
        message = Message(chat_id=chat_id, sender=sender, content=content)
        db.add(message)
        await db.commit()
    except Exception as e:
        logger.error(f"Error saving message: {e}")

async def perform_indexing_with_progress(doc_rag: 'DocumentRAG', chat_id: str):
    """
    Perform indexing with progress updates via SSE
    
    Args:
        doc_rag: DocumentRAG instance
        chat_id: Chat identifier
        
    Yields:
        SSE formatted progress messages
    """
    try:
        # Send start indexing message (raw SSE format without [DONE])
        start_message = "###PROC_INFO: Indexing uploaded documents...###"
        yield f"data: {start_message}\n\n"
        
        # Perform actual indexing
        result = await doc_rag.index_documents(chat_id)
        
        logger.info(f"Indexing completed for chat {chat_id}: {result}")
        
        # Send completion message (clear the processing info) - raw SSE format without [DONE]
        end_message = "###PROC_INFO:###"
        yield f"data: {end_message}\n\n"
            
    except Exception as e:
        logger.error(f"Error during indexing for chat {chat_id}: {e}")
        # Send error completion message - raw SSE format without [DONE]
        error_message = "###PROC_INFO:###"
        yield f"data: {error_message}\n\n"

# Endpoint to stop an active query
@app.post("/stop_query/{chat_id}")
async def stop_query(
    chat_id: int,
    token: str
):
    """
    Stop an active query for a specific chat
    
    Args:
        chat_id: ID of the chat with the query to stop
        token
        
    Returns:
        Status message indicating whether the query was stopped
    """
    async with SessionLocal() as db:
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

# --- Helper functions for query processing ---

async def _get_session_data(session_id: str, chat_id: int) -> tuple[Optional[str], List[dict]]:
    """
    Retrieve and validate session data
    
    Returns:
        tuple of (query, image_contents)
    """
    if not session_id:
        return None, []
    
    logger.info(f"Processing query with session_id: {session_id} for chat {chat_id}")
    
    session = pending_queries.get(session_id)
    if not session:
        # Only clean up if session not found - might be expired
        logger.error(f"Session {session_id} not found. Available sessions: {list(pending_queries.keys())}")
        raise HTTPException(status_code=400, detail="Invalid or expired session_id")
        
    # Verify chat_id matches
    if session["chat_id"] != chat_id:
        logger.error(f"Chat ID mismatch: session has {session['chat_id']}, request has {chat_id}")
        raise HTTPException(status_code=400, detail="chat_id mismatch with session_id")
        
    # Check if session is expired
    current_time = time.time()
    session_age = current_time - session.get("created_at", 0)
    
    if session_age > (SESSION_EXPIRY_MINUTES * 60):
        logger.error(f"Session {session_id} has expired")
        del pending_queries[session_id]
        raise HTTPException(status_code=400, detail="Session has expired")
        
    query = session["query"]
    image_contents = session.get("images", [])
    logger.info(f"Using session {session_id} with query='{query}' and {len(image_contents)} images")
    
    return query, image_contents

def _prepare_image_messages(image_contents: List[dict], chat_id: int) -> List[str]:
    """
    Convert image contents to JSON messages
    
    Returns:
        List of JSON strings for image messages
    """
    img_json_messages = []
    if not image_contents:
        return img_json_messages
        
    for img in image_contents:
        try:
            img_json = json.dumps({
                "type": "image",
                "data": img["base64"],
                "format": img["content_type"]
            })
            img_json_messages.append(img_json)
            logger.info(f"Prepared image message for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error preparing image message: {e}")
    
    return img_json_messages

def _parse_source(source: Optional[str]) -> tuple[bool, Optional[str], bool]:
    """
    Parse source parameter into component flags
    
    Args:
        source: Source string (None, "code", "web", "kb.<name>")
        
    Returns:
        tuple of (is_code, kb_name, web_search)
    """
    if source == "code":
        return True, None, False
    elif source == "web":
        return False, None, True
    elif source and source.startswith("kb."):
        kb_name = source[3:]  # Remove "kb." prefix
        return False, kb_name if kb_name else None, False
    else:  # None or any other value defaults to standard document mode
        return False, None, False

async def _get_document_context(
    chat_id: int, 
    query: Optional[str], 
    is_code: bool, 
    kb_name: Optional[str], 
    web_search: bool,
    keep_original_files: bool
) -> str:
    """
    Get document context based on query type and parameters
    
    Returns:
        Document context string
    """
    document_context = ""
    
    try:
        if is_code:
            # Use code context for code-specific queries
            logger.info(f"Using code context for chat {chat_id}")
            code_context = get_code_context(chat_id)
            document_context = code_context if code_context else ""
                
        elif kb_name:
            # Use specified knowledge base
            logger.info(f"Using knowledge base: {kb_name} for chat {chat_id}")
            document_context = await get_document_context(
                kb_name, query, top_k=5, keep_original_files=False, source_type="kb"
            )
            
        else:
            # Use document context from chat's documents
            document_context = await get_document_context(
                str(chat_id), query, top_k=3, keep_original_files=keep_original_files
            )
            
        # Add web search context if requested (not for code queries or original files)
        if web_search and query and not is_code and not keep_original_files:
            logger.info(f"Web search requested for chat {chat_id}")
            async with SessionLocal() as db:
                web_context = await web_search_handler.get_document_context(
                    db=db, chat_id=str(chat_id), query=query, top_k=5
                )
            
            if web_context:
                if document_context:
                    document_context += f"\n\n{web_context}"
                else:
                    document_context = web_context
            
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        # Re-raise HTTP exceptions (like word limit exceeded)
        if isinstance(e, HTTPException):
            raise
    
    return document_context

def _build_message_content(query: Optional[str], document_context: str, image_contents: List[dict]) -> any:
    """
    Build message content combining text and images
    
    Returns:
        Message content (string or list for multimodal)
    """
    # Combine query with document context
    enhanced_query = query or ""
    if document_context:
        if query:
            enhanced_query = f"{query}\n{document_context}"
        else:
            enhanced_query = document_context
    
    # Handle multimodal content if images are present
    if image_contents:
        message_content = []
        if enhanced_query:
            message_content.append({"type": "text", "text": enhanced_query})
        
        for img in image_contents:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['content_type']};base64,{img['base64']}"
                }
            })
        return message_content
    
    return enhanced_query if enhanced_query else ""

# --- Shared helper functions for queries ---

async def setup_query(
    chat_id: int,
    query: Optional[str],
    token: str,
    img_json_messages: Optional[List[str]] = None,
    keep_original_files: bool = False
) -> tuple:
    """
    Setup common resources for a query
    
    Args:
        chat_id: Chat identifier
        query: User's question (can be None if only images)
        token: Authentication token
        img_json_messages: List of image JSON messages
        keep_original_files: Whether to send original file contents instead of RAG excerpts (may hit token limits)
        
    Returns:
        A tuple of (cancel_event, memory, system_prompt)
    """
    # Ensure we have at least query or images
    if not query and not img_json_messages:
        raise HTTPException(status_code=400, detail="Must provide either query text or images")
    
    async with SessionLocal() as db:
        user = await get_user_from_token(token, db)
        await verify_chat_ownership(chat_id, user.id, db)
        
        # Update chat with keep_original_files preference only if different
        result = await db.execute(
            select(Chat).filter(Chat.id == chat_id)
        )
        chat = result.scalars().first()
        if chat and chat.keep_original_files != keep_original_files:
            chat.keep_original_files = keep_original_files
            await db.commit()
        
        # Save query message if provided
        if query:
            await save_message(db, chat_id, "user", query)
        
        # Save image messages if provided
        if img_json_messages:
            for img_json in img_json_messages:
                await save_message(db, chat_id, "user", img_json)
        
        # Load conversation history
        memory = await load_memory(db, chat_id)
        
    # Create a cancellation event for this query
    cancel_event = register_active_query(chat_id)
    return (cancel_event, memory, app.state.system_prompt)

@app.post("/start_query_session/{chat_id}")
async def start_query_session(
    chat_id: int,
    query: Optional[str] = Form(None),
    images: List[UploadFile] = File(None),
    _: User = Depends(verify_chat_owner()),
):
    """
    Start a new query session by uploading a long query and optional images.
    
    Args:
        chat_id: Chat identifier
        query: User's question (optional if images are provided)
        images: Optional list of image files to include with the query
        
    Returns:
        Session ID to use with query_chat
    """
    # Ensure at least query or images are provided
    if not query and not images:
        raise HTTPException(status_code=400, detail="Must provide either query text or images")
    
    session_id = str(uuid.uuid4())
    session_data = {
        "chat_id": chat_id,
        "query": query,  # Keep None if None, don't convert to empty string
        "created_at": time.time(),
        "images": []
    }
    
    logger.info(f"Creating new session {session_id} for chat {chat_id} with query='{query}' and {len(images) if images else 0} images")
    
    # Process and store images in session
    if images:
        for img in images:
            if img is not None:
                try:
                    # Read image content
                    image_data = await img.read()
                    
                    # Convert to base64
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    
                    # Get proper extension from content type
                    extension = CONTENT_TYPE_TO_EXT.get(img.content_type, "jpg")
                    
                    # Store image metadata in session for later use in query_chat
                    session_data["images"].append({
                        "filename": img.filename or f"image_{uuid.uuid4()}.{extension}",
                        "content_type": img.content_type,
                        "base64": base64_image
                    })
                    
                    # Reset file pointer
                    await img.seek(0)
                    logger.info(f"Processed image {img.filename} for session {session_id}")
                except Exception as e:
                    logger.error(f"Error processing image {img.filename} in session {session_id}: {e}")
    
    pending_queries[session_id] = session_data
    logger.info(f"Session {session_id} created successfully. Total pending sessions: {len(pending_queries)}")
    
    return {"session_id": session_id}

@app.get("/query/")
async def query_chat(
    chat_id: int,
    token: str,
    query: Optional[str] = None,
    session_id: Optional[str] = None,
    deployment_name: Optional[str] = "GPT-4.1",
    keep_original_files: bool = False,
    source: Optional[str] = None,  # None (default), "code", "web", "kb.<name>"
):
    """
    Process user queries and stream responses
    
    This endpoint handles both general queries and code-specific queries.
    It requires either a direct query or a session_id from start_query_session.
    For queries with images, first call start_query_session and then use the returned
    session_id with this endpoint.
    
    For chat documents (source=None), the function automatically checks if indexing
    is needed and performs it with progress updates via SSE before processing the query.
    
    Source Types:
    - None (default): Use document context from chat documents (auto-indexes if needed)
    - "code": Use code files context 
    - "web": Use web search for up-to-date information
    - "kb.<name>": Use specific knowledge base (e.g., "kb.confluence")
    
    Examples:
        /query/?chat_id=1&query=hello                        # Default documents (no source)
        /query/?chat_id=1&query=fix bug&source=code          # Code analysis
        /query/?chat_id=1&query=latest news&source=web       # Web search
        /query/?chat_id=1&query=policy&source=kb.confluence  # Knowledge base
    
    Args:
        chat_id: Chat identifier
        token: Authentication token
        query: User's question (required if session_id is not provided)
        session_id: Optional session ID from start_query_session (required for image queries)
        deployment_name: Name of the model deployment to use (default: GPT-4.1)
        keep_original_files: Whether to send original file contents instead of RAG excerpts (may hit token limits)
        source: Context source type - None (default), "code", "web", or "kb.<name>"
        
    Returns:
        Streaming response in SSE (Server-Sent Events) format
        Special SSE messages during indexing (source=None only):
        - ###PROC_INFO: Indexing uploaded documents...### (start)
        - ###PROC_INFO:### (completion)
    """
    
    # Get query and images from session or direct input
    if session_id:
        query, image_contents = await _get_session_data(session_id, chat_id)
    else:
        if not query:
            raise HTTPException(status_code=400, detail="Missing query - provide either query text or use session_id for images")
        image_contents = []

    # Prepare image messages
    img_json_messages = _prepare_image_messages(image_contents, chat_id)
    
    # Setup query authentication, memory, and save messages
    cancel_event, memory, system_prompt = await setup_query(
        chat_id, query, token, img_json_messages if img_json_messages else None, keep_original_files
    )
    
    # Variable to store the complete response for saving
    complete_response = {"content": ""}
    
    async def stream_response():
        """Stream the LLM response"""
        start_time = time.time()
        
        try:
            # Check if indexing is needed for default source (chat documents) and not using original files
            if source is None and not keep_original_files:  # Default source = chat documents AND not using original files
                doc_rag = get_document_rag(str(chat_id))
                indexing_needed = await doc_rag.check_indexing_needed(str(chat_id))
                if indexing_needed:
                    logger.info(f"Indexing needed for chat {chat_id}, performing indexing...")
                    # Stream indexing progress messages
                    async for indexing_chunk in perform_indexing_with_progress(doc_rag, str(chat_id)):
                        yield indexing_chunk
                    logger.info(f"Indexing completed for chat {chat_id}")
            
            # Parse source and get document context after potential indexing
            is_code, kb_name, web_search = _parse_source(source)
            document_context = await _get_document_context(
                chat_id, query, is_code, kb_name, web_search, keep_original_files
            )
            
            # Build message content combining text and images
            message_content = _build_message_content(query, document_context, image_contents)
            
            # Create conversation with system prompt, history, and user message
            conversation = [SystemMessage(content=system_prompt)] + memory.messages
            conversation.append(HumanMessage(content=message_content))
            
            # Stream response from LLM
            async def message_generator():
                first_chunk = True
                async for content in await process_langchain_messages(
                    messages=conversation,
                    model_config="default",
                    stream=True,
                    deployment_name=deployment_name
                ):
                    # Check for cancellation
                    if should_cancel_query(chat_id):
                        query_type = "Code query" if is_code else "Query"
                        logger.info(f"{query_type} for chat {chat_id} was cancelled")
                        break
                    
                    # Clean up session on first chunk
                    if first_chunk and session_id and session_id in pending_queries:
                        try:
                            del pending_queries[session_id]
                            logger.info(f"Removed session {session_id} from pending_queries after first chunk")
                        except KeyError:
                            logger.warning(f"Session {session_id} was already removed from pending_queries")
                        first_chunk = False
                    
                    # Accumulate response content
                    complete_response["content"] += content
                    yield content
            
            # Stream to client
            async for chunk in stream_generator_as_sse(message_generator()):
                yield chunk
            
            # Log completion
            elapsed_time = time.time() - start_time
            query_type = "Code query" if is_code else "Query"
            logger.info(f"{query_type} processed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in stream_response for chat {chat_id}: {e}")
            error_text = f"[ERROR] {str(e)}"
            complete_response["content"] = error_text  # Save error as well
            async for chunk in stream_text_as_sse(error_text):
                yield chunk
        finally:
            # Always save the response, regardless of how the stream ended
            if complete_response["content"]:
                try:
                    async with SessionLocal() as db:
                        await save_message(db, chat_id, "assistant", complete_response["content"])
                        logger.info(f"Assistant response saved successfully for chat {chat_id}")
                except Exception as save_error:
                    logger.error(f"Failed to save assistant response for chat {chat_id}: {save_error}")
            
            unregister_active_query(chat_id)
    
    return StreamingResponse(
        stream_response(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
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
    result = None
    try:
        # Authentication and authorization
        async with SessionLocal() as db:
            user = await get_user_from_token(token, db)
            await verify_chat_ownership(chat_id, user.id, db)
            await save_message(db, chat_id, "user", prompt)

        # Register this query for potential cancellation
        cancel_event = register_active_query(chat_id)
        
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

        # Return response
        return result
        
    except Exception as e:
        error_message = f"Error processing image for chat: {str(e)}"
        result = {"error": error_message}
        return JSONResponse(
            status_code=500,
            content=result
        )
    finally:
        # Always save the assistant message and unregister query
        try:
            if result:
                message_content = json.dumps({"type": "image", "filename": result.get('filename', ''), "url": result.get('url', ''), "created": result.get('created', '')}) if "error" not in result else result["error"]
                
                async with SessionLocal() as db:
                    await save_message(db, chat_id, "assistant", message_content)
                    logger.info(f"Assistant image response saved successfully for chat {chat_id}")
        except Exception as save_error:
            logger.error(f"Failed to save assistant image response for chat {chat_id}: {save_error}")
        
        # Always unregister the query
        try:
            unregister_active_query(chat_id)
        except:
            pass

# Endpoint to get available models
@app.get("/query/available_models")
async def get_available_models():
    """
    Get a list of available AI models with their descriptions
    
    Returns:
        List of model objects with name and description
    """
    models = [
        {"name": model_name, "description": description}
        for model_name, description in app.state.available_models
    ]
    
    return JSONResponse(content={"models": models})

# Debug endpoint for pending sessions
@app.get("/debug/pending_sessions")
async def get_pending_sessions():
    """
    Debug endpoint to check pending sessions
    
    Returns:
        Information about pending sessions
    """
    current_time = time.time()
    session_info = []
    
    for session_id, session_data in pending_queries.items():
        age_minutes = (current_time - session_data.get("created_at", 0)) / 60
        session_info.append({
            "session_id": session_id,
            "chat_id": session_data.get("chat_id"),
            "age_minutes": round(age_minutes, 2),
            "image_count": len(session_data.get("images", [])),
            "query_preview": session_data.get("query", "")[:100] + "..." if len(session_data.get("query", "")) > 100 else session_data.get("query", "")
        })
    
    return {
        "total_sessions": len(pending_queries),
        "session_expiry_minutes": SESSION_EXPIRY_MINUTES,
        "sessions": session_info
    }