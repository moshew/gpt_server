"""
Application initialization module

This module:
1. Sets up the FastAPI application
2. Configures middleware and global settings
3. Registers startup events
4. Initializes the code analyzer, document RAG, and web search handler
"""

import os
import openai
import asyncio
import time, datetime
import contextlib
from typing import Optional, AsyncGenerator, Dict, Any, List, Set, Tuple
from functools import partial
from pydantic import BaseModel

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi import Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .database import SessionLocal, Chat, Message, User, get_engine_status

# Import the code analysis handler and code-specific endpoints
#from code_rag import CodeAnalysisHandler

# Import modules containing the application endpoints
from .db_manager import cleanup_stale_db_sessions

# Create FastAPI app with proxy headers configuration
app = FastAPI(
    title="Code & Document Analysis Chat System", 
    description="A chat system with sophisticated code analysis, document retrieval, and web search capabilities",
    version="1.0.0",
    # Configure for reverse proxy
    root_path="",
    servers=[{"url": "/"}]
)

# Configure CORS to ensure streaming works properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for reverse proxy
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Middleware to handle proxy headers
@app.middleware("http")
async def proxy_headers_middleware(request, call_next):
    # Handle X-Forwarded-Proto header for HTTPS detection
    if "x-forwarded-proto" in request.headers:
        request.scope["scheme"] = request.headers["x-forwarded-proto"]
    
    # Handle X-Forwarded-Host header
    if "x-forwarded-host" in request.headers:
        request.scope["server"] = (request.headers["x-forwarded-host"], None)
    
    response = await call_next(request)
    return response

# Track background tasks by chat_id
class BackgroundTaskManager:
    def __init__(self):
        self.tasks: Dict[int, Set[asyncio.Task]] = {}
        self.locks: Dict[int, asyncio.Lock] = {}
    
    def get_lock(self, chat_id: int) -> asyncio.Lock:
        """Get or create a lock for a specific chat"""
        if chat_id not in self.locks:
            self.locks[chat_id] = asyncio.Lock()
        return self.locks[chat_id]
    
    @contextlib.asynccontextmanager
    async def chat_lock(self, chat_id: int):
        """Context manager for locking chat operations"""
        lock = self.get_lock(chat_id)
        try:
            await lock.acquire()
            yield
        finally:
            lock.release()
    
    def add_task(self, chat_id: int, task: asyncio.Task):
        """Register a background task for a chat"""
        if chat_id not in self.tasks:
            self.tasks[chat_id] = set()
        self.tasks[chat_id].add(task)
        
        # Set up task cleanup when done
        task.add_done_callback(lambda t: self._cleanup_task(chat_id, t))
    
    def _cleanup_task(self, chat_id: int, task: asyncio.Task):
        """Remove a completed task"""
        if chat_id in self.tasks and task in self.tasks[chat_id]:
            self.tasks[chat_id].remove(task)
            # Clean up empty task sets
            if not self.tasks[chat_id]:
                del self.tasks[chat_id]
    
    def get_chat_tasks(self, chat_id: int) -> Set[asyncio.Task]:
        """Get all tasks for a chat"""
        return self.tasks.get(chat_id, set())
    
    def has_running_tasks(self, chat_id: int) -> bool:
        """Check if a chat has running tasks"""
        return chat_id in self.tasks and len(self.tasks[chat_id]) > 0

# Function to load available models from file
def load_available_models() -> List[Tuple[str, str]]:
    """Load available models from file"""
    models = []
    try:
        with open("config/avaliable_models.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(';', 1)
                    if len(parts) == 2:
                        model_name, description = parts
                        models.append((model_name, description))
    except Exception as e:
        print(f"Error loading available models: {e}")
    return models

# Async function to load system prompt
async def load_system_prompt():
    """Load system prompt from file asynchronously"""
    import aiofiles
    
    try:
        async with aiofiles.open("config/system_prompt.txt", "r", encoding="utf-8") as f:
            return await f.read()
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        # Fallback prompt in case file is not found
        return "You are a helpful AI assistant specialized in code analysis, document retrieval, and web search."

# Initialize system prompt, code analyzer, document RAG, and web search handler asynchronously at startup
@app.on_event("startup")
async def startup_event():
    """Initialize components at application startup"""
    # Load system prompt
    app.state.system_prompt = await load_system_prompt()
    
    # Load available models
    app.state.available_models = load_available_models()
    print(f"Loaded {len(app.state.available_models)} available models")
    
    # Create necessary directories
    create_directories()

    # Initialize code analyzer
    #app.state.code_analyzer = CodeAnalysisHandler(chats_dir="chats")
    
    # Import document_rag after FastAPI app is created to avoid circular imports
    from .rag_documents import document_rag_handler
    
    # Store document RAG handler in app state
    app.state.document_rag_handler = document_rag_handler
    
    # Import web search handler after FastAPI app is created
    from .query_web import web_search_handler
    
    # Store web search handler in app state
    app.state.web_search_handler = web_search_handler
    
    # Initialize the background task manager
    app.state.task_manager = BackgroundTaskManager()
    
    print("Application initialized successfully")

# Add this to the startup event to schedule periodic cleanup
@app.on_event("startup")
async def schedule_db_cleanup():
    """Schedule periodic database session cleanup"""
    async def cleanup_task():
        while True:
            await cleanup_stale_db_sessions()
            # Run every 5 minutes
            await asyncio.sleep(300)
    
    # Start the cleanup task in the background
    asyncio.create_task(cleanup_task())

# Health check endpoint that includes DB pool status
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running
    
    Returns:
        Health status information including database pool stats
    """
    db_status = await get_engine_status()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "db_pool": db_status
    }

# Create necessary directories
def create_directories():
    """Create all necessary directories for the application"""
    base_data_dir = "data"
    
    directories = [
        base_data_dir,
        f"{base_data_dir}/chats",
        f"{base_data_dir}/code", 
        f"{base_data_dir}/images",
        f"{base_data_dir}/knowledgebase",
        f"{base_data_dir}/rag"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

# Call the function to create directories
create_directories()