"""
Database session management module

This module:
1. Provides functions for database session management
2. Includes utilities for memory loading and message saving
3. Contains session monitoring and cleanup functions
"""

import asyncio
import time
import datetime
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import SessionLocal, Chat, Message, User, get_engine_status
from fastapi import HTTPException
from langchain.memory import ConversationBufferWindowMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_manager")

async def get_db():
    """
    Async dependency to provide database session and handle proper cleanup
    
    This implementation ensures the database session is properly managed even
    in concurrent scenarios or when used in background tasks.
    
    Yields:
        AsyncSession: Database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        # Log the exception but don't close the session here - might be used by others
        logger.error(f"Exception in get_db: {e}")
        # Only rollback, don't close
        await db.rollback()
        raise
    finally:
        print("closing db")
        # Safe close that checks session state first
        if db.is_active:
            try:
                await db.close()
            except Exception as e:
                logger.error(f"Error closing DB session: {e}")

# Create a new independent database session
async def get_new_db_session():
    """
    Create a new independent database session
    
    This is useful for background tasks or operations that need
    their own dedicated session.
    
    Returns:
        AsyncSession: New database session
    """
    return SessionLocal()

# Helper function to safely close a session
async def safe_close_session(db, timeout=5):
    """Close session with timeout guarantee"""
    if db is not None:
        try:
            if db.is_active:
                # הוספת timeout לסגירת החיבור
                close_task = asyncio.create_task(db.close())
                try:
                    await asyncio.wait_for(close_task, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Session close timeout after {timeout}s")
        except Exception as e:
            logger.error(f"Error closing session: {e}")

async def load_memory(db: AsyncSession, chat_id: int):
    """
    Load conversation history from the database
    
    Args:
        db: Database session
        chat_id: Chat ID to load history for
        
    Returns:
        LangChain memory object with loaded messages
    """
    memory = ConversationBufferWindowMemory(k=10, return_messages=True)
    
    try:
        # Query messages with async syntax
        result = await db.execute(
            select(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp)
        )
        messages = result.scalars().all()
        
        for msg in messages:
            if msg.sender == "user":
                memory.chat_memory.add_user_message(msg.content)
            else:
                memory.chat_memory.add_ai_message(msg.content)
    except Exception as e:
        print(f"Error loading memory: {e}")
        # Continue with empty memory if there's an error
    
    return memory

async def save_message(db: AsyncSession, chat_id: int, sender: str, content: str, max_retries: int = 3):
    """
    Save a message to the database with retry capability and session safety
    
    Args:
        db: Database session
        chat_id: Chat ID to save message in
        sender: Message sender ("user" or "assistant")
        content: Message content
        max_retries: Maximum number of retry attempts
    """
    retries = 0
    own_session = False
    session = db
    
    while retries < max_retries:
        try:
            # Check if provided session is valid
            if session is None or not hasattr(session, 'is_active') or not session.is_active:
                # Create a new session if needed
                session = await get_new_db_session()
                own_session = True
                
            message = Message(chat_id=chat_id, sender=sender, content=content)
            session.add(message)
            await session.commit()
            return
        except Exception as e:
            retries += 1
            logger.error(f"Error saving message (attempt {retries}/{max_retries}): {e}")
            
            # Safely rollback
            try:
                if hasattr(session, 'rollback'):
                    await session.rollback()
            except Exception as rollback_err:
                logger.error(f"Error during rollback: {rollback_err}")
                
            if retries < max_retries:
                # Exponential backoff
                await asyncio.sleep(0.5 * (2 ** retries))
                
                # If session has issues, create a new one
                if "connection" in str(e).lower() or "closed" in str(e).lower() or "this transaction is closed" in str(e).lower():
                    if own_session and hasattr(session, 'close'):
                        # Close previous session if we created it
                        await safe_close_session(session)
                    
                    # Create new session
                    session = await get_new_db_session()
                    own_session = True
                    logger.warning(f"Created replacement DB session after error (chat_id: {chat_id})")
            else:
                logger.error(f"Failed to save message after {max_retries} attempts for chat {chat_id}")
        finally:
            # Only close if we created our own session and we're done with retries
            if own_session and retries >= max_retries and hasattr(session, 'close'):
                await safe_close_session(session)

# Function to detect and clean up stale database sessions
async def cleanup_stale_db_sessions():
    """
    Background task to detect and clean up stale database sessions
    
    This should be run periodically to ensure the pool doesn't get exhausted
    """
    try:
        # Get current pool stats
        pool_stats = await get_engine_status()
        
        # Log the current state
        logger.info(f"DB Pool status: {pool_stats}")
        
        # If too many connections are checked out for too long, log a warning
        if pool_stats["checked_out_connections"] > pool_stats["pool_size"] * 0.8:
            logger.warning("WARNING: High number of checked out database connections")
            
        # Actual cleanup is handled by SQLAlchemy's pool_recycle parameter,
        # but we could add additional cleanup logic here if needed
    except Exception as e:
        logger.error(f"Error in cleanup_stale_db_sessions: {e}")

# Function to get database connection pool status
async def get_db_status_data():
    """
    Get current database connection pool stats
    
    Returns:
        Dict with pool status information and a timestamp
    """
    try:
        # Get pool stats
        pool_stats = await get_engine_status()
        
        return {
            "status": "healthy",
            "pool_stats": pool_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting DB status data: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }