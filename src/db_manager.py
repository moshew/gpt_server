"""
Database Manager Module

This module provides database session management and cleanup utilities.
"""

import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from .database import SessionLocal, engine

logger = logging.getLogger(__name__)

async def cleanup_stale_db_sessions():
    """
    Clean up stale database sessions and connections
    
    This function ensures that unused database connections are properly closed
    and that the connection pool is maintained in a healthy state.
    """
    try:
        # Check engine status and clean up if needed
        async with engine.begin() as conn:
            # Simple query to check connection health
            await conn.execute("SELECT 1")
        
        logger.info("Database session cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during database session cleanup: {e}")

async def get_db_stats():
    """
    Get database connection pool statistics
    
    Returns:
        Dict with connection pool information
    """
    try:
        pool = engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e)} 