# Updated database.py with improved connection pooling
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import secrets
import bcrypt
import datetime
import sys
import os

# Add parent directory to path for config access
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.config import DATABASE_URL

# Check if DATABASE_URL is set, provide default if not
if not DATABASE_URL:
    print("Warning: DATABASE_URL not set in environment variables. Using default PostgreSQL database.")
    DATABASE_URL = "postgresql://user:password@localhost:5432/gpt_server"

# Convert regular SQLAlchemy DATABASE_URL to async version
# Replace postgresql:// with postgresql+asyncpg:// if needed
if DATABASE_URL.startswith('postgresql://'):
    ASYNC_DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://', 1)
else:
    ASYNC_DATABASE_URL = DATABASE_URL

# Enhanced engine configuration with better pool settings
engine = create_async_engine(
    ASYNC_DATABASE_URL, 
    # Increase pool size for better concurrency
    pool_size=20,  # Increased from 10
    max_overflow=10,  # Increased from 5
    pool_timeout=30,  # Add timeout to avoid hanging forever
    pool_pre_ping=True,  # Check connection validity before using
    pool_recycle=3600,  # Recycle connections after an hour to prevent stale connections
    future=True
)

# Enhanced session configuration
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine, 
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# User model for authentication and API key management
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    api_key = Column(String, unique=True)

    def __init__(self, username):
        self.username = username
        key = secrets.token_hex(16)
        self.api_key = key  # Store plain key for now (will be hashed in create_user)
        return key  # Return plaintext key to be provided to user

# Chat model representing a conversation session
class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    chat_name = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Message model to store conversation history
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    sender = Column(String)  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# File model to store chat files
class File(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    file_type = Column(String(10), default="doc")  # "doc" or "code"
    file_name = Column(String(255), nullable=False)

# Function to get engine status - useful for monitoring
async def get_engine_status():
    """
    Get current database engine pool status
    
    Returns:
        Dict with pool statistics
    """
    # Access the internal pool status
    pool = engine.pool
    
    return {
        "pool_size": pool.size(),
        "checked_out_connections": pool.checkedout(),
        "overflow": pool.overflow(),
        "checkedin_connections": pool.checkedin(),
    }

# Initialize database tables - now async
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
# Run this from a script to initialize the DB
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(init_db())