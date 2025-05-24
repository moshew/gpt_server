"""
Microsoft Authentication Module

This module:
1. Implements Microsoft OAuth2 authentication flow
2. Handles token exchange and validation
3. Integrates with the existing user system
"""

import os
import sys
# Add parent directory to path for config access
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Optional, Dict, Any, List
import json
from datetime import datetime, timedelta
from urllib.parse import urlencode

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from jose import jwt, JWTError

from .database import User, Chat, SessionLocal
from config.config import MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT_ID, MS_REDIRECT_URI, SECRET_KEY
from functools import wraps

# Microsoft OAuth2 URLs
MS_AUTH_URL = f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/authorize"
MS_TOKEN_URL = f"https://login.microsoftonline.com/{MS_TENANT_ID}/oauth2/v2.0/token"
MS_GRAPH_URL = "https://graph.microsoft.com/v1.0/me"

# Token expiry time (365 * 24 hours)
ACCESS_TOKEN_EXPIRE_MINUTES = 525600

# Cache for Microsoft tokens to reduce API calls
ms_token_cache = {}

async def get_authorization_url() -> str:
    """
    Generate the Microsoft authorization URL
    
    Returns:
        URL to redirect the user to for Microsoft login
    """
    params = {
        "client_id": MS_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": MS_REDIRECT_URI,
        "response_mode": "query",
        "scope": "openid profile email User.Read",
        "state": "state_for_csrf_protection"  # In production, this should be a randomly generated value
    }
    return f"{MS_AUTH_URL}?{urlencode(params)}"

async def exchange_code_for_token(code: str) -> Dict[str, Any]:
    """
    Exchange authorization code for access token
    
    Args:
        code: Authorization code from Microsoft
        
    Returns:
        Token response with access token and user information
    """
    data = {
        "client_id": MS_CLIENT_ID,
        "scope": "openid profile email User.Read",
        "code": code,
        "redirect_uri": MS_REDIRECT_URI,
        "grant_type": "authorization_code",
        "client_secret": MS_CLIENT_SECRET
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(MS_TOKEN_URL, data=data)
            response.raise_for_status()
            token_data = response.json()
            
            # Get user information from Microsoft Graph
            user_info = await get_user_info(token_data["access_token"])
            
            # Cache the token
            if "email" in user_info:
                ms_token_cache[user_info["email"]] = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_at": datetime.now() + timedelta(seconds=token_data["expires_in"])
                }
            
            return {
                "access_token": token_data["access_token"],
                "id_token": token_data.get("id_token"),
                "user_info": user_info
            }
    except httpx.HTTPStatusError as e:
        print(f"Error exchanging code for token: {e.response.text}")
        raise HTTPException(status_code=400, detail=f"Failed to exchange code: {str(e)}")
    except Exception as e:
        print(f"Error in token exchange: {e}")
        raise HTTPException(status_code=500, detail=f"Token exchange error: {str(e)}")

async def get_user_info(access_token: str) -> Dict[str, Any]:
    """
    Get user information from Microsoft Graph API
    
    Args:
        access_token: Microsoft access token
        
    Returns:
        User profile information from Microsoft
    """
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = await client.get(MS_GRAPH_URL, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"Error getting user info: {e.response.text}")
        raise HTTPException(status_code=401, detail="Invalid Microsoft token")
    except Exception as e:
        print(f"Error in get_user_info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting user information: {str(e)}")

async def refresh_microsoft_token(refresh_token: str) -> Dict[str, Any]:
    """
    Refresh Microsoft access token
    
    Args:
        refresh_token: Microsoft refresh token
        
    Returns:
        New token information
    """
    data = {
        "client_id": MS_CLIENT_ID,
        "scope": "openid profile email User.Read",
        "refresh_token": refresh_token,
        "redirect_uri": MS_REDIRECT_URI,
        "grant_type": "refresh_token",
        "client_secret": MS_CLIENT_SECRET
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(MS_TOKEN_URL, data=data)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error refreshing Microsoft token: {e}")
        raise HTTPException(status_code=401, detail="Failed to refresh token")

def create_access_token(data: Dict[str, Any]) -> str:
    """
    Create an access token for the application
    
    Args:
        data: Data to encode in the token
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    expiry = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expiry})
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

async def get_user_from_token(token: str, db: AsyncSession) -> User:
    """
    Get user from token
    
    Args:
        token: JWT token
        db: Database session
        
    Returns:
        User object
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload.get("email")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Find user by Microsoft email
    result = await db.execute(select(User).filter(User.username == email))
    user = result.scalars().first()
    
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

async def get_current_user(request: Request, db: AsyncSession) -> User:
    """
    Get current user from session or token
    
    Args:
        request: FastAPI request
        db: Database session
        
    Returns:
        User object
    """
    # Try to get token from Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ")[1]
    else:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return await get_user_from_token(token, db)


async def get_or_create_user(email: str) -> User:
    """
    Get existing user or create a new one based on Microsoft account
    
    Args:
        email: User's email from Microsoft
        display_name: User's display name from Microsoft
        db: Database session
        
    Returns:
        User object
    """
    async with SessionLocal() as db:
        # Check if user already exists by Microsoft email
        result = await db.execute(select(User).filter(User.username == email))
        user = result.scalars().first()
        
        if user:
            return user
        
        # Create new user
        new_user = User(
            username=email
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
    
    return new_user

# Chat ownership verification function
async def verify_chat_ownership(chat_id: int, user_id: int, db: AsyncSession) -> Chat:
    """
    Verify that a chat belongs to the authenticated user
    
    Args:
        chat_id: ID of the chat to check
        user_id: ID of the authenticated user
        db: Database session
        
    Returns:
        Chat object if it belongs to the user
        
    Raises:
        HTTPException: If chat doesn't exist or doesn't belong to the user
    """
    try:
        # Query for the chat with both chat_id and user_id constraints
        result = await db.execute(
            select(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id)
        )
        chat = result.scalars().first()
        
        if not chat:
            raise HTTPException(
                status_code=404, 
                detail="Chat not found or access denied"
            )
        
        return chat
    except HTTPException:
        # Re-raise HTTPExceptions as they're already formatted correctly
        raise
    except Exception as e:
        print(f"Error verifying chat ownership: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error while verifying chat ownership: {str(e)}"
        )


#chat ownership verification
def verify_chat_owner():
    async def dependency(
        chat_id: int,
        request: Request,
    ):
        async with SessionLocal() as db:
            user = await get_current_user(request, db)
        await verify_chat_ownership(chat_id, user.id, db)
        return user  # Return the user for convenience
        
    return dependency
