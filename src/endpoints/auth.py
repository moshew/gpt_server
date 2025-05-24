"""
Microsoft Authentication endpoints module

This module:
1. Implements OAuth2 flow with Microsoft Identity
2. Handles login and callback endpoints
3. Creates/updates user records based on Microsoft accounts
4. Manages session tokens for authenticated users
"""

import datetime
import json
import base64
import urllib.parse
from fastapi import Request, Cookie
from fastapi.responses import RedirectResponse
from typing import Optional

from ..app_init import app
from ..database import SessionLocal
from ..auth import get_authorization_url, exchange_code_for_token, get_or_create_user, create_access_token, get_current_user

# Login endpoint to initiate Microsoft authentication
@app.get("/auth/microsoft/login")
async def login_microsoft(request: Request, redirect_uri: Optional[str] = None):
    """
    Initiate Microsoft OAuth2 login flow
    
    Args:
        request: The incoming request
        redirect_uri: Where to redirect after successful login
        
    Returns:
        Redirect to Microsoft login page
    """
    # Use provided redirect_uri or get the referer header
    redirect_to = redirect_uri
    if not redirect_to:
        redirect_to = request.headers.get("referer", "/")
    
    # Encode the redirect URI in the state parameter
    state_data = {"redirect_uri": redirect_to}
    encoded_state = base64.urlsafe_b64encode(json.dumps(state_data).encode()).decode()
    
    # Get the authorization URL
    auth_url = await get_authorization_url()
    
    # Create response with the state cookie
    response = RedirectResponse(url=auth_url)
    response.set_cookie(
        key="auth_state",
        value=encoded_state,
        httponly=True,
        max_age=600,  # 10 minutes
        path="/",
        samesite="lax",
        secure=False  # Set to True in production with HTTPS
    )
    
    return response

# Callback endpoint for Microsoft authentication
@app.get("/auth/microsoft/callback")
async def auth_callback(
    request: Request,
    code: str, 
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    auth_state: Optional[str] = Cookie(None)
):
    """
    Handle the callback from Microsoft OAuth2
    
    Args:
        request: The incoming request
        code: Authorization code from Microsoft
        state: State parameter (from OAuth provider)
        error: Error code if authentication failed
        error_description: Error description if authentication failed
        auth_state: Cookie containing our encoded state data
        
    Returns:
        Redirect to frontend with token in cookie
    """
    try:
        # Handle authentication error
        if error:
            print(f"Authentication error: {error} - {error_description}")
            return RedirectResponse(url=f"/login?error={error}")
        
        # Default redirect location
        redirect_uri = "/"
        
        # Try to get redirect URI from cookie
        if auth_state:
            try:
                state_data = json.loads(base64.urlsafe_b64decode(auth_state).decode())
                redirect_uri = state_data.get("redirect_uri", "/")
            except Exception as e:
                print(f"Error decoding state from cookie: {e}")
        
        # Exchange code for tokens
        token_response = await exchange_code_for_token(code)
        
        # Get user info from token response
        user_info = token_response.get("user_info", {})
        email = user_info.get("mail") or user_info.get("userPrincipalName")
        
        if not email:
            return RedirectResponse(url="/login?error=email_missing")
        
        # Create or get user
        user = await get_or_create_user(email)
        
        # Create application JWT token
        token_data = {
            "email": email,
        }
        app_token = create_access_token(token_data)
        print(f"app_token: {app_token}")
        
        # Add query parameter to the redirect URI
        parsed_uri = urllib.parse.urlparse(redirect_uri)
        query_params = urllib.parse.parse_qs(parsed_uri.query)
        query_params["token"] = [app_token]
        
        # Reconstruct the URI with the key parameter
        updated_query = urllib.parse.urlencode(query_params, doseq=True)
        redirect_path = parsed_uri._replace(query=updated_query).geturl()
        
        return RedirectResponse(url=redirect_path)
        
    except Exception as e:
        print(f"Error in callback: {e}")
        return RedirectResponse(url=f"/login?error=server_error")

# Get user information endpoint
@app.get("/user/")
async def get_user(request: Request):
    """
    Get user information
       
    Returns:
        User information
    """
    async with SessionLocal() as db:
        user = await get_current_user(request, db)
    return {"username": user.username, "id": user.id}