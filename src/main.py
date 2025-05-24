"""
Main application module for the Chat System with Code Analysis capabilities

This is the entry point to the application that imports all the modules
and starts the FastAPI server.
"""

# Import the core application components
from .app_init import app

# Import database initialization
from . import db_manager

# Import all endpoints via the endpoints package
from . import endpoints

# This is the entry point when running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)