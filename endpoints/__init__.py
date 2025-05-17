"""
Endpoints package

This package contains all the FastAPI endpoint modules.
Importing this package registers all endpoints with the FastAPI app.
"""

# Import all endpoint modules to register routes with the app
from . import chat
from . import query
from . import auth
from . import code
from . import files
from . import root
