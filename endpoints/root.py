"""
Root endpoint module

This module:
1. Serves the main application HTML
2. Handles static file serving
3. Provides the favicon and other basic web assets
"""

import os
from fastapi import HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app_init import app

# Define the directory where static and images files are located
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
ASSETS_DIR = os.path.join(STATIC_DIR, "assets")

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Mount static directories
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Specific endpoints for individual files
@app.get("/runtime-config.js")
async def serve_runtime_config():
    """Serve runtime-config.js from root URL"""
    runtime_config_path = os.path.join(STATIC_DIR, "runtime-config.js")
    if os.path.exists(runtime_config_path):
        return FileResponse(runtime_config_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="runtime-config.js not found")

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon.ico"""
    try:
        favicon_path = os.path.join(STATIC_DIR, "favicon.ico")
        if os.path.exists(favicon_path):
            return FileResponse(favicon_path)
        # Return a 204 No Content response if favicon doesn't exist
        return Response(status_code=204)
    except Exception as e:
        print(f"Error serving favicon: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving favicon: {str(e)}")

# Root endpoint to serve the React app
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the React application at the root endpoint
    
    Returns:
        HTML content from index.html
    """
    try:
        # Return the index.html file
        html_path = os.path.join(STATIC_DIR, "index.html")
        
        # If the file exists, return it
        if os.path.exists(html_path):
            return FileResponse(html_path)
        
        # Otherwise, return a fallback HTML
        fallback_html = """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>React App</title>
            <link rel="stylesheet" href="/assets/index-Deoe4bZ1.css">
          </head>
          <body>
            <div id="root"></div>
            <script>
              document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('root').innerHTML = '<div style="padding: 20px; text-align: center;"><h1>React App Setup Required</h1><p>Please ensure your React application files are copied to the static directory.</p></div>';
              });
            </script>
          </body>
        </html>
        """
        return HTMLResponse(content=fallback_html)
    except Exception as e:
        print(f"Error serving root endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving root page: {str(e)}")