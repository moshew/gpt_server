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

from ..app_init import app

# Define the directory where static and images files are located
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")
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
            <title>Loading Error - ElbitGPT</title>
            <link rel="stylesheet" href="/assets/index-BZCThWaO.css">
            <style>
              body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
              }
              .error-container {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                text-align: center;
                max-width: 500px;
                margin: 20px;
              }
              .error-icon {
                font-size: 3rem;
                margin-bottom: 1rem;
              }
              h1 {
                color: #e74c3c;
                margin-bottom: 1rem;
                font-size: 1.5rem;
              }
              p {
                color: #666;
                line-height: 1.5;
                margin-bottom: 1rem;
              }
              .reload-btn {
                background: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 1rem;
                margin-top: 1rem;
                transition: background 0.3s;
              }
              .reload-btn:hover {
                background: #2980b9;
              }
            </style>
          </head>
          <body>
            <div class="error-container">
              <div class="error-icon">⚠️</div>
              <h1>Website Failed to Load</h1>
              <p>We couldn't load the main website files.</p>
              <p>There might be an issue with React files or server configuration.</p>
              <p><strong>Possible solutions:</strong></p>
              <ul style="text-align: left; margin: 1rem 0;">
                <li>Ensure React files are in the static directory</li>
                <li>Check that the server is running properly</li>
                <li>Try refreshing the page</li>
              </ul>
              <button class="reload-btn" onclick="window.location.reload()">Reload Page</button>
            </div>
          </body>
        </html>
        """
        return HTMLResponse(content=fallback_html)
    except Exception as e:
        print(f"Error serving root endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving root page: {str(e)}")