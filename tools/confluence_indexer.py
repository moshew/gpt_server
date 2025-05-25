#!/usr/bin/env python3
"""
Confluence Indexer with SSL fixes and DocumentRAG integration

This application indexes Confluence spaces and pages, processes the content,
and creates a searchable index for later retrieval. It includes fixes for
SSL certificate verification issues and integration with DocumentRAG for
consistent vector search capabilities.

IMPORTANT: This version enforces full-page indexing, preventing any chunking of documents.
"""

import asyncio
import logging
import os
import json
from typing import List, Dict, Any, Optional
import re
import aiohttp
from datetime import datetime
import time
import configparser
from pathlib import Path
import certifi
import urllib3
import ssl
import numpy as np
import pickle
import faiss

# Import our utility modules
from utils import (
    run_in_executor,
    run_tasks_with_limit,
    parse_json_string,
    serialize_to_json,
    embed_documents
)

# Import DocumentRAG for index creation
from query_docs import DocumentRAG

# Import our custom no-chunk wrapper
# Make sure to place the no_chunk_rag.py file in the same directory
from no_chunk_rag import NoChunkDocumentRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("confluence-indexer")

# Apply SSL certificate fixes
def apply_ssl_fixes():
    """Apply fixes for SSL certificate verification issues"""
    # Set environment variable to use certifi's certificate bundle
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
    
    # Create a custom SSL context using certifi's certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # Disable SSL warnings during testing
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    logger.info(f"Applied SSL fixes using certifi: {certifi.where()}")

# Default configuration
DEFAULT_CONFIG = {
    "confluence": {
        "base_url": "http://confluence.sys.ecix/",
        "username": "cix_user",
        "api_token": "Elbit-1",
        "space_keys": "DEV",
    },
    "indexer": {
        "output_dir": "confluence_index",
        "chunk_size": "1000",  # Not used anymore - always full page indexing
        "chunk_overlap": "200", # Not used anymore - always full page indexing
        "concurrent_requests": "5",
        "knowledge_base_dir": "knowledgebase"  # Directory for knowledge bases
    }
}

class ConfluenceIndexer:
    """Main class for indexing Confluence content with full-page indexing"""
    
    def __init__(
        self, 
        base_url: str = None, 
        username: str = None, 
        api_token: str = None,
        output_dir: str = None,
        concurrent_requests: int = None,
        knowledge_base_dir: str = None,
        kb_name: str = None,
        config_file: str = "config.ini"
    ):
        """
        Initialize the Confluence indexer
        
        Args:
            base_url: Base URL of the Confluence instance
            username: Confluence username (email)
            api_token: Confluence API token
            output_dir: Directory to store the index
            concurrent_requests: Number of concurrent requests to Confluence API
            knowledge_base_dir: Directory for storing knowledge bases
            kb_name: Name of the knowledge base to create
            config_file: Path to config file (default: "config.ini")
        """
        # Load config from file if it exists
        config = self.load_config(config_file)
        
        # Set parameters, with priority: args > config file > environment > defaults
        self.base_url = (base_url or 
                        config["confluence"].get("base_url") or 
                        os.environ.get("CONFLUENCE_URL") or 
                        DEFAULT_CONFIG["confluence"]["base_url"]).rstrip('/')
                        
        self.username = (username or 
                        config["confluence"].get("username") or 
                        os.environ.get("CONFLUENCE_USERNAME") or 
                        DEFAULT_CONFIG["confluence"]["username"])
                        
        self.api_token = (api_token or 
                        config["confluence"].get("api_token") or 
                        os.environ.get("CONFLUENCE_API_TOKEN") or 
                        DEFAULT_CONFIG["confluence"]["api_token"])
                        
        self.output_dir = (output_dir or 
                        config["indexer"].get("output_dir") or 
                        os.environ.get("CONFLUENCE_OUTPUT_DIR") or 
                        DEFAULT_CONFIG["indexer"]["output_dir"])
                                
        self.concurrent_requests = int(concurrent_requests or 
                                    config["indexer"].get("concurrent_requests") or 
                                    os.environ.get("CONFLUENCE_CONCURRENT_REQUESTS") or 
                                    DEFAULT_CONFIG["indexer"]["concurrent_requests"])
        
        # Knowledge base directory
        self.knowledge_base_dir = (knowledge_base_dir or
                                  config["indexer"].get("knowledge_base_dir") or
                                  os.environ.get("CONFLUENCE_KB_DIR") or
                                  DEFAULT_CONFIG["indexer"]["knowledge_base_dir"])
        
        # Knowledge base name (if provided)
        self.kb_name = kb_name
                                    
        # ALWAYS use full page indexing - no option to change this
        self.full_page_indexing = True
        
        # Get space keys from config if not provided
        self.space_keys = None
        space_keys_str = (config["confluence"].get("space_keys") or 
                        os.environ.get("CONFLUENCE_SPACE_KEYS") or 
                        DEFAULT_CONFIG["confluence"]["space_keys"])
        if space_keys_str:
            self.space_keys = [s.strip() for s in space_keys_str.split(',') if s.strip()]
        
        # Auth tuple for basic auth
        self.auth = aiohttp.BasicAuth(self.username, self.api_token) if self.username and self.api_token else None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "spaces"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "pages"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "chunks"), exist_ok=True)
        
        # Ensure knowledge base directory exists
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # Create knowledge base directory if kb_name is provided
        if self.kb_name:
            os.makedirs(os.path.join(self.knowledge_base_dir, self.kb_name), exist_ok=True)
        
        # Create NoChunkDocumentRAG instance for index creation (instead of regular DocumentRAG)
        if self.kb_name:
            # NoChunkDocumentRAG will use the knowledge base directory instead of the standard docs_dir/chats
            kb_path = os.path.join(self.knowledge_base_dir, self.kb_name)
            self.document_rag = NoChunkDocumentRAG(docs_dir=kb_path, rag_storage_dir=kb_path)
        else:
            # Use default output directory for NoChunkDocumentRAG
            self.document_rag = NoChunkDocumentRAG(docs_dir=self.output_dir, rag_storage_dir=self.output_dir)
        
        # Stats for reporting
        self.stats = {
            "spaces_processed": 0,
            "pages_processed": 0,
            "chunks_created": 0,
            "embeddings_created": 0,
            "errors": 0,
            "start_time": time.time(),
        }
        
        # Log configuration
        logger.info(f"Confluence Indexer initialized with:")
        logger.info(f"- Base URL: {self.base_url}")
        logger.info(f"- Username: {self.username}")
        logger.info(f"- Output directory: {self.output_dir}")
        logger.info(f"- Knowledge base directory: {self.knowledge_base_dir}")
        logger.info(f"- Knowledge base name: {self.kb_name or 'Not specified'}")
        logger.info(f"- Concurrent requests: {self.concurrent_requests}")
        logger.info(f"- Always using FULL PAGE indexing - no chunking")
        if self.space_keys:
            logger.info(f"- Space keys: {', '.join(self.space_keys)}")
        
    def load_config(self, config_file: str) -> Dict[str, Dict[str, str]]:
        """
        Load configuration from file
        
        Args:
            config_file: Path to config file
            
        Returns:
            Configuration dictionary
        """
        # Create default config structure
        config = {
            "confluence": {},
            "indexer": {}
        }
        
        # Try to load from file
        if os.path.exists(config_file):
            try:
                parser = configparser.ConfigParser()
                parser.read(config_file)
                
                # Copy sections to our config dict
                for section in parser.sections():
                    if section not in config:
                        config[section] = {}
                    for key, value in parser.items(section):
                        config[section][key] = value
                
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
        else:
            # Create default config file
            try:
                self.save_default_config(config_file)
                logger.info(f"Created default configuration file at {config_file}")
            except Exception as e:
                logger.error(f"Error creating default config file: {str(e)}")
        
        return config
        
    def save_default_config(self, config_file: str) -> None:
        """
        Save default configuration to file
        
        Args:
            config_file: Path to config file
        """
        parser = configparser.ConfigParser()
        
        # Add sections and options
        for section, options in DEFAULT_CONFIG.items():
            parser.add_section(section)
            for key, value in options.items():
                parser[section][key] = value
        
        # Write to file
        with open(config_file, 'w') as f:
            parser.write(f)
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the Confluence API
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            
        Returns:
            Response JSON
        """
        url = f"{self.base_url}/rest/api{endpoint}"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Create a connector with custom SSL context for better certificate handling
        # Limit the maximum number of connections to avoid "Too many open files" errors
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=self.concurrent_requests, limit_per_host=self.concurrent_requests)
        
        # Use a timeout to prevent hanging connections
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=10)
        
        # Add retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    async with session.get(url, auth=self.auth, headers=headers, params=params) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            error_msg = f"Error {response.status} for {url}: {error_text}"
                            logger.error(error_msg)
                            self.stats["errors"] += 1
                            return {"error": error_msg}
                        
                        return await response.json()
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"Request timed out for {url}, attempt {attempt+1}/{max_retries}, retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    error_msg = f"Request timed out for {url} after {max_retries} attempts"
                    logger.error(error_msg)
                    self.stats["errors"] += 1
                    return {"error": error_msg}
            except Exception as e:
                error_msg = f"Request failed for {url}: {str(e)}"
                
                # Check if this is a "Too many open files" error
                if "Too many open files" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"Too many open files for {url}, attempt {attempt+1}/{max_retries}, retrying in {retry_delay}s")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                
                logger.error(error_msg)
                self.stats["errors"] += 1
                return {"error": error_msg}
    
    async def get_all_spaces(self) -> List[Dict]:
        """
        Get all spaces from Confluence
        
        Returns:
            List of space objects
        """
        logger.info("Fetching all Confluence spaces...")
        spaces = []
        start = 0
        limit = 25
        
        while True:
            params = {
                "start": start,
                "limit": limit,
                "type": "global"  # Only get global spaces
            }
            
            response = await self._make_request("/space", params)
            
            if "error" in response:
                return spaces
            
            if "results" in response and response["results"]:
                spaces.extend(response["results"])
                
                # Check if we've reached the end
                if len(response["results"]) < limit:
                    break
                
                start += limit
            else:
                break
        
        logger.info(f"Found {len(spaces)} spaces")
        return spaces
    
    async def get_all_pages_in_space(self, space_key: str) -> List[Dict]:
        """
        Get all pages in a specific space
        
        Args:
            space_key: Space key
            
        Returns:
            List of page objects
        """
        logger.info(f"Fetching all pages in space {space_key}...")
        pages = []
        start = 0
        limit = 25
        
        while True:
            params = {
                "spaceKey": space_key,
                "start": start,
                "limit": limit,
                "expand": "version"
            }
            
            response = await self._make_request("/content", params)
            
            if "error" in response:
                return pages
            
            if "results" in response and response["results"]:
                pages.extend(response["results"])
                
                # Check if we've reached the end
                if len(response["results"]) < limit:
                    break
                
                start += limit
            else:
                break
        
        logger.info(f"Found {len(pages)} pages in space {space_key}")
        return pages
    
    async def get_page_content(self, page_id: str) -> Dict:
        """
        Get content of a specific page
        
        Args:
            page_id: Page ID
            
        Returns:
            Page content object
        """
        logger.debug(f"Fetching content for page {page_id}...")
        
        params = {
            "expand": "body.storage,version,space",
        }
        
        response = await self._make_request(f"/content/{page_id}", params)
        return response
    
    async def process_space(self, space: Dict) -> None:
        """
        Process a single space
        
        Args:
            space: Space object
        """
        space_key = space["key"]
        space_name = space["name"]
        logger.info(f"Processing space: {space_name} ({space_key})")
        
        # Determine the target directory based on kb_name
        if self.kb_name:
            kb_dir = os.path.join(self.knowledge_base_dir, self.kb_name)
            target_dir = os.path.join(kb_dir, "spaces")
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_dir = os.path.join(self.output_dir, "spaces")
        
        # Save space metadata
        space_metadata = {
            "id": space["id"],
            "key": space_key,
            "name": space_name,
            "description": space.get("description", {}).get("plain", {}).get("value", ""),
            "type": space.get("type", ""),
            "indexed_at": datetime.now().isoformat(),
        }
        
        space_file = os.path.join(target_dir, f"{space_key}.json")
        with open(space_file, "w", encoding="utf-8") as f:
            json.dump(space_metadata, f, ensure_ascii=False, indent=2)
        
        # Get all pages in the space
        pages = await self.get_all_pages_in_space(space_key)
        
        # Process pages in batches to avoid "Too many open files" error
        batch_size = min(self.concurrent_requests, 5)  # Limit concurrent connections
        logger.info(f"Processing {len(pages)} pages in batches of {batch_size}")
        
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i+batch_size]
            page_tasks = [self.process_page(page, space_key) for page in batch]
            await asyncio.gather(*page_tasks)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(pages) + batch_size - 1)//batch_size} ({i+len(batch)}/{len(pages)} pages)")
        
        self.stats["spaces_processed"] += 1
        logger.info(f"Completed processing space: {space_name} ({space_key})")
    
    async def process_page(self, page: Dict, space_key: str) -> None:
        """
        Process a single page
        
        Args:
            page: Page object
            space_key: Space key
        """
        page_id = page["id"]
        
        # Get full page content
        page_content = await self.get_page_content(page_id)
        
        if "error" in page_content:
            logger.error(f"Error getting content for page {page_id}")
            self.stats["errors"] += 1
            return
        
        # Extract content from page
        title = page_content.get("title", "")
        version = page_content.get("version", {}).get("number", 0)
        body = page_content.get("body", {}).get("storage", {}).get("value", "")
        
        # Determine the target directory based on kb_name
        if self.kb_name:
            kb_dir = os.path.join(self.knowledge_base_dir, self.kb_name)
            pages_dir = os.path.join(kb_dir, "pages")
            chunks_dir = os.path.join(kb_dir, "chunks")
            os.makedirs(pages_dir, exist_ok=True)
            os.makedirs(chunks_dir, exist_ok=True)
        else:
            pages_dir = os.path.join(self.output_dir, "pages")
            chunks_dir = os.path.join(self.output_dir, "chunks")
        
        # Save page metadata and content
        page_metadata = {
            "id": page_id,
            "space_key": space_key,
            "title": title,
            "version": version,
            "created_at": page_content.get("created", ""),
            "updated_at": page_content.get("version", {}).get("when", ""),
            "indexed_at": datetime.now().isoformat(),
        }
        
        # Save page metadata
        page_file = os.path.join(pages_dir, f"{page_id}.json")
        with open(page_file, "w", encoding="utf-8") as f:
            json.dump(page_metadata, f, ensure_ascii=False, indent=2)
        
        # Save full page content
        content_file = os.path.join(pages_dir, f"{page_id}_content.html")
        with open(content_file, "w", encoding="utf-8") as f:
            f.write(body)
        
        # Clean content
        clean_content = self._clean_html(body)
        
        chunk_metadata = []
        
        # Always use full page indexing - NEVER do chunking
        chunk_id = f"{page_id}_full"
        
        # Create chunk metadata
        chunk_data = {
            "id": chunk_id,
            "page_id": page_id,
            "space_key": space_key,
            "chunk_index": 0,
            "page_title": title,
            "content": clean_content,
            "token_count": len(clean_content.split()),  # Simple token count estimation
            "is_full_page": True
        }
        
        # Save chunk data
        chunk_file = os.path.join(chunks_dir, f"{chunk_id}.json")
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        chunk_metadata.append({
            "id": chunk_id,
            "page_id": page_id,
            "space_key": space_key,
            "chunk_index": 0,
            "token_count": len(clean_content.split()),
            "is_full_page": True
        })
        
        self.stats["chunks_created"] += 1
        
        # Update page metadata with chunks info
        page_metadata["chunks"] = chunk_metadata
        page_metadata["full_page_indexing"] = True  # Always true
        
        with open(page_file, "w", encoding="utf-8") as f:
            json.dump(page_metadata, f, ensure_ascii=False, indent=2)
        
        self.stats["pages_processed"] += 1
        logger.info(f"Processed page: {title} (ID: {page_id})")
    
    def _clean_html(self, html_content: str) -> str:
        """
        Clean HTML content by removing tags and normalizing whitespace
        
        Args:
            html_content: HTML content string
            
        Returns:
            Clean text content
        """
        # Simple HTML tag removal
        # For production, consider using a proper HTML parser like BeautifulSoup
        tag_pattern = re.compile(r'<[^>]+>')
        clean_text = tag_pattern.sub(' ', html_content)
        
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    async def create_embeddings_with_document_rag(self) -> None:
        """Create embeddings using our NoChunkDocumentRAG wrapper"""
        logger.info("Creating embeddings using NoChunkDocumentRAG (preventing chunking)...")
        
        # Determine the target directory
        if self.kb_name:
            kb_dir = os.path.join(self.knowledge_base_dir, self.kb_name)
            chunks_dir = os.path.join(kb_dir, "chunks")
        else:
            chunks_dir = os.path.join(self.output_dir, "chunks")
        
        # Check if chunks directory exists and has files
        if not os.path.exists(chunks_dir):
            logger.warning(f"Chunks directory {chunks_dir} does not exist")
            return
            
        # List all chunk files
        chunk_files = [
            os.path.join(chunks_dir, f)
            for f in os.listdir(chunks_dir)
            if f.endswith(".json") and not f.endswith("_embedding.json")
        ]
        
        if not chunk_files:
            logger.warning(f"No chunk files found in {chunks_dir}")
            return
            
        logger.info(f"Found {len(chunk_files)} chunk files to process (full pages)")
        
        # Create a temporary directory to store text files for DocumentRAG
        temp_dir = os.path.join(self.output_dir, "temp_for_rag")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create the chat_confluence subfolder to match DocumentRAG's expected structure
        # DocumentRAG expects files to be in: docs_dir/chat_<chat_id>/
        chat_dir = os.path.join(temp_dir, "chat_confluence")
        os.makedirs(chat_dir, exist_ok=True)
        
        # Create text files from chunks for DocumentRAG to process
        # Save them in the chat_confluence subfolder
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                
                # Create a text file with the chunk content
                chunk_id = chunk_data["id"]
                # Save to the chat_confluence subfolder
                with open(os.path.join(chat_dir, f"{chunk_id}.txt"), "w", encoding="utf-8") as f:
                    # Include metadata in the text to provide context
                    f.write(f"Title: {chunk_data['page_title']}\n")
                    f.write(f"Space: {chunk_data['space_key']}\n\n")
                    f.write(chunk_data["content"])
            except Exception as e:
                logger.error(f"Error processing chunk file {chunk_file}: {e}")
        
        # Use NoChunkDocumentRAG to create embeddings
        try:
            # Determine the target directory for storing the index
            target_dir = os.path.join(self.knowledge_base_dir, self.kb_name) if self.kb_name else self.output_dir
            
            # Configure NoChunkDocumentRAG to use the temp directory as the base directory for documents
            logger.info(f"Creating NoChunkDocumentRAG with docs_dir={temp_dir}, rag_storage_dir={target_dir}")
            document_rag = NoChunkDocumentRAG(docs_dir=temp_dir, rag_storage_dir=target_dir)
            
            # Use NoChunkDocumentRAG's indexing function to prevent any further chunking
            # Note that chat_id="confluence" will look for files in docs_dir/chat_confluence/
            result = await document_rag.index_documents(
                chat_id="confluence"  # Using "confluence" as a placeholder chat_id
            )
            
            # Update stats
            if "chunks" in result:
                self.stats["embeddings_created"] = result["chunks"]
            
            logger.info(f"NoChunkDocumentRAG indexing result: {result}")
            
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
                
        except Exception as e:
            logger.error(f"Error creating embeddings with NoChunkDocumentRAG: {e}")
            self.stats["errors"] += 1
    
    async def create_index(self) -> None:
        """Create a master index of all content"""
        logger.info("Creating master index...")
        
        # Determine the target directory
        if self.kb_name:
            target_dir = os.path.join(self.knowledge_base_dir, self.kb_name)
            spaces_dir = os.path.join(target_dir, "spaces")
            pages_dir = os.path.join(target_dir, "pages")
        else:
            target_dir = self.output_dir
            spaces_dir = os.path.join(self.output_dir, "spaces")
            pages_dir = os.path.join(self.output_dir, "pages")
        
        # Load all space metadata
        spaces = {}
        for file_name in os.listdir(spaces_dir):
            if file_name.endswith(".json"):
                with open(os.path.join(spaces_dir, file_name), "r", encoding="utf-8") as f:
                    space_data = json.load(f)
                    spaces[space_data["key"]] = space_data
        
        # Load all page metadata
        pages = {}
        for file_name in os.listdir(pages_dir):
            if file_name.endswith(".json") and not file_name.endswith("_content.json"):
                try:
                    with open(os.path.join(pages_dir, file_name), "r", encoding="utf-8") as f:
                        page_data = json.load(f)
                        pages[page_data["id"]] = page_data
                except Exception as e:
                    logger.error(f"Error reading page file {file_name}: {e}")
        
        # Create master index
        index = {
            "spaces": spaces,
            "pages": pages,
            "stats": self.stats,
            "created_at": datetime.now().isoformat(),
            "kb_name": self.kb_name,
            "full_page_indexing": True  # Always true
        }
        
        # Save index
        index_file = os.path.join(target_dir, "index.json")
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Master index created at {index_file}")

    async def run(self, space_keys: Optional[List[str]] = None) -> Dict:
        """
        Run the indexing process
        
        Args:
            space_keys: Optional list of space keys to process (None for all)
            
        Returns:
            Statistics about the indexing process
        """
        # Apply SSL fixes before running
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi.where()
        logger.info(f"Applied SSL fixes using certifi: {certifi.where()}")
        
        logger.info("Starting Confluence indexing process with FULL PAGE indexing...")
        
        # Use space keys from init if not provided here
        if space_keys is None:
            space_keys = self.space_keys
            
        # Get all spaces
        all_spaces = await self.get_all_spaces()
        
        # Filter spaces if space_keys is provided
        if space_keys:
            spaces = [s for s in all_spaces if s["key"] in space_keys]
            logger.info(f"Filtered to {len(spaces)} spaces")
        else:
            spaces = all_spaces
        
        # Process each space sequentially to avoid concurrency issues
        for space in spaces:
            await self.process_space(space)
        
        # Create embeddings using NoChunkDocumentRAG
        #await self.create_embeddings_with_document_rag()
        
        # Create master index
        await self.create_index()
        
        # Update final stats
        self.stats["end_time"] = time.time()
        self.stats["total_duration"] = self.stats["end_time"] - self.stats["start_time"]
        
        logger.info(f"Indexing complete! Processed {self.stats['spaces_processed']} spaces, "
                   f"{self.stats['pages_processed']} pages, created {self.stats['chunks_created']} full-page chunks "
                   f"and {self.stats['embeddings_created']} embeddings in "
                   f"{self.stats['total_duration']:.2f} seconds")
        
        return self.stats

if __name__ == "__main__":
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Index Confluence content for RAG")
    parser.add_argument("--base-url", help="Confluence base URL")
    parser.add_argument("--username", help="Confluence username")
    parser.add_argument("--api-token", help="Confluence API token")
    parser.add_argument("--output-dir", help="Output directory for index")
    parser.add_argument("--kb-name", help="Knowledge base name")
    parser.add_argument("--concurrent-requests", type=int, help="Number of concurrent requests")
    parser.add_argument("--space-keys", help="Comma-separated list of space keys to index")
    parser.add_argument("--config-file", default="config.ini", help="Path to config file")
    # Removed full-page flag since it's always on
    
    args = parser.parse_args()
    
    # Get space keys from command line
    space_keys = None
    if args.space_keys:
        space_keys = [s.strip() for s in args.space_keys.split(",") if s.strip()]
    
    # Create indexer
    indexer = ConfluenceIndexer(
        base_url=args.base_url,
        username=args.username,
        api_token=args.api_token,
        output_dir=args.output_dir,
        concurrent_requests=args.concurrent_requests,
        kb_name=args.kb_name,
        config_file=args.config_file
    )
    
    # Run indexer
    try:
        import asyncio
        asyncio.run(indexer.run(space_keys=space_keys))
    except KeyboardInterrupt:
        print("\nIndexing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during indexing: {e}")
        sys.exit(1)