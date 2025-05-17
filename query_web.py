"""
Web Search module

This module:
1. Handles web search for up-to-date information
2. Extracts content from web pages
3. Formats content for use with RAG-enhanced queries
4. Interfaces with Google search through Serper API
5. Enhances search queries using conversation history
"""

import time
import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
import aiohttp
from bs4 import BeautifulSoup
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import Message
from config import SERPER_API_KEY
from utils import call_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_search")

class WebSearchHandler:
    """
    Class to handle web search operations and content extraction
    """
    
    def __init__(self, max_results: int = 5, max_retries: int = 3):
        """
        Initialize the WebSearchHandler
        
        Args:
            max_results: Maximum number of search results to process
            max_retries: Maximum number of retry attempts for API calls
        """
        self.max_results = max_results
        self.max_retries = max_retries
        self.extract_retries = 1  # Set to 1 for content extraction - move to next URL faster
        self.serper_api_url = "https://google.serper.dev/search"
        
        # Check if API key is available
        if not SERPER_API_KEY:
            logger.warning("SERPER_API_KEY not configured in config.py. Web search will not work.")
    
    async def generate_enhanced_search_query(self, current_query: str, chat_history: List[Dict]) -> str:
        """
        Use LLM to generate an enhanced search query that incorporates context from conversation history
        
        Args:
            current_query: The current user question
            chat_history: Conversation history (list of messages)
            
        Returns:
            Enhanced search query
        """
        # If there are fewer than 2 messages, this is likely the first question
        # No need to enhance the query in this case
        if len(chat_history) < 2:
            logger.info(f"First question detected. Using original query: '{current_query}'")
            return current_query
            
        # Check if there's only one user message (current one)
        user_message_count = sum(1 for msg in chat_history if msg.get("sender") == "user")
        if user_message_count <= 1:
            logger.info(f"No prior user questions found. Using original query: '{current_query}'")
            return current_query
        
        # Load recent history (e.g., last 5 messages)
        history_context = ""
        recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
        
        for msg in recent_history:
            sender = msg.get("sender", "")
            content = msg.get("content", "")
            if sender and content:
                history_context += f"{sender}: {content}\n\n"
        
        # Instructions for the LLM to add context to the query, not expand it
        prompt = f"""
You need to modify a search query to include necessary context from a conversation.
DO NOT expand or broaden the query - ONLY add specific context details from previous messages that would help find precise information.

Conversation history:
{history_context}

Current question: {current_query}

Modified search query with context (keep it focused, clear, precise and concise, without explanations):
"""
        
        # Use LLM to generate enhanced query
        try:
            enhanced_query = await call_llm(prompt=prompt, model_config="fast")
            logger.info(f"Original query: '{current_query}' → Modified query with context: '{enhanced_query}'")
            return enhanced_query
        except Exception as e:
            logger.error(f"Error generating enhanced search query: {e}")
            return current_query  # Use original query in case of error
    
    async def search_google(self, query: str) -> List[Dict]:
        """
        Search Google using Serper API
        
        Args:
            query: Search query
            
        Returns:
            List of search result items with URL and title
        """
        if not SERPER_API_KEY:
            logger.error("SERPER_API_KEY not configured in config.py. Cannot perform web search.")
            return []
        
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "gl": "us",
            "hl": "en",
            "num": self.max_results * 2  # Request more results to account for filtering
        }
        
        for retry in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.serper_api_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    ) as response:
                        if response.status != 200:
                            logger.error(f"Error calling Serper API: {response.status}")
                            await asyncio.sleep(1 * (retry + 1))  # Exponential backoff
                            continue
                        
                        data = await response.json()
                        
                        # Extract organic search results
                        organic = data.get("organic", [])
                        
                        # Filter and format results
                        results = []
                        for item in organic:
                            if "link" in item and "title" in item:
                                results.append({
                                    "url": item["link"],
                                    "title": item["title"],
                                    "snippet": item.get("snippet", "")
                                })
                                
                                # Stop once we have enough results
                                if len(results) >= self.max_results:
                                    break
                        
                        logger.info(f"Found {len(results)} search results for query: {query}")
                        return results
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout calling Serper API (retry {retry+1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Error during search: {e} (retry {retry+1}/{self.max_retries})")
            
            # Wait before retrying
            await asyncio.sleep(1 * (retry + 1))
        
        logger.error(f"Failed to get search results after {self.max_retries} retries")
        return []
    
    async def extract_content_from_url(self, url: str) -> str:
        """
        Extract readable content from a URL using BeautifulSoup
        
        Args:
            url: Web page URL
            
        Returns:
            Extracted text content
        """
        for retry in range(self.extract_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=15) as response:
                        if response.status != 200:
                            logger.warning(f"Error fetching URL {url}: {response.status}")
                            await asyncio.sleep(0.2)  # Brief pause before giving up
                            continue
                        
                        html = await response.text()
                        
                        # Use BeautifulSoup for content extraction
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script, style, nav, and other non-content elements
                        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header']):
                            element.decompose()
                        
                        # Extract main content elements
                        main_content = ""
                        
                        # First try to find main content containers
                        for container in soup.select('main, article, .content, .main, #content, #main'):
                            if container and len(container.get_text(strip=True)) > 100:
                                main_content = container.get_text(separator='\n', strip=True)
                                break
                        
                        # If no main content found, try paragraphs
                        if not main_content:
                            paragraphs = []
                            for p in soup.find_all('p'):
                                text = p.get_text(strip=True)
                                if len(text) > 20:  # Filter out short paragraphs
                                    paragraphs.append(text)
                            
                            main_content = '\n\n'.join(paragraphs)
                        
                        # If still no content, fall back to all text
                        if not main_content:
                            main_content = soup.get_text(separator='\n', strip=True)
                            
                            # Clean up whitespace
                            lines = (line.strip() for line in main_content.splitlines() if line.strip())
                            main_content = '\n'.join(lines)
                        
                        # Limit the length of the extracted text
                        max_length = 5000  # Reasonable limit for content extraction
                        if main_content and len(main_content) > max_length:
                            main_content = main_content[:max_length] + "... (content truncated)"
                        
                        if main_content:
                            logger.info(f"Successfully extracted content from {url}")
                            return main_content
                        else:
                            logger.warning(f"No content extracted from {url}")
                            return ""
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout extracting content from {url} (retry {retry+1}/{self.extract_retries})")
            except Exception as e:
                logger.error(f"Error extracting content from {url}: {e}")
            
            # Minimal wait for extract_retries since we prefer to move on quickly
            await asyncio.sleep(0.2)
        
        logger.error(f"Failed to extract content from {url} after {self.extract_retries} retries")
        return ""
    
    async def get_web_search_context(self, query: str) -> str:
        """
        Perform web search and extract content from top results
        
        Args:
            query: Search query
            
        Returns:
            Formatted context from web search results
        """
        try:
            start_time = time.time()
            
            # Search Google
            search_results = await self.search_google(query)
            
            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return ""
            
            # Keep track of processed URLs to avoid duplicates
            processed_urls = set()
            
            # Extract content from URLs with a fallback strategy
            # Try URLs until we get enough content or run out of URLs
            content_results = []
            url_index = 0
            
            # Keep trying URLs until we get enough content or run out of URLs
            while len(content_results) < self.max_results and url_index < len(search_results):
                result = search_results[url_index]
                url = result.get("url")
                title = result.get("title", "")
                
                url_index += 1
                
                if not url or url in processed_urls:
                    continue
                
                processed_urls.add(url)
                
                # Try to extract content from this URL
                content = await self.extract_content_from_url(url)
                
                # If we got content, add it to results
                if content:
                    content_results.append({
                        "url": url,
                        "title": title,
                        "content": content
                    })
                    logger.info(f"Added content from {url} ({len(content_results)}/{self.max_results})")
                else:
                    logger.warning(f"Failed to extract content from {url}, moving to next URL")
            
            # Format the context with source information
            context_parts = []
            for i, result in enumerate(content_results):
                # Add source information
                source_info = f"Source {i+1}: {result['title']} ({result['url']})"
                context_parts.append(f"{source_info}\n{result['content']}\n")
            
            # Combine all context parts
            if context_parts:
                context = "\n\nWeb search results:\n\n" + "\n".join(context_parts)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Web search completed in {elapsed_time:.2f} seconds with {len(content_results)} successful results out of {url_index} attempts")
                
                return context
            else:
                logger.warning("No content extracted from any search results")
                return ""
                
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return ""
    
    async def get_document_context(
        self, 
        chat_id: str, 
        query: str, 
        top_k: int = 5,
        db: Optional[AsyncSession] = None,  # Add database session parameter
    ) -> str:
        """
        Get document context from web search with enhanced query from conversation history
        
        Args:
            chat_id: Chat identifier
            query: User's question
            top_k: Number of results to fetch (default 5)
            db: Database session (optional)
            
        Returns:
            Document context from web search
        """
        # Update max_results with top_k parameter
        self.max_results = top_k
        
        # If database session is provided, load conversation history
        enhanced_query = query
        if db:
            try:
                # Load chat history
                chat_id_int = int(chat_id)
                result = await db.execute(
                    select(Message).where(Message.chat_id == chat_id_int).order_by(Message.timestamp)
                )
                messages = result.scalars().all()
                
                # Convert to simple data structure
                history = [
                    {"sender": msg.sender, "content": msg.content}
                    for msg in messages
                ]
                
                # Only check for enhancing if we have history and this isn't the first question
                if history and len(history) >= 2:
                    # Count previous user messages (excluding the current one)
                    previous_user_messages = sum(1 for msg in history[:-1] if msg.get("sender") == "user")
                    
                    if previous_user_messages > 0:
                        # There are previous user messages, so this is a follow-up question
                        # Generate enhanced search query
                        enhanced_query = await self.generate_enhanced_search_query(query, history)
                    else:
                        logger.info(f"First question detected for chat {chat_id}. Using original query.")
                else:
                    logger.info(f"Insufficient history for chat {chat_id}. Using original query.")
            except Exception as e:
                logger.error(f"Error loading chat history: {e}")
        
        # Get web search context with enhanced query
        context = await self.get_web_search_context(enhanced_query)
        return context

# Create global WebSearchHandler instance
web_search_handler = WebSearchHandler()

# Function to get WebSearchHandler
def get_web_search_handler() -> WebSearchHandler:
    """Get WebSearchHandler instance"""
    return web_search_handler