#!/usr/bin/env python3
"""
DocumentRAG Wrapper for Full-Page Indexing

This module provides a customized wrapper that forces full-page indexing
by preventing the default chunking behavior.
Simplified version with only the essential functionality.
"""

import os
import logging
import asyncio
import time
from typing import Dict
import importlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("no-chunk-indexer")

class NoChunkDocumentRAG:
    """
    A wrapper that prevents automatic chunking of documents.
    This class intercepts the index_documents call and modifies the behavior to
    prevent the RecursiveCharacterTextSplitter from breaking documents into chunks.
    """
    
    def __init__(self, docs_dir: str, rag_storage_dir: str):
        try:
            query_docs = importlib.import_module('query_docs')
            DocumentRAG = getattr(query_docs, 'DocumentRAG')
            self.document_rag = DocumentRAG(docs_dir=docs_dir, rag_storage_dir=rag_storage_dir)
            logger.info(f"Initialized NoChunkDocumentRAG wrapper with docs_dir={docs_dir}, rag_storage_dir={rag_storage_dir}")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import DocumentRAG: {e}")
            raise

    async def index_documents(self, chat_id: str) -> Dict:
        logger.info(f"Starting non-chunking indexing for chat_id: {chat_id}")
        
        # ייבוא טרי בתוך הפונקציה
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        original_rag = self.document_rag
        original_split_documents = RecursiveCharacterTextSplitter.split_documents

        def no_split_documents(self, documents):
            logger.info(f"Preventing document splitting - keeping {len(documents)} full pages intact")
            return documents

        max_retries = 10
        base_delay = 90
        
        try:
            RecursiveCharacterTextSplitter.split_documents = no_split_documents
            
            for attempt in range(max_retries):
                try:
                    result = await original_rag.index_documents(chat_id)
                    logger.info(f"Completed non-chunking indexing with result: {result}")
                    return result
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Retrying after {delay:.0f} seconds.")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Error during indexing: {e}")
                        raise
            return {"error": f"Failed after {max_retries} attempts due to rate limits"}
        finally:
            RecursiveCharacterTextSplitter.split_documents = original_split_documents
            logger.info("Restored original document splitting behavior")
