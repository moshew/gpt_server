"""
Document RAG module

This module:
1. Handles document processing and storage
2. Implements vector-based retrieval for document search
3. Provides RAG functionality for document-related queries
4. Supports various document formats (PDF, TXT, DOCX, etc.)
5. Enhanced for text-embedding-3-large model
"""

import os
import asyncio
import aiofiles
import aiofiles.os  
import aiofiles.ospath
import time
import logging
import pickle
import numpy as np
import faiss
import datetime
from typing import Dict, List, Optional, Tuple, Any
from functools import partial

from .utils import run_in_executor
import uuid
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    CSVLoader
)

from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from .utils import embed_documents, embed_query

# Define path for knowledge bases
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "data/knowledgebase")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("query_docs")

# Define embedding model constants
EMBEDDING_MODEL = "text-embedding-3-large"  # The embedding model we're using
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072
}

class DocumentRAG:
    """
    Class to handle document RAG operations including:
    - Document storage and management
    - Vector search and retrieval
    - RAG-enhanced query processing
    """
    
    def __init__(self, docs_dir: str, rag_storage_dir: str):
        """
        Initialize the DocumentRAG system
        
        Args:
            docs_dir: Base directory for document storage
            rag_storage_dir: Base directory for RAG index storage
        """
        self.docs_dir = docs_dir
        self.rag_storage_dir = rag_storage_dir
        
    def _get_chat_folder(self, chat_id: str) -> str:
        """
        Get the document folder for a specific chat
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            Path to chat's document folder
        """
        # Changed to use 'chats/chat_<chat_id>' path structure
        chat_folder = os.path.join(self.docs_dir, f"chat_{chat_id}")
        os.makedirs(chat_folder, exist_ok=True)
        return chat_folder
    
    def _get_rag_storage_folder(self, chat_id: str) -> str:
        """
        Get the RAG storage folder path for a specific chat (doesn't create it)
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            Path to chat's RAG storage folder
        """
        return os.path.join(self.rag_storage_dir, chat_id)
    
    def _save_file(self, file_path: str, content: bytes):
        """
        Save file content to disk
        
        Args:
            file_path: Path to save the file
            content: File content bytes
        """
        with open(file_path, "wb") as f:
            f.write(content)
    
    async def _load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of LangChain Document objects
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Use run_in_executor to run blocking document loading in a separate thread
            if file_ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_ext == ".txt":
                loader = TextLoader(file_path)
            elif file_ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_ext == ".csv":
                loader = CSVLoader(file_path)
            else:
                # Default to text loader for other types
                loader = TextLoader(file_path)
                
            return await run_in_executor(loader.load)
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    def _create_faiss_index(self, dimension: int, embeddings_array: np.ndarray) -> faiss.Index:
        """
        Create an optimized FAISS index for high-dimensional embeddings
        
        Args:
            dimension: Vector dimension
            embeddings_array: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        # For small document sets, use a simple flat index
        if len(embeddings_array) < 1000:
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            return index
            
        # For larger document sets, use an IVF index for better search performance
        # with high-dimension vectors like text-embedding-3-large (3072 dimensions)
        
        # Calculate number of clusters based on dataset size
        # Rule of thumb: sqrt(n) clusters where n is number of vectors
        n_clusters = min(4096, max(int(np.sqrt(len(embeddings_array))), 50))
        
        quantizer = faiss.IndexFlatL2(dimension)  # The quantizer defines how vectors are assigned to clusters
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_L2)
        
        # IVF indexes need to be trained before adding vectors
        if not index.is_trained:
            index.train(embeddings_array)
        
        # Add vectors to the trained index
        index.add(embeddings_array)
        
        # Set the number of nearest clusters to search
        # Higher values = more accurate but slower searches
        index.nprobe = min(n_clusters, 10)  # Search 10 clusters by default
        
        logger.info(f"Created IVF index with {index.ntotal} vectors, {n_clusters} clusters")
        return index
    
    def _save_index_files(self, index_file: str, faiss_index_file: str, 
                         doc_store: Dict, ids: List, indexed_files: List, 
                         failed_files: List, dimension: int):
        """
        Save index and document store to files with model information
        
        Args:
            index_file: Path to save index data
            faiss_index_file: Path to save FAISS index
            doc_store: Document store dictionary
            ids: Document IDs
            indexed_files: List of indexed file names
            failed_files: List of failed files
            dimension: Dimension of the embedding vectors
        """
        with open(index_file, "wb") as f:
            pickle.dump({
                "doc_store": doc_store,
                "ids": ids,
                "indexed_files": indexed_files,
                "failed_files": failed_files,
                "indexed_at": datetime.datetime.now().isoformat(),
                "embedding_model": EMBEDDING_MODEL,  # Store model info
                "dimension": dimension  # Store dimension info
            }, f)
    
    async def index_documents(self, chat_id: str, cancellation_check: Optional[callable] = None) -> Dict:
        """
        Index all documents for a specific chat with better error handling
        
        Args:
            chat_id: Chat identifier
            cancellation_check: Optional callback function to check if indexing should be cancelled
            
        Returns:
            Indexing results (only after all operations are complete)
        """
        chat_folder = self._get_chat_folder(chat_id)
        rag_folder = self._get_rag_storage_folder(chat_id)
        
        # Create the RAG folder only when indexing
        os.makedirs(rag_folder, exist_ok=True)
        
        # Store index files in the RAG storage folder
        index_file = os.path.join(rag_folder, "document_index.pkl")
        faiss_index_file = os.path.join(rag_folder, "document_faiss.index")
        
        # Use file-based locking to prevent concurrent indexing of the same chat
        lock_file = os.path.join(rag_folder, "indexing.lock")
        
        # Check if indexing is already in progress
        if await aiofiles.ospath.exists(lock_file):
            # Check if the lock is stale (older than 1 hour)
            lock_time = await aiofiles.os.path.getmtime(lock_file)
            if time.time() - lock_time < 3600:  # Less than 1 hour old
                return {"message": "Indexing already in progress"}
            else:
                # Remove stale lock
                try:
                    await aiofiles.os.remove(lock_file)
                except Exception as e:
                    logger.error(f"Error removing stale lock: {e}")
                    # Continue anyway
        
        # Create lock file
        try:
            await self._create_lock_file_async(lock_file)
        except Exception as e:
            logger.error(f"Error creating lock file: {e}")
            # Continue anyway
        
        try:
            # Check for cancellation before starting
            if cancellation_check and cancellation_check():
                logger.info(f"Indexing cancelled before start for chat {chat_id}")
                return {"message": "Indexing cancelled"}
            
            # List all document files in the folder recursively using run_in_executor
            def find_all_files():
                all_files = []
                if os.path.exists(chat_folder):
                    # Note: os.walk doesn't have an async equivalent in aiofiles
                    # so we still use run_in_executor for directory traversal
                    for root, dirs, files in os.walk(chat_folder):
                        for file in files:
                            full_path = os.path.join(root, file)
                            # Get relative path from chat_folder
                            relative_path = os.path.relpath(full_path, chat_folder)
                            
                            # Skip system files and index files
                            if (not file.startswith("document_") and 
                                file != "indexing.lock" and
                                not file.startswith(".")):  # Skip hidden files
                                all_files.append({
                                    "relative_path": relative_path,
                                    "full_path": full_path,
                                    "filename": file
                                })
                return all_files
            
            file_info_list = await run_in_executor(find_all_files)
            
            logger.info(f"Found {len(file_info_list)} files to index in chat {chat_id}")
            
            if not file_info_list:
                # Ensure lock file is removed before returning
                if await aiofiles.ospath.exists(lock_file):
                    await aiofiles.os.remove(lock_file)
                return {"message": "No documents found to index"}
            
            # Process documents
            all_docs = []
            indexed_files = []
            failed_files = []
            
            for file_info in file_info_list:
                # Check for cancellation during file processing
                if cancellation_check and cancellation_check():
                    logger.info(f"Indexing cancelled during file processing for chat {chat_id}")
                    return {"message": "Indexing cancelled"}
                
                try:
                    file_path = file_info["full_path"]
                    relative_path = file_info["relative_path"]
                    
                    # Use the async method for loading documents - WAIT for completion
                    documents = await self._load_document(file_path)
                    
                    # Yield control back to the event loop
                    await asyncio.sleep(0)
                    
                    if documents:
                        # Add file info to document metadata including subdirectory path
                        for doc in documents:
                            doc.metadata["file_name"] = relative_path  # Use relative path
                            doc.metadata["original_filename"] = file_info["filename"]  # Original filename
                            
                        all_docs.extend(documents)
                        indexed_files.append(relative_path)
                    else:
                        logger.warning(f"No documents loaded from {relative_path}")
                        failed_files.append(relative_path)
                except Exception as e:
                    logger.error(f"Error loading document {file_info['relative_path']}: {e}")
                    failed_files.append(file_info["relative_path"])
                    continue
                
                # Yield control back to the event loop after each file
                await asyncio.sleep(0)
            
            if not all_docs:
                # Ensure lock file is removed before returning
                if await aiofiles.ospath.exists(lock_file):
                    await aiofiles.os.remove(lock_file)
                return {"message": "No content could be extracted from documents", "failed_files": failed_files}
            
            # Check for cancellation before chunking
            if cancellation_check and cancellation_check():
                logger.info(f"Indexing cancelled before chunking for chat {chat_id}")
                return {"message": "Indexing cancelled"}
            
            logger.info(f"Splitting {len(all_docs)} documents into chunks...")
            # Split documents into chunks using run_in_executor - WAIT for completion
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = await run_in_executor(text_splitter.split_documents, all_docs)
            
            # Yield control back to the event loop
            await asyncio.sleep(0)
            
            # Check for cancellation before creating embeddings
            if cancellation_check and cancellation_check():
                logger.info(f"Indexing cancelled before embeddings for chat {chat_id}")
                return {"message": "Indexing cancelled"}
            
            # Create document store
            doc_store = {}
            for i, chunk in enumerate(chunks):
                doc_store[str(i)] = chunk
                
                # Yield control every 100 chunks
                if i % 100 == 0:
                    await asyncio.sleep(0)
                    # Check for cancellation periodically during doc store creation
                    if cancellation_check and cancellation_check():
                        logger.info(f"Indexing cancelled during doc store creation for chat {chat_id}")
                        return {"message": "Indexing cancelled"}
            
            # Generate embeddings
            texts = [chunk.page_content for chunk in chunks]
            ids = list(range(len(chunks)))
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            # Use the embed_documents function - WAIT for completion
            embeddings_list = await embed_documents(texts)
            
            # Yield control back to the event loop
            await asyncio.sleep(0)
            
            # Check for cancellation before creating FAISS index
            if cancellation_check and cancellation_check():
                logger.info(f"Indexing cancelled before FAISS index creation for chat {chat_id}")
                return {"message": "Indexing cancelled"}
            
            # Create FAISS index optimized for high-dimensional vectors
            dimension = len(embeddings_list[0])
            
            # Convert to numpy array and create index - WAIT for completion
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            index = await run_in_executor(
                lambda: self._create_faiss_index(dimension, embeddings_array)
            )
            
            # Yield control back to the event loop
            await asyncio.sleep(0)
            
            # Check for cancellation before saving
            if cancellation_check and cancellation_check():
                logger.info(f"Indexing cancelled before saving for chat {chat_id}")
                return {"message": "Indexing cancelled"}
            
            # Save index and document store using run_in_executor - WAIT for completion
            await run_in_executor(
                lambda: self._save_document_index(index_file, doc_store, ids, 
                                                 indexed_files, failed_files, dimension)
            )
            
            await run_in_executor(faiss.write_index, index, faiss_index_file)
            
            # Verify that both files were actually saved
            if not (await aiofiles.ospath.exists(index_file) and 
                    await aiofiles.ospath.exists(faiss_index_file)):
                raise Exception("Failed to save index files to disk")
            
            result = {
                "message": "Documents indexed successfully",
                "indexed_files": indexed_files,
                "failed_files": failed_files,
                "chunks": len(chunks),
                "embedding_model": EMBEDDING_MODEL,
                "dimension": dimension
            }
            logger.info(f"Indexing completed: {len(indexed_files)} files, {len(chunks)} chunks")
            return result
            
        except Exception as e:
            error_msg = f"Error indexing documents: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
        finally:
            # ALWAYS remove lock file, wait for completion
            try:
                if await aiofiles.ospath.exists(lock_file):
                    await aiofiles.os.remove(lock_file)
                    # Log cleanup for debugging
                    if cancellation_check and cancellation_check():
                        logger.info(f"Cleaned up indexing lock file after cancellation for chat {chat_id}")
                    else:
                        logger.debug(f"Cleaned up indexing lock file after completion for chat {chat_id}")
            except Exception as e:
                logger.error(f"Error removing lock file: {e}")
            
            # Final verification that everything is complete
            logger.info(f"Indexing process for chat {chat_id} fully completed")
    
    async def _create_lock_file_async(self, lock_file: str):
        """
        Create a lock file asynchronously
        
        Args:
            lock_file: Path to the lock file
        """
        async with aiofiles.open(lock_file, "w") as f:
            await f.write(f"Indexing started at {datetime.datetime.now().isoformat()}")
    
    def _save_document_index(self, index_file: str, doc_store: Dict, ids: List, 
                            indexed_files: List, failed_files: List, dimension: int = None):
        """
        Save document index to a file with model information
        
        Args:
            index_file: Path to the index file
            doc_store: Document store dictionary
            ids: Document IDs
            indexed_files: List of indexed files
            failed_files: List of failed files
            dimension: Dimension of the embedding vectors
        """
        with open(index_file, "wb") as f:
            index_data = {
                "doc_store": doc_store,
                "ids": ids,
                "indexed_files": indexed_files,
                "failed_files": failed_files,
                "indexed_at": datetime.datetime.now().isoformat(),
                "embedding_model": EMBEDDING_MODEL
            }
            
            # Add dimension if provided
            if dimension:
                index_data["dimension"] = dimension
                
            pickle.dump(index_data, f)
    
    async def list_documents(self, chat_id: str) -> Dict:
        """
        List all documents for a specific chat, including subdirectories
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            List of document files and indexing status
        """
        chat_folder = self._get_chat_folder(chat_id)
        rag_folder = self._get_rag_storage_folder(chat_id)
        
        # List all document files in the folder recursively using run_in_executor
        def find_all_files():
            all_files = []
            if os.path.exists(chat_folder):
                # Note: os.walk doesn't have an async equivalent in aiofiles
                # so we still use run_in_executor for directory traversal
                for root, dirs, files in os.walk(chat_folder):
                    for file in files:
                        full_path = os.path.join(root, file)
                        # Get relative path from chat_folder
                        relative_path = os.path.relpath(full_path, chat_folder)
                        
                        # Skip system files and index files
                        if (not file.startswith("document_") and 
                            file != "indexing.lock" and
                            not file.startswith(".")):  # Skip hidden files
                            all_files.append(relative_path)
            return all_files
        
        files = await run_in_executor(find_all_files)
        
        # Check if RAG folder exists and index exists
        indexed_files = []
        indexed_at = None
        embedding_model = None
        dimension = None
        
        # Only check for index if the RAG folder exists
        if await aiofiles.ospath.exists(rag_folder):
            index_file = os.path.join(rag_folder, "document_index.pkl")
            if await aiofiles.ospath.exists(index_file):
                index_data = await run_in_executor(self._load_index_data, index_file)
                indexed_files = index_data.get("indexed_files", [])
                indexed_at = index_data.get("indexed_at")
                embedding_model = index_data.get("embedding_model")
                dimension = index_data.get("dimension")
        
        result = {
            "document_files": files,
            "indexed_files": indexed_files,
            "indexed_at": indexed_at
        }
        
        # Add model info if available
        if embedding_model:
            result["embedding_model"] = embedding_model
        if dimension:
            result["dimension"] = dimension
            
        return result
    
    def _load_index_data(self, index_file: str) -> Dict:
        """
        Load index data from a file
        
        Args:
            index_file: Path to the index file
            
        Returns:
            Index data dictionary
        """
        with open(index_file, "rb") as f:
            return pickle.load(f)
    
    async def check_indexing_needed(self, chat_id: str) -> bool:
        """
        Check if indexing is needed for a chat
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            True if indexing is needed, False otherwise
        """
        try:
            doc_info = await self.list_documents(chat_id)
            
            document_files = doc_info.get("document_files", [])
            indexed_files = doc_info.get("indexed_files", [])
            
            # If no documents exist, no indexing needed
            if not document_files:
                return False
            
            # If no index exists but documents exist, indexing needed
            if not indexed_files:
                logger.info(f"No index found for chat {chat_id}, indexing needed")
                return True
            
            # Check if new files were added or existing files changed
            current_files = set(document_files)
            indexed_files_set = set(indexed_files)
            
            # If there are new files, indexing needed
            if current_files != indexed_files_set:
                logger.info(f"Files changed for chat {chat_id}: current={current_files}, indexed={indexed_files_set}")
                return True
            
            logger.info(f"Index is up to date for chat {chat_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error checking indexing status for chat {chat_id}: {e}")
            # If we can't determine, assume indexing is needed for safety
            return True

    async def retrieve_relevant_documents(self, chat_id: str, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            chat_id: Chat identifier
            query: Query text
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant document chunks
        """
        rag_folder = self._get_rag_storage_folder(chat_id)
        
        # Check if RAG folder exists first
        if not await aiofiles.ospath.exists(rag_folder):
            return []
        
        index_file = os.path.join(rag_folder, "document_index.pkl")
        faiss_index_file = os.path.join(rag_folder, "document_faiss.index")
        
        # Check if index exists
        if not (await aiofiles.ospath.exists(index_file) and 
                await aiofiles.ospath.exists(faiss_index_file)):
            return []
        
        # Load index data
        index_data = await run_in_executor(self._load_index_data, index_file)
        doc_store = index_data["doc_store"]
        
        # Check compatibility
        saved_model = index_data.get("embedding_model")
        if saved_model and saved_model != EMBEDDING_MODEL:
            logger.warning(f"Index was created with model {saved_model}, but current model is {EMBEDDING_MODEL}")
            logger.warning("This may cause poor retrieval results. Consider reindexing.")
        
        # Load FAISS index
        index = await run_in_executor(faiss.read_index, faiss_index_file)
        
        # Generate query embedding using the new function from llm_helpers
        query_embedding = await embed_query(query)
        
        # Convert to numpy array
        query_embedding_array = np.array([query_embedding], dtype=np.float32)
        
        # Search for similar documents using run_in_executor
        D, I = await run_in_executor(index.search, query_embedding_array, top_k)
        
        # Retrieve documents from doc store
        results = []
        for i in I[0]:
            if str(i) in doc_store:
                results.append(doc_store[str(i)])
        
        return results


    async def get_document_context(self, chat_id: str, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant document context for a query if available
        
        Args:
            chat_id: Chat identifier
            query: User's question
            top_k: Number of top chunks to retrieve
            
        Returns:
            Document context as string or empty string if no relevant context found
        """
        try:
            # Retrieve relevant documents (handles all existence checks internally)
            relevant_docs = await self.retrieve_relevant_documents(str(chat_id), query, top_k=top_k)
            
            if relevant_docs:
                # Format the document context in a useful way
                context = "\n\nRelevant document information:\n" + "\n\n".join([
                    f"From document '{doc.metadata.get('file_name', 'unknown')}':\n{doc.page_content}"
                    for doc in relevant_docs
                ])
                
                logger.info(f"Retrieved context from {len(relevant_docs)} document chunks for query in chat {chat_id}")
                return context
        except Exception as e:
            logger.error(f"Error retrieving document context for chat {chat_id}: {e}")
        
        return ""

# Create DocumentRAG instance with updated paths under data directory
document_rag_handler = DocumentRAG(docs_dir="data/chats", rag_storage_dir="data/rag")

# Function to get DocumentRAG for specific chat
def get_document_rag(chat_id: str) -> DocumentRAG:
    """Get DocumentRAG instance for a specific chat"""
    return document_rag_handler

async def get_document_context(
    chat_id: str, 
    query: Optional[str] = None, 
    top_k: int = 3, 
    keep_original_files: bool = False,
    source_type: str = "chat"  # "chat", "kb", or "code"
) -> str:
    """
    Get document context for a query, either using RAG or returning original files
    
    Args:
        chat_id: Chat identifier (or knowledge base name for KB queries)
        query: Query text (required for RAG, optional for original files)
        top_k: Number of top chunks to retrieve (for RAG only)
        keep_original_files: Whether to return original file contents instead of RAG
        source_type: Type of source - "chat", "kb", or "code"
        
    Returns:
        Document context string
    """
    if keep_original_files:
        return await _get_original_files_content(chat_id, source_type)
    else:
        # Use RAG
        if not query:
            return ""
            
        if source_type == "kb":
            # Knowledge base RAG
            kb_path = os.path.join(KNOWLEDGE_BASE_DIR, chat_id)
            if os.path.exists(kb_path):
                kb_rag = DocumentRAG(docs_dir=KNOWLEDGE_BASE_DIR, rag_storage_dir=KNOWLEDGE_BASE_DIR)
                context = await kb_rag.get_document_context(chat_id, query, top_k=5)
                if context:
                    return f"Information from knowledge base '{chat_id}':\n{context}"
            return ""
        else:
            # Chat documents RAG
            doc_rag = get_document_rag(str(chat_id))
            context = await doc_rag.get_document_context(str(chat_id), query, top_k=top_k)
            return context or ""

async def _get_original_files_content(chat_id: str, source_type: str) -> str:
    """
    Read original file contents instead of using RAG
    
    Args:
        chat_id: Chat identifier (or knowledge base name for KB queries)
        source_type: Type of source - "chat", "kb", or "code"
        
    Returns:
        Combined file contents
        
    Raises:
        HTTPException: If total word count exceeds 100K words
    """
    MAX_WORDS = 100000  # 100K words limit
    
    # Determine source directory based on type
    if source_type == "kb":
        docs_dir = os.path.join(KNOWLEDGE_BASE_DIR, chat_id)
        source_description = f"knowledge base '{chat_id}'"
    elif source_type == "code":
        docs_dir = f"data/code/{chat_id}"
        source_description = f"code files for chat {chat_id}"
    else:  # chat
        docs_dir = f"data/chats/chat_{chat_id}"
        source_description = f"chat {chat_id} documents"
    
    all_content = []
    total_words = 0
    
    try:
        if not os.path.exists(docs_dir):
            return ""
        
        # Supported file extensions
        supported_extensions = {
            '.txt', '.md', '.py', '.js', '.html', '.css', '.json', 
            '.yaml', '.yml', '.xml', '.csv', '.pdf', '.docx', '.doc'
        }
        
        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # Skip unsupported file types
                if file_ext not in supported_extensions:
                    continue
                
                try:
                    # Read file content based on type
                    if file_ext == '.pdf':
                        content = await _read_pdf_content(file_path)
                    elif file_ext in ['.docx', '.doc']:
                        content = await _read_docx_content(file_path)
                    else:
                        # Text-based files
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    
                    if not content.strip():
                        continue
                    
                    # Count words in this file
                    word_count = len(content.split())
                    
                    # Check if adding this file would exceed limit
                    if total_words + word_count > MAX_WORDS:
                        logger.warning(f"File content exceeds 100K words limit. Current: {total_words}, File: {word_count}")
                        raise HTTPException(
                            status_code=400, 
                            detail=f"File contents are too large ({total_words + word_count:,} words). "
                                   f"Maximum allowed is {MAX_WORDS:,} words. "
                                   f"Consider unchecking 'keep original files' to use document excerpts instead."
                        )
                    
                    # Add file content with header
                    relative_path = os.path.relpath(file_path, docs_dir)
                    file_header = f"\n--- File: {relative_path} ---\n"
                    all_content.append(file_header + content)
                    total_words += word_count
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
                    continue
        
        if not all_content:
            return ""
        
        combined_content = "\n".join(all_content)
        return f"Original file contents from {source_description}:\n{combined_content}"
        
    except HTTPException:
        # Re-raise HTTP exceptions (word limit exceeded)
        raise
    except Exception as e:
        logger.error(f"Error reading original files from {docs_dir}: {e}")
        return ""

async def _read_pdf_content(file_path: str) -> str:
    """Read content from PDF file"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = await run_in_executor(loader.load)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

async def _read_docx_content(file_path: str) -> str:
    """Read content from DOCX file"""
    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = await run_in_executor(loader.load)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error reading DOCX {file_path}: {e}")
        return ""