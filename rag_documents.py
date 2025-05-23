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
import time
import pickle
import asyncio
import datetime
import shutil
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from functools import partial

from utils import run_in_executor
import numpy as np
import faiss
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

from utils import embed_documents, embed_query

# Import aiofiles for async file operations
import aiofiles
import aiofiles.os
import aiofiles.ospath

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_documents")

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
        Create a FAISS index from embeddings, optimized for high-dimensional vectors
        This function is designed to be run in an executor for thread safety
        
        Args:
            dimension: Embedding dimension
            embeddings_array: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        print(f"Creating FAISS index with {len(embeddings_array)} vectors of dimension {dimension}")
        
        # For small document sets, use a simple flat index
        if len(embeddings_array) < 1000:
            print("Using IndexFlatL2 for small dataset")
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            print(f"Successfully created flat index with {index.ntotal} vectors")
            return index
            
        # For larger document sets, use an IVF index for better search performance
        # with high-dimension vectors like text-embedding-3-large (3072 dimensions)
        
        print("Using IndexIVFFlat for large dataset")
        
        # Calculate number of clusters based on dataset size
        # Rule of thumb: sqrt(n) clusters where n is number of vectors
        n_clusters = min(4096, max(int(np.sqrt(len(embeddings_array))), 50))
        print(f"Using {n_clusters} clusters for IVF index")
        
        quantizer = faiss.IndexFlatL2(dimension)  # The quantizer defines how vectors are assigned to clusters
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_L2)
        
        # IVF indexes need to be trained before adding vectors
        if not index.is_trained:
            print("Training IVF index...")
            index.train(embeddings_array)
        
        # Add vectors to the trained index
        print("Adding vectors to index...")
        index.add(embeddings_array)
        
        # Set the number of nearest clusters to search
        # Higher values = more accurate but slower searches
        index.nprobe = min(n_clusters, 10)  # Search 10 clusters by default
        
        print(f"Successfully created IVF index with {index.ntotal} vectors, {n_clusters} clusters, nprobe={index.nprobe}")
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
    
    async def index_documents(self, chat_id: str) -> Dict:
        """
        Index all documents for a specific chat with better error handling
        
        Args:
            chat_id: Chat identifier
            
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
            
            print(f"Found {len(file_info_list)} files to index in chat {chat_id} (including subdirectories)")
            if file_info_list:
                print("Files found:")
                for file_info in file_info_list:
                    print(f"  - {file_info['relative_path']}")
            
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
                try:
                    file_path = file_info["full_path"]
                    relative_path = file_info["relative_path"]
                    print(f"Loading document {relative_path}...")
                    
                    # Use the async method for loading documents - WAIT for completion
                    documents = await self._load_document(file_path)
                    
                    # Yield control back to the event loop
                    await asyncio.sleep(0)
                    
                    if documents:
                        print(f"Successfully loaded {len(documents)} documents from {relative_path}")
                        # Add file info to document metadata including subdirectory path
                        for doc in documents:
                            doc.metadata["file_name"] = relative_path  # Use relative path
                            doc.metadata["original_filename"] = file_info["filename"]  # Original filename
                            
                        all_docs.extend(documents)
                        indexed_files.append(relative_path)
                    else:
                        print(f"No documents loaded from {relative_path}")
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
            
            print(f"Splitting {len(all_docs)} documents into chunks...")
            # Split documents into chunks using run_in_executor - WAIT for completion
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = await run_in_executor(text_splitter.split_documents, all_docs)
            print(f"Created {len(chunks)} chunks")
            
            # Yield control back to the event loop
            await asyncio.sleep(0)
            
            # Create document store
            doc_store = {}
            for i, chunk in enumerate(chunks):
                doc_store[str(i)] = chunk
                
                # Yield control every 100 chunks
                if i % 100 == 0:
                    await asyncio.sleep(0)
            
            # Generate embeddings
            texts = [chunk.page_content for chunk in chunks]
            ids = list(range(len(chunks)))
            
            print(f"Generating embeddings for {len(texts)} chunks...")
            # Use the embed_documents function - WAIT for completion
            embeddings_list = await embed_documents(texts)
            print(f"Generated {len(embeddings_list)} embeddings")
            
            # Yield control back to the event loop
            await asyncio.sleep(0)
            
            # Create FAISS index optimized for high-dimensional vectors
            dimension = len(embeddings_list[0])
            logger.info(f"Creating FAISS index with {dimension} dimensions")
            
            # Convert to numpy array and create index - WAIT for completion
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            index = await run_in_executor(
                lambda: self._create_faiss_index(dimension, embeddings_array)
            )
            
            # Yield control back to the event loop
            await asyncio.sleep(0)
            
            # Save index and document store using run_in_executor - WAIT for completion
            print(f"Saving index to {index_file}...")
            await run_in_executor(
                lambda: self._save_document_index(index_file, doc_store, ids, 
                                                 indexed_files, failed_files, dimension)
            )
            
            print(f"Saving FAISS index to {faiss_index_file}...")
            await run_in_executor(faiss.write_index, index, faiss_index_file)
            
            # Verify that both files were actually saved
            if not (await aiofiles.ospath.exists(index_file) and 
                    await aiofiles.ospath.exists(faiss_index_file)):
                raise Exception("Failed to save index files to disk")
            
            print(f"Index files successfully saved and verified")
            
            result = {
                "message": "Documents indexed successfully",
                "indexed_files": indexed_files,
                "failed_files": failed_files,
                "chunks": len(chunks),
                "embedding_model": EMBEDDING_MODEL,
                "dimension": dimension
            }
            print(f"Indexing completed: {result}")
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
                    print(f"Lock file {lock_file} removed")
            except Exception as e:
                logger.error(f"Error removing lock file: {e}")
            
            # Final verification that everything is complete
            print(f"Indexing process for chat {chat_id} fully completed")
    
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
            logger.info(f"No RAG storage folder found for chat {chat_id}")
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
            rag_folder = self._get_rag_storage_folder(str(chat_id))
            
            # Check if the folder exists before proceeding
            if not await aiofiles.ospath.exists(rag_folder):
                logger.info(f"No RAG storage folder found for chat {chat_id}")
                return ""
                
            index_file = os.path.join(rag_folder, "document_index.pkl")
            faiss_index_file = os.path.join(rag_folder, "document_faiss.index")
            
            # Check if index exists using aiofiles for non-blocking I/O
            if (await aiofiles.ospath.exists(index_file) and 
                await aiofiles.ospath.exists(faiss_index_file)):
                
                # Retrieve relevant documents
                relevant_docs = await self.retrieve_relevant_documents(str(chat_id), query, top_k=top_k)
                
                if relevant_docs:
                    # Format the document context in a useful way
                    context = "\n\nRelevant document information:\n" + "\n\n".join([
                        f"From document '{doc.metadata.get('file_name', 'unknown')}':\n{doc.page_content}"
                        for doc in relevant_docs
                    ])
                    
                    logger.info(f"Retrieved context from {len(relevant_docs)} document chunks for query in chat {chat_id}")
                    return context
                else:
                    logger.info(f"No relevant document chunks found for query in chat {chat_id}")
            else:
                logger.info(f"No document index found for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error retrieving document context for chat {chat_id}: {e}")
        
        return ""

# Create DocumentRAG instance with updated paths
document_rag_handler = DocumentRAG(docs_dir="chats", rag_storage_dir="rag_storage")

# Function to get DocumentRAG for specific chat
def get_document_rag(chat_id: str) -> DocumentRAG:
    """Get DocumentRAG instance for a specific chat"""
    return document_rag_handler