#!/usr/bin/env python3
"""
Embeddings Generator for Confluence Index

This script specifically handles the embeddings generation part of the Confluence indexing process
in small batches to avoid rate limits. It assumes that pages have already been indexed and
chunks created using the standard confluence_indexer.py script.
"""

import asyncio
import logging
import os
import sys
import argparse
import time
import json
import importlib
from typing import List, Optional, Dict, Any

# Import the required dependencies
import certifi

# Avoid direct import to prevent circular imports
# We'll import NoChunkDocumentRAG dynamically in the code

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embeddings-creator")

async def create_embeddings_in_batches(
    output_dir: str = "confluence_index",
    kb_name: Optional[str] = None,
    knowledge_base_dir: str = "knowledgebase",
    batch_size: int = 25,
    wait_time: int = 120
) -> Dict[str, Any]:
    """
    Create embeddings in small batches to avoid rate limits
    
    Args:
        output_dir: Output directory used in the indexing process
        kb_name: Knowledge base name
        knowledge_base_dir: Base directory for knowledge bases
        batch_size: Number of documents to process in each batch
        wait_time: Time to wait between batches in seconds
        
    Returns:
        Dictionary with statistics about the process
    """
    logger.info(f"Starting embeddings creation in batches of {batch_size}")
    start_time = time.time()
    
    # Apply SSL fixes
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
    logger.info(f"Applied SSL fixes using certifi: {certifi.where()}")
    
    # Import NoChunkDocumentRAG dynamically to avoid circular imports
    try:
        no_chunk_module = importlib.import_module('no_chunk_rag')
        NoChunkDocumentRAG = getattr(no_chunk_module, 'NoChunkDocumentRAG')
        logger.info("Successfully imported NoChunkDocumentRAG")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import NoChunkDocumentRAG: {e}")
        return {"error": f"Failed to import NoChunkDocumentRAG: {e}"}
    
    # Determine the chunks directory and target directory
    if kb_name:
        kb_dir = os.path.join(knowledge_base_dir, kb_name)
        chunks_dir = os.path.join(kb_dir, "chunks")
        target_dir = kb_dir
    else:
        chunks_dir = os.path.join(output_dir, "chunks")
        target_dir = output_dir
    
    # Check if chunks directory exists and has files
    if not os.path.exists(chunks_dir):
        logger.error(f"Chunks directory {chunks_dir} does not exist")
        return {"error": "Chunks directory not found"}
        
    # List all chunk files
    chunk_files = [
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.endswith(".json") and not f.endswith("_embedding.json")
    ]
    
    if not chunk_files:
        logger.error(f"No chunk files found in {chunks_dir}")
        return {"error": "No chunk files found"}
        
    logger.info(f"Found {len(chunk_files)} chunk files to process")
    
    # Create a main temporary directory to store text files for DocumentRAG
    main_temp_dir = os.path.join(output_dir, "temp_for_rag")
    os.makedirs(main_temp_dir, exist_ok=True)
    
    # Create the main chat_confluence subfolder to match DocumentRAG's expected structure
    main_chat_dir = os.path.join(main_temp_dir, "chat_confluence")
    os.makedirs(main_chat_dir, exist_ok=True)
    
    # Create text files from chunks for DocumentRAG to process
    logger.info("Converting chunk files to text files...")
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            
            # Create a text file with the chunk content
            chunk_id = chunk_data["id"]
            # Save to the chat_confluence subfolder
            with open(os.path.join(main_chat_dir, f"{chunk_id}.txt"), "w", encoding="utf-8") as f:
                # Include metadata in the text to provide context
                f.write(f"Title: {chunk_data['page_title']}\n")
                f.write(f"Space: {chunk_data['space_key']}\n\n")
                f.write(chunk_data["content"])
        except Exception as e:
            logger.error(f"Error processing chunk file {chunk_file}: {e}")
    
    # Process files in smaller batches to avoid rate limits
    # Get all text files
    text_files = [f for f in os.listdir(main_chat_dir) if f.endswith(".txt")]
    total_files = len(text_files)
    total_batches = (total_files + batch_size - 1) // batch_size
    embeddings_created = 0
    
    logger.info(f"Will process {total_files} text files in {total_batches} batches of up to {batch_size} files each")
    
    # Process each batch
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = text_files[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_files)} files")
        
        # Create a temporary directory for this batch
        batch_temp_dir = os.path.join(output_dir, f"temp_batch_{batch_num}")
        os.makedirs(batch_temp_dir, exist_ok=True)
        
        # Create a chat_confluence subfolder in this batch directory
        batch_chat_dir = os.path.join(batch_temp_dir, "chat_confluence")
        os.makedirs(batch_chat_dir, exist_ok=True)
        
        # Copy only the files for this batch
        for file_name in batch_files:
            with open(os.path.join(main_chat_dir, file_name), "r", encoding="utf-8") as src_file:
                content = src_file.read()
                
            with open(os.path.join(batch_chat_dir, file_name), "w", encoding="utf-8") as dest_file:
                dest_file.write(content)
        
        # Process this batch with NoChunkDocumentRAG
        try:
            # Configure a new NoChunkDocumentRAG instance for this batch
            batch_rag = NoChunkDocumentRAG(
                docs_dir=batch_temp_dir, 
                rag_storage_dir=target_dir
            )
            
            # Try to index the documents with retry logic
            max_retries = 10
            base_delay = 90
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Use our NoChunkDocumentRAG for this batch
                    result = await batch_rag.index_documents(chat_id="confluence")
                    
                    # Update stats
                    if "chunks" in result:
                        batch_embeddings = result["chunks"]
                        embeddings_created += batch_embeddings
                        logger.info(f"Created {batch_embeddings} embeddings in batch {batch_num}")
                    
                    success = True
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a rate limit error (429)
                    if "429" in error_str and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). "
                                     f"Retrying after {delay:.0f} seconds.")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Error during batch {batch_num} embedding generation: {e}")
                        break
            
            # Clean up this batch's temp directory
            import shutil
            shutil.rmtree(batch_temp_dir)
            
            # If successful and not the last batch, wait before the next one
            if success and batch_num < total_batches:
                logger.info(f"Waiting {wait_time} seconds before next batch...")
                await asyncio.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            # Continue with the next batch even if this one failed
    
    # Clean up the main temp directory
    if os.path.exists(main_temp_dir):
        import shutil
        shutil.rmtree(main_temp_dir)
        logger.info(f"Cleaned up main temporary directory: {main_temp_dir}")
    
    # Update final stats
    end_time = time.time()
    total_duration = end_time - start_time
    
    stats = {
        "embeddings_created": embeddings_created,
        "total_files": total_files,
        "batches_processed": total_batches,
        "start_time": start_time,
        "end_time": end_time,
        "total_duration": total_duration
    }
    
    logger.info(f"Embeddings creation complete! Created {embeddings_created} embeddings in {total_duration:.2f} seconds")
    
    return stats
    
async def create_final_index(
    output_dir: str = "confluence_index",
    kb_name: Optional[str] = None,
    knowledge_base_dir: str = "knowledgebase",
    embeddings_stats: Dict[str, Any] = None
):
    """
    Create the final index file with updated statistics
    
    Args:
        output_dir: Output directory 
        kb_name: Knowledge base name
        knowledge_base_dir: Base directory for knowledge bases
        embeddings_stats: Statistics from embedding creation
    """
    logger.info("Creating final index with updated statistics...")
    
    # Determine the target directory
    if kb_name:
        target_dir = os.path.join(knowledge_base_dir, kb_name)
        spaces_dir = os.path.join(target_dir, "spaces")
        pages_dir = os.path.join(target_dir, "pages")
    else:
        target_dir = output_dir
        spaces_dir = os.path.join(output_dir, "spaces")
        pages_dir = os.path.join(output_dir, "pages")
    
    # Check if index already exists
    index_file = os.path.join(target_dir, "index.json")
    if os.path.exists(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
                
            # Update embeddings count if available
            if embeddings_stats and "embeddings_created" in embeddings_stats:
                if "stats" in index:
                    index["stats"]["embeddings_created"] = embeddings_stats["embeddings_created"]
                else:
                    index["stats"] = {"embeddings_created": embeddings_stats["embeddings_created"]}
                
            # Update index timestamp
            index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            
            # Save updated index
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Updated index file at {index_file}")
            return
        except Exception as e:
            logger.error(f"Error updating existing index: {e}")
    
    # If index doesn't exist or updating failed, create a new one
    try:
        # Load all space metadata
        spaces = {}
        if os.path.exists(spaces_dir):
            for file_name in os.listdir(spaces_dir):
                if file_name.endswith(".json"):
                    with open(os.path.join(spaces_dir, file_name), "r", encoding="utf-8") as f:
                        space_data = json.load(f)
                        spaces[space_data["key"]] = space_data
        
        # Load all page metadata
        pages = {}
        if os.path.exists(pages_dir):
            for file_name in os.listdir(pages_dir):
                if file_name.endswith(".json") and not file_name.endswith("_content.json"):
                    try:
                        with open(os.path.join(pages_dir, file_name), "r", encoding="utf-8") as f:
                            page_data = json.load(f)
                            pages[page_data["id"]] = page_data
                    except Exception as e:
                        logger.error(f"Error reading page file {file_name}: {e}")
        
        # Create stats object
        stats = {
            "spaces_processed": len(spaces),
            "pages_processed": len(pages),
            "embeddings_created": embeddings_stats.get("embeddings_created", 0) if embeddings_stats else 0,
            "total_duration": embeddings_stats.get("total_duration", 0) if embeddings_stats else 0
        }
        
        # Create master index
        index = {
            "spaces": spaces,
            "pages": pages,
            "stats": stats,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "kb_name": kb_name,
            "full_page_indexing": True  # Always true
        }
        
        # Save index
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created new index file at {index_file}")
        
    except Exception as e:
        logger.error(f"Error creating new index: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings for Confluence index in batches")
    parser.add_argument("--output-dir", default="confluence_index", help="Output directory used in the indexing process")
    parser.add_argument("--kb-name", help="Knowledge base name")
    parser.add_argument("--knowledge-base-dir", default="knowledgebase", help="Base directory for knowledge bases")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of files to process in each batch")
    parser.add_argument("--wait-time", type=int, default=120, help="Time to wait between batches in seconds")
    
    args = parser.parse_args()
    
    # Run embeddings creation
    try:
        logger.info("Starting embeddings creation process...")
        
        async def main():
            # Create embeddings
            embeddings_stats = await create_embeddings_in_batches(
                output_dir=args.output_dir,
                kb_name=args.kb_name,
                knowledge_base_dir=args.knowledge_base_dir,
                batch_size=args.batch_size,
                wait_time=args.wait_time
            )
            
            # Update or create final index
            await create_final_index(
                output_dir=args.output_dir,
                kb_name=args.kb_name,
                knowledge_base_dir=args.knowledge_base_dir,
                embeddings_stats=embeddings_stats
            )
            
            logger.info("Embeddings creation and index update complete!")
        
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\nEmbeddings creation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during embeddings creation: {e}")
        sys.exit(1)