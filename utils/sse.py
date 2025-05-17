"""
Server-Sent Events (SSE) Utilities

This module contains utilities for streaming responses as SSE (Server-Sent Events)
which can be used across the entire application for real-time streaming to clients.
"""

import asyncio
from typing import AsyncGenerator, Any

async def stream_text_as_sse(text: str, chunk_size: int = 100) -> AsyncGenerator[str, None]:
    """
    Stream text as Server-Sent Events (SSE) compliant chunks
    
    Args:
        text: The text to stream
        chunk_size: Size of each chunk in characters
        
    Yields:
        SSE formatted text chunks
    """
    # Stream the response in chunks
    for i in range(0, len(text), chunk_size):
        content = text[i:i+chunk_size]
        safe_content = content.replace('\n', '[NEWLINE]')
        yield f"data: {safe_content}\n\n"
        await asyncio.sleep(0)
    
    yield "data: [DONE]\n\n"

async def stream_generator_as_sse(generator, buffer_size: int = 100) -> AsyncGenerator[str, None]:
    """
    Stream content from another async generator as SSE compliant chunks
    
    Args:
        generator: Async generator that yields text
        buffer_size: Size threshold to send buffer in characters
        
    Yields:
        SSE formatted text chunks
    """
    buffer = ""
    try:
        async for chunk in generator:
            buffer += chunk
            if len(buffer) >= buffer_size:
                safe_content = buffer.replace('\n', '[NEWLINE]')
                yield f"data: {safe_content}\n\n"
                buffer = ""
                await asyncio.sleep(0)
        
        # Send any remaining buffer content
        if buffer:
            safe_content = buffer.replace('\n', '[NEWLINE]')
            yield f"data: {safe_content}\n\n"
        
        yield "data: [DONE]\n\n"
    except Exception as e:
        # Handle exceptions during streaming
        error_msg = f"Error during streaming: {str(e)}"
        safe_error = error_msg.replace('\n', '[NEWLINE]')
        yield f"data: {safe_error}\n\n"
        yield "data: [DONE]\n\n"

async def stream_code_as_sse(code_generator) -> AsyncGenerator[str, None]:
    """
    Stream code content from a generator as SSE compliant chunks without language tags
    
    This specialized function handles code content specifically, ensuring language tags
    aren't included in the output.
    
    Args:
        code_generator: Async generator that yields code content
        
    Yields:
        SSE formatted code chunks without language tags
    """
    buffer = ""
    in_code_block = False
    language_tag = ""
    
    try:
        async for chunk in code_generator:
            # Process language tags only at the beginning of code blocks
            if "```" in chunk and not in_code_block:
                parts = chunk.split("```", 1)
                if len(parts) > 1:
                    # First part is before the code block, second part may contain language tag
                    before_code = parts[0]
                    code_part = parts[1]
                    
                    # Check for language specifier in the first line
                    if "\n" in code_part:
                        first_line, rest = code_part.split("\n", 1)
                        if first_line.strip() and not first_line.strip()[0].isdigit():
                            # Language specifier found, remember it but don't include it
                            language_tag = first_line.strip()
                            code_part = rest
                    
                    # Only add the code part to the buffer
                    buffer += code_part
                    in_code_block = True
                    
            # End of code block
            elif "```" in chunk and in_code_block:
                parts = chunk.split("```", 1)
                if len(parts) > 0:
                    # Add everything before the closing marker
                    buffer += parts[0]
                
                # Send buffer as SSE
                if buffer:
                    safe_content = buffer.replace('\n', '[NEWLINE]')
                    yield f"data: {safe_content}\n\n"
                    buffer = ""
                
                in_code_block = False
                
            # Inside code block or normal text
            else:
                buffer += chunk
                
                # If buffer gets large, emit it
                if len(buffer) >= 100:
                    safe_content = buffer.replace('\n', '[NEWLINE]')
                    yield f"data: {safe_content}\n\n"
                    buffer = ""
                    await asyncio.sleep(0)
        
        # Send any remaining buffer content
        if buffer:
            safe_content = buffer.replace('\n', '[NEWLINE]')
            yield f"data: {safe_content}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        # Handle exceptions during streaming
        error_msg = f"Error during code streaming: {str(e)}"
        safe_error = error_msg.replace('\n', '[NEWLINE]')
        yield f"data: {safe_error}\n\n"
        yield "data: [DONE]\n\n"