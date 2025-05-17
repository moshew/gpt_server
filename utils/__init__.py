"""
Utility package for application

This package provides shared utilities across the application,
organized by functionality.
"""

# Async helpers
from .async_helpers import (
    run_in_executor,
    run_tasks_with_limit,
    AsyncLock,
    async_lock_manager
)

# Text processing
from .text_processing import (
    parse_json_string,
    serialize_to_json,
    run_regex_search,
    run_regex_findall,
    extract_code_from_markdown
)

# SSE utilities
from .sse import (
    stream_text_as_sse,
    stream_generator_as_sse,
    stream_code_as_sse
)

# LLM helpers
from .llm_helpers_azure import (
    call_llm,
    process_json_response,
    process_langchain_messages,
    embed_documents,
    embed_query,
    execute_image_generation
)


"""
# RAG-specific utilities
from .rag_utils import (
    chunk_content,
    chunk_content_async
)
"""

# Make legacy aliases available for backward compatibility
# These will maintain compatibility with existing code
# while encouraging use of the more specialized modules directly

# Alias for backward compatibility
run_cpu_bound = run_in_executor