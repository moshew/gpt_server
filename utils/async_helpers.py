"""
Async Helper Utilities

This module contains general-purpose asynchronous utility functions that can be used
across the entire application, not just in RAG-specific contexts.
"""

import asyncio
import concurrent.futures
from typing import Any, Callable, List, AsyncGenerator
from functools import partial

# Thread pool for CPU-intensive operations
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

async def run_in_executor(func, *args, **kwargs):
    """
    Run a blocking function in an executor to avoid blocking the event loop
    
    Args:
        func: Function to run
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Result of the function
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        return await loop.run_in_executor(
            _thread_pool, 
            lambda: func(*args, **kwargs)
        )
    else:
        return await loop.run_in_executor(
            _thread_pool, 
            lambda: func(*args)
        )

async def run_tasks_with_limit(tasks: List[Callable[[], AsyncGenerator]], limit: int = 5) -> AsyncGenerator[Any, None]:
    """
    Run multiple async tasks with a concurrency limit
    
    Args:
        tasks: List of task functions that return async generators
        limit: Maximum number of concurrent tasks
        
    Yields:
        Results from tasks as they complete
    """
    # Convert tasks to semaphore-limited coroutines
    semaphore = asyncio.Semaphore(limit)
    
    async def run_task_with_semaphore(task_fn):
        async with semaphore:
            async for result in task_fn():
                yield result
    
    # Create task runners
    task_runners = [run_task_with_semaphore(task) for task in tasks]
    
    # Process results as they arrive
    for runner in asyncio.as_completed(task_runners):
        async for result in await runner:
            yield result

class AsyncLock:
    """
    Simplified async lock implementation for resource protection
    """
    def __init__(self):
        self._locks = {}
        self._lock_creation_lock = asyncio.Lock()
        
    async def acquire(self, resource_id: str) -> bool:
        """
        Acquire a lock for a specific resource
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            True if lock was acquired, False otherwise
        """
        async with self._lock_creation_lock:
            if resource_id not in self._locks:
                self._locks[resource_id] = asyncio.Lock()
            
        # Acquire the lock with a timeout to prevent deadlocks
        try:
            await asyncio.wait_for(self._locks[resource_id].acquire(), timeout=30)
            return True
        except asyncio.TimeoutError:
            print(f"Timeout acquiring lock for resource {resource_id}")
            return False
            
    def release(self, resource_id: str):
        """
        Release a lock for a specific resource
        
        Args:
            resource_id: Resource identifier
        """
        if resource_id in self._locks and self._locks[resource_id].locked():
            self._locks[resource_id].release()
            
    async def __aenter__(self, resource_id: str):
        await self.acquire(resource_id)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

# Create a global async lock manager that can be shared
async_lock_manager = AsyncLock()