"""
LLM Helper Utilities for Azure AI Services

This module contains generic utilities designed to work with Large Language Models (LLMs)
through Azure AI Services for various tasks like generating responses or handling specialized outputs.
It supports both direct API calls and integration with langchain.
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List, Union
import requests, json
import tempfile
import os
import logging

import openai
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import AzureOpenAIEmbeddings
from .async_helpers import run_in_executor
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION
from config import DALLE_API_KEY, DALLE_ENDPOINT, DALLE_API_VERSION
from config import MAX_CONCURRENT_LLM_CALLS, MAX_CONCURRENT_EMBEDDING_CALLS

# Configure logging
logger = logging.getLogger("llm_helpers")

# Model configurations for different use cases with Azure deployment names
MODEL_CONFIGS = {
    "default": {
        "deployment_name": "gpt-4.1",
        "temperature": 0.2,
    },
    "code": {
        "deployment_name": "gpt-4.1",
        "temperature": 0.1,
        "max_tokens": 4000,
    },
    "fast": {
        "deployment_name": "gpt-4.1",
        "temperature": 0.2,
        "max_tokens": 2000,
    },
    "precise": {
        "deployment_name": "gpt-4.1",
        "temperature": 0.1,
        "max_tokens": 500,
    },
    "mini": {
        "deployment_name": "gpt-4.1",
        "temperature": 0.3,
        "max_tokens": 1000,
    }
}

# Allowed parameters for specific models
ALLOWED_PARAMS = {
    "o3-mini": ["model", "messages", "max_completion_tokens", "stream"],
    "default": ["model", "messages", "max_completion_tokens", "temperature", "stream"]
}

# Global instances for reuse
_embeddings = None
_openai_client = None
_dalle_client = None

# Rate limiting semaphore to prevent too many concurrent requests
# These can be adjusted based on your OpenAI rate limits
_openai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
_embedding_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDING_CALLS)

def get_openai_client():
    """
    Get or create a global Azure OpenAI client instance
    
    Returns:
        AsyncAzureOpenAI client instance configured for Azure
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncAzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
    return _openai_client

def get_embeddings():
    """
    Get or create a global embeddings instance for Azure
    
    Returns:
        AzureOpenAIEmbeddings instance
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-large",  # Azure deployment name for embeddings
            openai_api_key=AZURE_API_KEY,
            openai_api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
    return _embeddings

async def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text documents using Azure with rate limiting
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    # Check if we need to wait for rate limiting
    if _embedding_semaphore.locked():
        logger.info(f"Embedding rate limiting active - waiting for available slot for {len(texts)} documents")
    
    async with _embedding_semaphore:  # Rate limiting for embeddings
        logger.debug(f"Acquired embedding semaphore slot for {len(texts)} documents")
        try:
            embeddings = get_embeddings()
            return await run_in_executor(embeddings.embed_documents, texts)
        finally:
            logger.debug("Released embedding semaphore slot")

async def embed_query(text: str) -> List[float]:
    """
    Generate embeddings for a single query text using Azure with rate limiting
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector
    """
    # Check if we need to wait for rate limiting
    if _embedding_semaphore.locked():
        logger.info("Embedding rate limiting active - waiting for available slot for query")
    
    async with _embedding_semaphore:  # Rate limiting for embeddings
        logger.debug("Acquired embedding semaphore slot for query")
        try:
            embeddings = get_embeddings()
            return await run_in_executor(embeddings.embed_query, text)
        finally:
            logger.debug("Released embedding semaphore slot")

async def call_llm(
    prompt: str = None,
    messages: List[Dict[str, str]] = None,
    model_config: str = "default",
    custom_config: Optional[Dict[str, Any]] = None,
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Call an LLM with Azure AI Services with rate limiting
    
    Args:
        prompt: Text prompt to send to the LLM
        messages: List of message dictionaries (alternative to prompt)
        model_config: Configuration preset to use ("default", "code", "fast", "precise", "mini")
        custom_config: Custom configuration parameters to override defaults
        stream: Whether to stream the response
        
    Returns:
        LLM response text or async generator for streaming
    """
    
    # Check if we need to wait for rate limiting
    if _openai_semaphore.locked():
        logger.info("Rate limiting active - waiting for available slot for LLM call")
    
    async with _openai_semaphore:  # Rate limiting
        logger.debug("Acquired LLM semaphore slot")
        try:
            # Get the base configuration
            config = MODEL_CONFIGS.get(model_config, MODEL_CONFIGS["default"]).copy()
            
            # Apply any custom configuration parameters
            if custom_config:
                config.update(custom_config)
                
            # Extract parameters
            deployment_name = config.pop("deployment_name", "gpt-4.1")
            temperature = config.pop("temperature", 0.2)
            max_tokens = config.pop("max_tokens", 8000)
            
            # Get the Azure OpenAI client
            client = get_openai_client()
            
            # Prepare messages
            if messages is None:
                if prompt is None:
                    raise ValueError("Either prompt or messages must be provided")
                messages = [{"role": "user", "content": prompt}]
            
            # Determine which model we're using to filter allowed parameters
            model_type = "default"
            if "o3-mini" in deployment_name:
                model_type = "o3-mini"
            
            # Prepare API parameters with only allowed parameters for this model
            api_params = {
                "model": deployment_name,
                "messages": messages,
                "max_completion_tokens": max_tokens,
            }
            
            # Only add temperature if this model supports it
            if "temperature" in ALLOWED_PARAMS[model_type]:
                api_params["temperature"] = temperature
                
            # Add streaming parameter
            api_params["stream"] = stream
            
            if stream:
                # For streaming, return the async generator
                return _stream_messages_response(client, messages, deployment_name, temperature, max_tokens, **config)
            else:
                # For non-streaming, get the complete response
                response = await client.chat.completions.create(**api_params)
                return response.choices[0].message.content
                
        except Exception as e:
            error_msg = f"Error calling LLM: {str(e)}"
            logger.error(error_msg)
            if stream:
                async def error_generator():
                    yield error_msg
                return error_generator()
            else:
                return error_msg
        finally:
            logger.debug("Released LLM semaphore slot")

async def _stream_messages_response(
    client, messages: List[Dict[str, str]], deployment_name: str, temperature: float, max_tokens: int, **kwargs
) -> AsyncGenerator[str, None]:
    """
    Internal helper function for streaming LLM responses from messages through Azure
    """
    try:
        # Determine which model we're using to filter allowed parameters
        model_type = "default"
        if "o3-mini" in deployment_name:
            model_type = "o3-mini"
            
        # Prepare API parameters
        api_params = {
            "model": deployment_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": True
        }
        
        # Add temperature only if it's supported by this model
        if temperature is not None and "temperature" in ALLOWED_PARAMS[model_type]:
            api_params["temperature"] = temperature
        
        # Add any additional parameters that are allowed for this model
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_PARAMS[model_type]}
        api_params.update(filtered_kwargs)
        
        response_stream = await client.chat.completions.create(**api_params)
        
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                await asyncio.sleep(0)  # Allow other tasks to run
    except Exception as e:
        yield f"Error streaming response: {str(e)}"

async def process_json_response(prompt: str, expected_keys: List[str] = None) -> Dict[str, Any]:
    """
    Process a response expected to be in JSON format using Azure
    
    Args:
        prompt: The prompt to send to the model
        expected_keys: List of keys expected in the JSON response
        
    Returns:
        Parsed JSON as a dictionary
    """
    try:
        # Add instructions for JSON format - specifically request an object, not an array
        json_prompt = f"{prompt}\n\nRespond with a valid JSON object (not an array), with no additional text, markdown formatting, or code blocks."
        if expected_keys:
            json_prompt += f" Include these keys in your response: {', '.join(expected_keys)}."
            
        response = await call_llm(
            prompt=json_prompt,
            custom_config={"response_format": {"type": "json_object"}}
        )
        
        # Clean up the response to handle possible markdown formatting
        cleaned_response = response
        
        # Remove markdown code block formatting if present
        if cleaned_response.startswith("```"):
            # Find the first and last backtick sections
            first_backticks_end = cleaned_response.find("\n")
            last_backticks_start = cleaned_response.rfind("```")
            
            if first_backticks_end > 0 and last_backticks_start > first_backticks_end:
                # Extract only the content between the backtick sections
                cleaned_response = cleaned_response[first_backticks_end+1:last_backticks_start].strip()
        
        # Parse the JSON response
        parsed_json = json.loads(cleaned_response)
        
        # Handle the case where the model returns an array instead of an object
        if isinstance(parsed_json, list):
            # Convert array to object with a container key
            result = {"items": parsed_json}
        else:
            result = parsed_json
        
        # Validate expected keys if provided
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                print(f"Warning: Missing expected keys in JSON response: {missing_keys}")
        
        return result
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Raw response was: {response}")
        return {}
    except Exception as e:
        print(f"Error in process_json_response: {e}")
        return {}

async def process_langchain_messages(
    messages: List[BaseMessage],
    model_config: str = "default",
    custom_config: Optional[Dict[str, Any]] = None,
    stream: bool = True,
    deployment_name: Optional[str] = None
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Process a list of langchain messages using the configured Azure LLM
    
    Args:
        messages: List of langchain message objects
        model_config: Name of a predefined configuration
        custom_config: Custom configuration parameters
        stream: Whether to stream the response
        deployment_name: Optional specific model deployment name to use (overrides model_config)
        
    Returns:
        Response from the LLM either as a string or a generator
    """
    # Convert langchain messages to OpenAI format
    openai_messages = []
    
    for msg in messages:
        if isinstance(msg, SystemMessage):
            openai_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            openai_messages.append({"role": "assistant", "content": msg.content})
        # Skip other message types
    
    # Create or update custom config with deployment_name if provided
    if deployment_name:
        # Convert deployment_name to lowercase
        deployment_name = deployment_name.lower()
        if custom_config is None:
            custom_config = {"deployment_name": deployment_name}
        else:
            custom_config = {**custom_config, "deployment_name": deployment_name}
    
    # Use the general-purpose call_llm function with the converted messages
    return await call_llm(
        messages=openai_messages,
        model_config=model_config,
        custom_config=custom_config,
        stream=stream
    )

async def execute_image_generation(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "natural",
    n: int = 1,
) -> Dict[str, Any]:
    """
    Generate a new image using DALL-E 3 through Azure AI Services
    
    Args:
        prompt: Text prompt for image generation
        size: Output image size (1024x1024, 1792x1024, or 1024x1792)
        quality: Output image quality (standard or hd)
        style: Output image style (natural or vivid)
        n: Number of images to generate
        
    Returns:
        Dictionary with generated image data
    """
    
    try:
        # Define the endpoint URL
        url = f"{DALLE_ENDPOINT}openai/deployments/dall-e-3/images/generations?api-version={DALLE_API_VERSION}"
        
        # Set headers
        headers = {
            "Content-Type": "application/json",
            "api-key": DALLE_API_KEY
        }
        
        # Prepare the payload
        payload = {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "style": style,
            "n": n,
        }
        
        # Define a synchronous function to run with run_in_executor
        def make_request():
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
                
        # Execute the synchronous function within our async context
        result = await run_in_executor(make_request)
        print(result)
        
        # Return the result
        return result
        
    except Exception as e:
        e_str = str(e)
        print(f"Error in execute_image_generation: {e_str}")
        if "ResponsibleAIPolicyViolation" in e_str:
            e_str = "Your request was rejected as a result of our safety system. Your prompt may contain text that is not allowed by our safety system."
        return {"error": e_str}

def get_rate_limit_status() -> Dict[str, Any]:
    """
    Get current rate limiting status
    
    Returns:
        Dictionary with semaphore status information
    """
    return {
        "llm_calls": {
            "max_concurrent": MAX_CONCURRENT_LLM_CALLS,
            "available_slots": _openai_semaphore._value,
            "in_use": MAX_CONCURRENT_LLM_CALLS - _openai_semaphore._value,
            "locked": _openai_semaphore.locked()
        },
        "embedding_calls": {
            "max_concurrent": MAX_CONCURRENT_EMBEDDING_CALLS,
            "available_slots": _embedding_semaphore._value,
            "in_use": MAX_CONCURRENT_EMBEDDING_CALLS - _embedding_semaphore._value,
            "locked": _embedding_semaphore.locked()
        }
    }