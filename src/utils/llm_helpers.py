"""
LLM Helper Utilities

This module contains generic utilities designed to work with Large Language Models (LLMs)
for various tasks like generating responses or handling specialized outputs.
It supports both direct API calls and integration with langchain.
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List, Union
import json

import openai
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import OpenAIEmbeddings
from utils import run_in_executor
from config import OPENAI_API_KEY

# Model configurations for different use cases
MODEL_CONFIGS = {
    "default": {
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 8000,
    },
    "code": {
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 4000,
    },
    "fast": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "max_tokens": 2000,
    },
    "precise": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 500,
    }
}

# Global instances for reuse
_embeddings = None
_openai_client = None

def _get_openai_client():
    """
    Get or create a global OpenAI client instance
    
    Returns:
        OpenAI client instance
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

def _get_embeddings():
    """
    Get or create a global embeddings instance
    
    Returns:
        OpenAIEmbeddings instance
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return _embeddings

async def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text documents
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = _get_embeddings()
    return await run_in_executor(embeddings.embed_documents, texts)

async def embed_query(text: str) -> List[float]:
    """
    Generate embeddings for a single query text
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector
    """
    embeddings = _get_embeddings()
    return await run_in_executor(embeddings.embed_query, text)

async def call_llm(
    prompt: str = None,
    messages: List[Dict[str, str]] = None,
    model_config: str = "default",
    custom_config: Optional[Dict[str, Any]] = None,
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    General-purpose LLM call function that supports different configurations
    
    Args:
        prompt: The prompt to send to the model (single message)
        messages: List of message objects in OpenAI format (if provided, overrides prompt)
        model_config: Name of a predefined configuration or "custom"
        custom_config: Custom configuration parameters (overrides predefined config)
        stream: Whether to stream the response
        
    Returns:
        Either the complete response as a string (stream=False) or 
        a generator that yields text chunks (stream=True)
    """
    try:
        # Get the base configuration
        config = MODEL_CONFIGS.get(model_config, MODEL_CONFIGS["default"]).copy()
        
        # Apply any custom configuration parameters
        if custom_config:
            config.update(custom_config)
            
        # Extract parameters
        model = config.pop("model", "gpt-4o")
        temperature = config.pop("temperature", 0.2)
        max_tokens = config.pop("max_tokens", 8000)
        
        # Get the OpenAI client
        client = _get_openai_client()
        
        # Prepare messages
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        
        if stream:
            return _stream_messages_response(
                client=client,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **config
            )
        else:
            # For non-streaming, we can run in executor to prevent blocking
            def generate_response():
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **config
                )
                return response.choices[0].message.content
                
            return await run_in_executor(generate_response)
            
    except Exception as e:
        print(f"Error in call_llm: {e}")
        if stream:
            async def error_generator():
                yield f"Error generating response: {str(e)}"
            return error_generator()
        return f"Error generating response: {str(e)}"

async def _stream_messages_response(
    client, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, **kwargs
) -> AsyncGenerator[str, None]:
    """
    Internal helper function for streaming LLM responses from messages
    """
    try:
        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                await asyncio.sleep(0)  # Allow other tasks to run
    except Exception as e:
        yield f"Error streaming response: {str(e)}"

async def process_json_response(prompt: str, expected_keys: List[str] = None) -> Dict[str, Any]:
    """
    Process a response expected to be in JSON format
    
    Args:
        prompt: The prompt to send to the model
        expected_keys: List of keys expected in the JSON response
        
    Returns:
        Parsed JSON as a dictionary
    """
    try:
        # Add instructions for JSON format
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, with no additional text."
        if expected_keys:
            json_prompt += f" Include these keys in your response: {', '.join(expected_keys)}."
            
        response = await call_llm(
            prompt=json_prompt,
            custom_config={"response_format": {"type": "json_object"}}
        )
        
        # Parse the JSON response
        result = json.loads(response)
        
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
    stream: bool = True
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Process a list of langchain messages using the configured LLM
    
    Args:
        messages: List of langchain message objects
        model_config: Name of a predefined configuration
        custom_config: Custom configuration parameters
        stream: Whether to stream the response
        
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
    
    # Use the general-purpose call_llm function with the converted messages
    return await call_llm(
        messages=openai_messages,
        model_config=model_config,
        custom_config=custom_config,
        stream=stream
    )