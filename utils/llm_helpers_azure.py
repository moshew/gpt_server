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

import openai
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import AzureOpenAIEmbeddings
from utils import run_in_executor
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION
from config import DALLE_API_KEY, DALLE_ENDPOINT, DALLE_API_VERSION

# Model configurations for different use cases with Azure deployment names
MODEL_CONFIGS = {
    "default": {
        "deployment_name": "gpt-4.1",
        "temperature": 0.2,
        "max_tokens": 30000,
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
        "deployment_name": "o3-mini",
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

def get_openai_client():
    """
    Get or create a global Azure OpenAI client instance
    
    Returns:
        OpenAI client instance configured for Azure
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AzureOpenAI(
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
    Generate embeddings for a list of text documents using Azure
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = get_embeddings()
    return await run_in_executor(embeddings.embed_documents, texts)

async def embed_query(text: str) -> List[float]:
    """
    Generate embeddings for a single query text using Azure
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector
    """
    embeddings = get_embeddings()
    return await run_in_executor(embeddings.embed_query, text)

async def call_llm(
    prompt: str = None,
    messages: List[Dict[str, str]] = None,
    model_config: str = "default",
    custom_config: Optional[Dict[str, Any]] = None,
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    General-purpose LLM call function that supports different Azure configurations
    
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
            return _stream_messages_response(
                client=client,
                messages=messages,
                deployment_name=deployment_name,
                temperature=temperature if "temperature" in ALLOWED_PARAMS[model_type] else None,
                max_tokens=max_tokens,
                **{k: v for k, v in config.items() if k in ALLOWED_PARAMS[model_type]}
            )
        else:
            # For non-streaming, we can run in executor to prevent blocking
            def generate_response():
                response = client.chat.completions.create(**api_params)
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
        
        response_stream = client.chat.completions.create(**api_params)
        
        for chunk in response_stream:
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