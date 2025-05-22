"""
Image Generation Service

This module provides functionality for image generation and variation
through DALL-E 3.
"""

import os
import datetime, time
import uuid
import base64
from typing import Optional, Dict, Any, Union, List
import tempfile
import requests

from fastapi import UploadFile
from utils import run_in_executor
from utils import execute_image_generation

class ImageService:
    """Service for handling image generation and variation"""
    
    def __init__(self, images_dir: str):
        """
        Initialize the image service
        
        Args:
            images_dir: Directory for storing generated images
        """
        self.images_dir = images_dir
        
    async def generate_image(
        self,
        prompt: str,
        uploaded_image: Optional[UploadFile] = None,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural",
    ) -> Dict[str, Any]:
        """
        Generate a new image or a variation of an existing image
        
        Args:
            prompt: Text prompt for image generation or guidance for variation
            uploaded_image: Optional image file for variation (if None, creates new image)
            size: Size of the output image
            quality: Quality of the output image
            style: Style of the output image
            
        Returns:
            Dictionary with image information
        """
        try:
            # Generate a unique filename for the result
            image_id = str(uuid.uuid4())
            filename = f"{image_id}.png"
            file_path = os.path.join(self.images_dir, filename)
            
            # Generate new image using the function from llm_helpers_azure
            result = await execute_image_generation(
                prompt=prompt,
                size=size,
                quality=quality,
                style=style
            )
            
            # Check if there was an error
            if "error" in result:
                return result
                
            # Process the result
            if "data" in result and len(result["data"]) > 0 and "url" in result["data"][0]:
                image_url = result["data"][0]["url"]
                
                # Save the image from URL
                success = await self._save_image(image_url, file_path)
                
                if not success:
                    return {"error": "Failed to save the generated image"}
                    
                # Create image URL (relative to the server)
                local_image_url = f"/images/{filename}"
                
                # Get the image data to prepare response
                image_data = result["data"][0]
                
                # Return image information
                response = {
                    "filename": filename,
                    "url": local_image_url,
                    "created": datetime.datetime.now().strftime("%d-%B-%Y, %H:%M"),
                }

                return response
            else:
                return {"error": "No image data received from API"}
                
        except Exception as e:
            print(f"Error in image generation: {e}")
            return {"error": str(e)}
    
    async def _save_image(self, image_url: str, file_path: str) -> bool:
        """
        Download and save an image from a URL
        
        Args:
            image_url: URL of the image to download
            file_path: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define a function to download and save the image
            def download_and_save():
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(file_path, "wb") as img_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            img_file.write(chunk)
                    return True
                else:
                    print(f"Failed to download image: {response.status_code}")
                    return False
            
            # Execute downloading in a thread pool
            return await run_in_executor(download_and_save)
        except Exception as e:
            print(f"Error saving image from URL: {e}")
            return False

# Create DocumentRAG instance with updated paths
image_handler = ImageService(images_dir="images")

# Function to get DocumentRAG for specific chat
def get_image_service() -> ImageService:
    """Get ImageService instance"""
    return image_handler