"""
AI module for interacting with language models to generate content for the media organizer.
"""
import os
import re
from pathlib import Path
from typing import Optional, List
import logging
from openai import OpenAI
from dotenv import load_dotenv
from config import DEFAULT_AI_MODEL, DEFAULT_TRIP_PREFIXES, DEFAULT_AI_BASE_URL
from helpers import image_to_base64, cli_friendly_logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set up AI API key
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    logging.error("OPENAI_API_KEY not found in environment variables.")
    logging.info("Please set the OPENAI_API_KEY environment variable.")
    raise ValueError("OPENAI_API_KEY not found. See logs for details.")

# Get AI model from environment or use default from config
AI_MODEL = os.environ.get("AI_MODEL", DEFAULT_AI_MODEL)

# Get base URL from environment or use default
AI_BASE_URL = os.environ.get("AI_BASE_URL", DEFAULT_AI_BASE_URL)

# Initialize OpenAI client
OPENAI_CLIENT = OpenAI(api_key=API_KEY,
                       base_url=AI_BASE_URL)


def ask_llm(prompt: str,
            system_prompt: str,
            images: Optional[List[str]] = None) -> str | None:
    """
    Send a prompt to a language model and return the response.
    
    Args:
        prompt (str): The prompt to send to the language model
        system_prompt (str): The system prompt to set the context
        images (List[str], optional): List of base64-encoded images to include in the prompt
        
    Returns:
        str: The generated text response
    """
    try:
        # Log the number of images being processed
        if images:
            logging.info(f"Processing request with {len(images)} image(s)")

        # Prepare messages in new format
        input_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + [
                    {"type": "image_url", "image_url": {"url": image}}
                    for image in (images or [])
                ]
            }
        ]

        # Create response from the LLM
        completion = OPENAI_CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=input_messages
        )

        # Extract the text response
        if completion and completion.choices:
            text_response = completion.choices[0].message.content
            logging.info(f"AI response: {text_response}")
            return text_response
        else:
            logging.error("No response from AI or empty choices.")
            return None



    except Exception as e:
        logging.error(f"Error calling LLM API: {e}")
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            logging.error("This appears to be an authentication error. Please check your API key.")
        return f"Error: {str(e)}"


def rename_albums_with_ai(output_dir, system_prompt, prompt,
                          language='English'):
    """
    Rename albums using AI based on collage images.

    Args:
        output_dir (Path): Directory containing albums
        system_prompt (str, optional): Custom system prompt for the AI
        prompt (str, optional): Custom prompt for the AI
        language (str): Language for album names (default: English)

    Returns:
        dict: Summary of renaming results
    """
    results = {"renamed": 0, "failed": 0, "skipped": 0}

    # Override query if language is not English
    if language != 'English':
        system_prompt += f"\nAlbum name should be in **{language}**."

    # Find all album directories that appear to be trip-related
    trip_albums = []
    for album_dir in output_dir.iterdir():
        if not album_dir.is_dir():
            continue

        # Check if this is a trip/vacation album by name pattern
        if any(album_dir.name.startswith(prefix) for prefix in DEFAULT_TRIP_PREFIXES):
            # Check if it has a collage
            collages = list(album_dir.glob("collage*.jpg"))
            if collages:
                trip_albums.append((album_dir, collages))

    if not trip_albums:
        logging.info("No trip albums with collages found to rename")
        return results

    logging.info(f"Found {len(trip_albums)} trip albums with collages to rename")

    # Process each trip album
    for album_dir, collages in cli_friendly_logging(trip_albums, desc="Renaming albums with AI", unit="album"):
        try:
            # Convert all collages to base64
            base64_images = []
            for collage_path in collages:
                base64_image = image_to_base64(str(collage_path))
                if base64_image:
                    base64_images.append(base64_image)

            if not base64_images:
                logging.error(f"Failed to convert any collages to base64 for album: {album_dir.name}")
                results["failed"] += 1
                continue

            # Call LLM API with all collages
            logging.info(
                f"Asking AI for album name suggestion for: {album_dir.name} (using {len(base64_images)} collages)")
            ai_response = ask_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                images=base64_images
            )

            if not ai_response:
                logging.error(f"Empty response from AI for album: {album_dir.name}")
                results["failed"] += 1
                continue

            # Clean up AI response
            ai_response = ai_response.strip().strip('"\'.,')

            # Extract date part from original album name
            original_name = album_dir.name
            date_part = ""
            if "-" in original_name:
                date_part = original_name.split("-", 1)[1].replace('_to_', '-')

            # Create new album name (preserve the date part)
            if date_part:
                new_name = f"{ai_response} ({date_part})"
            else:
                new_name = f"{ai_response}"

            # Replace invalid characters
            new_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)

            # Rename the directory
            new_path = album_dir.parent / new_name

            # If the new path already exists, add a suffix
            if new_path.exists():
                suffix = 1
                while (album_dir.parent / f"{new_name}_{suffix}").exists():
                    suffix += 1
                new_path = album_dir.parent / f"{new_name}_{suffix}"

            album_dir.rename(new_path)
            logging.info(f"Renamed: {original_name} → {new_path.name}")
            results["renamed"] += 1

        except Exception as e:
            logging.error(f"Error renaming album {album_dir}: {e}")
            results["failed"] += 1

    return results
