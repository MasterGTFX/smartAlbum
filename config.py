"""
Configuration settings for the photo organizer.
"""
import logging
import subprocess

# Default language for album names
DEFAULT_LANGUAGE = "Polish"

# Default system prompt for the AI assistant
DEFAULT_SYSTEM_PROMPT = (
    "You are a photo labeling assistant. Analyze the provided images and suggest "
    "a concise name based on the primary location(s). If no clear location is found, "
    "use the main theme/activity. Return only 2-4 words."
)

# Default user prompt for the AI assistant 
DEFAULT_USER_PROMPT = (
    "Based on the provided images, suggest a location-focused or thematic album name."
)

# Default trip prefixes used to identify trip albums
DEFAULT_TRIP_PREFIXES = ["Trip-", "Vacation-", "Event-"]

# Default collage parameters
DEFAULT_COLLAGE_MAX_WIDTH = 1920
DEFAULT_COLLAGE_MAX_HEIGHT = 1080
DEFAULT_COLLAGE_THRESHOLD = 128

# AI configuration
DEFAULT_AI_MODEL = "o4-mini"
DEFAULT_AI_BASE_URL = "https://api.openai.com/v1"

# Default gap days for trip merging
DEFAULT_MAX_GAP_DAYS = 3

# Check for FFmpeg (needed for video metadata)
FFMPEG_AVAILABLE = False
try:
    result = subprocess.run(['ffprobe', '-version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode == 0:
        FFMPEG_AVAILABLE = True
    else:
        logging.warning("FFmpeg not found. Video metadata extraction will be limited.")
except FileNotFoundError:
    logging.warning("FFmpeg not found. Video metadata extraction will be limited.")
