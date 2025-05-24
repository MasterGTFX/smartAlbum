# Smart Album

A smart command-line tool for organizing your photo and video collections into meaningful albums automatically.

## Overview

Smart Album analyzes your media files and organizes them into albums based on metadata such as dates, camera information, and location. It automatically detects trips and vacations (periods with higher-than-usual photo activity) and uses AI to generate meaningful names for these collections.

## Features

- Automatic organization of photos and videos into logical albums
- Trip detection based on photo frequency analysis
- AI-powered album naming for identified trips and events
- Support for various media types and metadata extraction
- Simple command-line interface

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/smart-album.git
   cd smart-album
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure API keys:
   ```
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key.

## Usage

Basic usage:
`python smart_album.py <path_to_media_directory>`

### Command-line Options

Smart Album offers various command-line options to customize its behavior:

#### Core Options:
- `input_dir` - Directory containing media files (required)
- `--output-dir`, `-o` - Output directory for albums (default: INPUT_DIR/Albums)
- `--copy`, `-c` - Copy files instead of moving them
- `--from`, `-f` - Import media files from SOURCE_DIR (and its subdirectories)

#### Import and Organization:
- `--revert` - Revert organization by moving files from albums back to root directory
- `--no-videos` - Skip video files and only process photos
- `--no-trip-detection` - Disable trip and event detection, use simple year-month organization
- `--max-gap-days` - Maximum number of days between trips to merge them
- `--remove-duplicates` - Identify and remove duplicate files based on content

#### Album Creation:
- `--no-collages` - Disable automatic collage creation for albums
- `--collage-threshold` - Maximum number of photos per collage

#### AI Options:
- `--no_ai` - Skip AI-based album renaming
- `--system-prompt` - Custom system prompt for the AI
- `--prompt` - Custom prompt for the AI
- `--language` - Language for album names (default: **English**)

### Examples

**Basic organization:**

Please be aware that when using AI-powered features, such as album renaming, the relevant photos (collages created from your albums) will be sent to the Large Language Model (LLM) of your choice for processing. 
Ensure you understand the privacy implications and use this feature wisely.

