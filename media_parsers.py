"""
Functions to handle media files (images and videos), including metadata extraction, collage creation, and more.
"""

import os
from PIL import ExifTags, Image
from datetime import datetime
from pathlib import Path
import logging
import subprocess
import json
import re
import numpy as np

from config import (
    DEFAULT_COLLAGE_MAX_WIDTH,
    DEFAULT_COLLAGE_MAX_HEIGHT, FFMPEG_AVAILABLE,
)


def extract_video_metadata(video_path):
    """
    Extract metadata from a video file.

    Args:
        video_path (str): Path to the video file

    Returns:
        dict: Dictionary containing metadata
    """
    metadata = {
        'filename': os.path.basename(video_path),
        'path': video_path,
        'date': None,
        'camera_model': None,
        'size': None,
        'duration': None,
        'type': 'video'
    }

    # Try to get creation date from filename (common in many cameras)
    # Patterns like: VID_20220115_123045.mp4 or VIDEO_20220115_123045.mp4
    filename = os.path.basename(video_path)
    date_patterns = [
        r'(?:VID|VIDEO)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',  # VID_YYYYMMDD_HHMMSS
        r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',  # YYYYMMDD_HHMMSS
        r'(\d{4})-(\d{2})-(\d{2})[\s_](\d{2})-(\d{2})-(\d{2})',  # YYYY-MM-DD HH-MM-SS
    ]

    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            try:
                date_str = f"{groups[0]}-{groups[1]}-{groups[2]} {groups[3]}:{groups[4]}:{groups[5]}"
                metadata['date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                break
            except (ValueError, IndexError):
                pass

    # If FFmpeg is available, try to extract more detailed metadata
    if FFMPEG_AVAILABLE:
        try:
            # Run ffprobe to get video metadata
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode == 0:
                # Parse JSON output
                info = json.loads(result.stdout)

                # Get video streams
                video_streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'video']
                if video_streams:
                    video_stream = video_streams[0]
                    if 'width' in video_stream and 'height' in video_stream:
                        metadata['size'] = (video_stream['width'], video_stream['height'])

                # Get creation date
                if not metadata['date']:
                    format_info = info.get('format', {})
                    # Try different date tags
                    for date_tag in ['creation_time', 'com.apple.quicktime.creationdate']:
                        if date_tag in format_info:
                            date_str = format_info[date_tag]
                            
                            # Try different date formats
                            date_formats = [
                                '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO format
                                '%Y-%m-%d %H:%M:%S',  # Standard format
                                '%Y:%m:%d %H:%M:%S'  # EXIF-like format
                            ]
                
                            for fmt in date_formats:
                                try:
                                    metadata['date'] = datetime.strptime(date_str, fmt)
                                    logging.debug(f"Parsed date {date_str} with format {fmt}")
                                    break
                                except ValueError:
                                    continue
                
                            if metadata['date']:
                                break
                    
                    # Handle tags separately
                    if not metadata['date'] and 'tags' in format_info and isinstance(format_info['tags'], dict):
                        tags = format_info['tags']
                        if 'creation_time' in tags:
                            date_str = tags['creation_time']
                            
                            # Try different date formats
                            date_formats = [
                                '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO format
                                '%Y-%m-%d %H:%M:%S',  # Standard format
                                '%Y:%m:%d %H:%M:%S'  # EXIF-like format
                            ]
                
                            for fmt in date_formats:
                                try:
                                    metadata['date'] = datetime.strptime(date_str, fmt)
                                    logging.debug(f"Parsed date from tags: {date_str} with format {fmt}")
                                    break
                                except ValueError:
                                    continue
                                except Exception as e:
                                    logging.warning(f"Unexpected error parsing date {date_str}: {e}")

                # Get duration
                if 'duration' in info.get('format', {}):
                    try:
                        metadata['duration'] = float(info['format']['duration'])
                    except (ValueError, TypeError):
                        pass

                # Try to get camera model from metadata
                if 'tags' in info.get('format', {}):
                    tags = info['format']['tags']
                    for key in ['model', 'com.apple.quicktime.model', 'make']:
                        if key in tags:
                            metadata['camera_model'] = tags[key]
                            break

        except Exception as e:
            logging.error(f"Error extracting metadata from video {video_path} using FFmpeg: {e}")
            logging.debug(f"FFmpeg command: {cmd}")
            if 'result' in locals() and hasattr(result, 'stderr'):
                stderr_output = result.stderr.decode('utf-8', errors='replace')
                if stderr_output:
                    logging.debug(f"FFmpeg stderr output: {stderr_output}")

    # Fall back to file modification time if no date found
    if not metadata['date']:
        try:
            mtime = os.path.getmtime(video_path)
            metadata['date'] = datetime.fromtimestamp(mtime)
            logging.info(f"No creation date found for {video_path}, using modification time: {metadata['date'].strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logging.error(f"Error getting modification time for {video_path}: {e}")

    return metadata


def extract_metadata(file_path):
    """
    Extract metadata from an image or video file.

    Args:
        file_path (str): Path to the media file

    Returns:
        dict: Dictionary containing metadata
    """
    # Determine file type by extension
    file_extension = Path(file_path).suffix.lower()

    # Define image and video extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}

    # Process based on file type
    if file_extension in image_extensions:
        # Extract image metadata
        metadata = {
            'filename': os.path.basename(file_path),
            'path': file_path,
            'date': None,
            'camera_model': None,
            'size': None,
            'type': 'image'
        }

        try:
            with Image.open(file_path) as img:
                # Get basic image info
                metadata['size'] = img.size

                # Extract EXIF data if available
                exif_data = img._getexif()
                if exif_data:
                    # Map EXIF tags to human-readable names
                    exif = {
                        ExifTags.TAGS[k]: v
                        for k, v in exif_data.items()
                        if k in ExifTags.TAGS
                    }

                    # Extract date
                    if 'DateTimeOriginal' in exif:
                        date_str = exif['DateTimeOriginal']
                        try:
                            metadata['date'] = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                        except ValueError:
                            logging.warning(f"Could not parse date: {date_str}")

                    # Extract camera model
                    if 'Model' in exif:
                        metadata['camera_model'] = exif['Model']

                # If no EXIF date, try filename patterns
                if not metadata['date']:
                    filename = os.path.basename(file_path)
                    # Common patterns: IMG_20220115_123045.jpg
                    date_patterns = [
                        r'(?:IMG|DSC|DSCN|DSCF)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
                        r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
                        r'(\d{4})-(\d{2})-(\d{2})[\s_](\d{2})-(\d{2})-(\d{2})',
                    ]

                    for pattern in date_patterns:
                        match = re.search(pattern, filename)
                        if match:
                            groups = match.groups()
                            try:
                                date_str = f"{groups[0]}-{groups[1]}-{groups[2]} {groups[3]}:{groups[4]}:{groups[5]}"
                                metadata['date'] = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                                break
                            except (ValueError, IndexError):
                                pass

                # Fall back to file modification time if no date found
                if not metadata['date']:
                    try:
                        mtime = os.path.getmtime(file_path)
                        metadata['date'] = datetime.fromtimestamp(mtime)
                        logging.info(f"No EXIF date found for {file_path}, using modification time: {metadata['date']}")
                    except Exception as e:
                        logging.error(f"Error getting modification time for {file_path}: {e}")

        except Exception as e:
            logging.error(f"Error extracting metadata from image {file_path}: {e}")

        return metadata

    elif file_extension in video_extensions:
        # Extract video metadata
        return extract_video_metadata(file_path)

    else:
        # Unsupported file type
        logging.warning(f"Unsupported file type: {file_extension} for {file_path}")
        return {
            'filename': os.path.basename(file_path),
            'path': file_path,
            'date': None,
            'camera_model': None,
            'size': None,
            'type': 'unknown'
        }


def create_collage(images, output_path, max_width=DEFAULT_COLLAGE_MAX_WIDTH, max_height=DEFAULT_COLLAGE_MAX_HEIGHT,
                   bg_color=(0, 0, 0)):
    """
    Create a collage from a list of images.

    Args:
        images (list): List of paths to image files
        output_path (str): Path to save the collage
        max_width (int): Maximum width of the collage
        max_height (int): Maximum height of the collage
        bg_color (tuple): Background color (R, G, B)

    Returns:
        bool: True if collage was created successfully, False otherwise
    """
    try:

        if not images:
            logging.warning("No images provided for collage")
            return False

        # Limit number of images to prevent huge collages
        images = images[:min(len(images), 256)]  # Hard limit on images in a single collage

        # Open and resize images
        img_objects = []
        for img_path in images:
            try:
                # Only process image files
                if not any(img_path.lower().endswith(ext) for ext in
                           ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']):
                    continue

                img = Image.open(img_path)

                # Convert to RGB mode if needed
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')

                # Resize to reasonable dimension while maintaining aspect ratio
                img.thumbnail((max_width // 8, max_height // 8))
                img_objects.append(img)
            except Exception as e:
                logging.warning(f"Could not process {img_path} for collage: {e}")

        if not img_objects:
            logging.warning("No valid images found for collage")
            return False

        # Determine grid size based on number of images
        n_images = len(img_objects)
        grid_size = int(np.ceil(np.sqrt(n_images)))

        # Calculate thumbnail size (all images will be resized to this)
        thumb_width = max_width // grid_size
        thumb_height = max_height // grid_size

        # Create blank canvas
        collage = Image.new('RGB', (max_width, max_height), bg_color)

        # Place images in grid
        for idx, img in enumerate(img_objects):
            if idx >= grid_size * grid_size:  # Safety check
                break

            # Calculate position
            row = idx // grid_size
            col = idx % grid_size

            # Resize image to thumbnail size while maintaining aspect ratio
            img_copy = img.copy()
            img_copy.thumbnail((thumb_width, thumb_height))

            # Calculate centered position
            x = col * thumb_width + (thumb_width - img_copy.width) // 2
            y = row * thumb_height + (thumb_height - img_copy.height) // 2

            # Paste image onto canvas
            collage.paste(img_copy, (x, y))

        # Save collage
        collage.save(output_path, quality=90)
        return True

    except Exception as e:
        logging.error(f"Error creating collage: {e}")
        return False