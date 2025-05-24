"""
Functions to handle media files (images and videos), including metadata extraction, collage creation, and more.
"""
import hashlib
import os
import shutil

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
from helpers import cli_friendly_logging

image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}


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
            logging.info(
                f"No creation date found for {video_path}, using modification time: {metadata['date'].strftime('%Y-%m-%d %H:%M:%S')}")
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


def import_media(source_dirs, target_dir, copy=False):
    """
    Import all media files from source directories to target directory.

    Args:
        source_dirs (list): List of directories to import media from
        target_dir (str): Target directory for media files
        copy (bool): Copy files instead of moving them

    Returns:
        dict: Summary of results
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True, parents=True)

    # Extensions for media files
    media_extensions = image_extensions.union(video_extensions)

    # Results tracking
    results = {
        'total_files': 0,
        'photos_imported': 0,
        'videos_imported': 0,
        'failed': 0,
        'skipped': 0
    }

    # Process each source directory
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        logging.info(f"Scanning directory: {source_path}")

        if not source_path.exists() or not source_path.is_dir():
            logging.error(f"Source directory not found or not a directory: {source_path}")
            continue

        # Find all media files in source directory (including subdirectories)
        all_files = []
        for root, _, files in os.walk(source_path):
            for filename in files:
                file_extension = Path(filename).suffix.lower()
                if file_extension in media_extensions:
                    file_path = Path(root) / filename
                    all_files.append((file_path, file_extension))

        logging.info(f"Found {len(all_files)} media files in {source_path}")

        # Process files with progress bar
        with cli_friendly_logging(all_files, desc=f"Importing from {source_path.name}", unit="file") as progress:
            for file_path, file_extension in progress:
                results['total_files'] += 1

                try:
                    # Destination path
                    dest_path = target_dir / file_path.name

                    # Handle duplicates by adding a suffix
                    if dest_path.exists():
                        base_name = dest_path.stem
                        extension = dest_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = target_dir / f"{base_name}_{counter}{extension}"
                            counter += 1

                        logging.debug(f"Renamed to avoid duplicate: {file_path.name} -> {dest_path.name}")

                    # Copy or move file
                    if copy:
                        shutil.copy2(file_path, dest_path)
                        logging.debug(f"Copied to {dest_path}")
                    else:
                        shutil.move(file_path, dest_path)
                        logging.debug(f"Moved to {dest_path}")

                    # Update results
                    if file_extension in image_extensions:
                        results['photos_imported'] += 1
                    elif file_extension in video_extensions:
                        results['videos_imported'] += 1

                except Exception as e:
                    logging.error(f"Failed to import {file_path}: {e}")
                    results['failed'] += 1

    return results


def identify_duplicates(directory, remove=False):
    """
    Identify duplicate files in a directory by comparing file content hashes.
    For image files, compares the actual image content ignoring metadata.
    For non-image files, compares the file content directly.

    Args:
        directory (str): Directory to scan for duplicates
        remove (bool): Whether to remove duplicate files

    Returns:
        dict: Summary of results including duplicate sets and stats
    """
    directory = Path(directory)

    # Results tracking
    results = {
        'total_files': 0,
        'unique_files': 0,
        'duplicate_sets': 0,
        'duplicate_files': 0,
        'deleted_files': 0,
        'duplicates': []  # Will contain lists of duplicate file paths
    }

    # Use a dictionary to track file hashes
    file_hashes = {}

    # Collect all media files recursively
    media_extensions = image_extensions.union(video_extensions)
    all_files = []

    logging.info(f"Scanning directory {directory} for media files...")
    for root, _, files in os.walk(directory):
        for filename in files:
            file_extension = Path(filename).suffix.lower()
            if file_extension in media_extensions:
                file_path = Path(root) / filename
                all_files.append(file_path)

    results['total_files'] = len(all_files)
    logging.info(f"Found {len(all_files)} media files to check for duplicates")

    # Process files with progress bar
    with cli_friendly_logging(all_files, desc="Hashing files", unit="file") as progress:
        for file_path in progress:
            try:
                file_extension = Path(file_path).suffix.lower()

                # For image files, use pixel data hash to ignore metadata
                if file_extension in image_extensions:
                    try:
                        # Load the image and convert to a standard format
                        with Image.open(file_path) as img:
                            # Convert to RGB to standardize format
                            if img.mode != 'RGB':
                                img = img.convert('RGB')

                            # Create a smaller version for faster comparison
                            img.thumbnail((100, 100))

                            # Generate hash from pixel data
                            hash_md5 = hashlib.md5()
                            hash_md5.update(img.tobytes())
                            image_hash = hash_md5.hexdigest()

                            # Use image dimensions + content hash as identifier
                            img_id = f"img_{img.width}x{img.height}_{image_hash}"

                            if img_id in file_hashes:
                                # Add to the list of duplicates
                                file_hashes[img_id]['paths'].append(file_path)
                            else:
                                # New unique image
                                file_hashes[img_id] = {
                                    'paths': [file_path],
                                    'is_duplicate': False
                                }
                            continue
                    except Exception as e:
                        logging.warning(f"Error processing image {file_path}, falling back to file hash: {e}")
                        # Fall back to file hash if image processing fails

                # For non-image files or if image processing failed, use file content hash
                file_size = os.path.getsize(file_path)

                with open(file_path, 'rb') as f:
                    # Read first chunk for quick comparison
                    chunk = f.read(8192)
                    quick_hash = hashlib.md5(chunk).hexdigest()

                # Use file size + first chunk hash as a preliminary identifier
                prelim_id = f"{file_size}_{quick_hash}"

                # If we find a potential match with the quick hash, do a full file hash
                if prelim_id in file_hashes:
                    # Full file hash for confirmation
                    hash_md5 = hashlib.md5()
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                    full_hash = hash_md5.hexdigest()

                    # Check if the full hash matches
                    if full_hash in file_hashes:
                        # Add to the list of duplicates
                        file_hashes[full_hash]['paths'].append(file_path)
                    else:
                        # New unique file with full hash
                        file_hashes[full_hash] = {
                            'size': file_size,
                            'paths': [file_path],
                            'is_duplicate': False
                        }
                else:
                    # No match with quick hash, store it
                    file_hashes[prelim_id] = {
                        'size': file_size,
                        'paths': [file_path],
                        'is_duplicate': False
                    }

            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

    # Find sets of duplicates (files with the same hash)
    duplicate_sets = []

    for file_hash, info in file_hashes.items():
        if len(info['paths']) > 1:
            info['is_duplicate'] = True
            duplicate_set = [str(path) for path in info['paths']]
            duplicate_sets.append(duplicate_set)
            results['duplicate_files'] += len(duplicate_set) - 1  # Count all but one as duplicates

    results['unique_files'] = results['total_files'] - results['duplicate_files']
    results['duplicate_sets'] = len(duplicate_sets)
    results['duplicates'] = duplicate_sets

    # Remove duplicates if requested
    if remove and duplicate_sets:
        logging.info(f"Removing {results['duplicate_files']} duplicate files...")
        deleted_count = 0

        for duplicate_set in duplicate_sets:
            # Keep the first file (original) and remove the rest
            original = duplicate_set[0]
            duplicates_to_remove = duplicate_set[1:]

            logging.info(f"Keeping: {original}")
            for duplicate in duplicates_to_remove:
                try:
                    logging.info(f"Removing duplicate: {duplicate}")
                    os.remove(duplicate)
                    deleted_count += 1
                except Exception as e:
                    logging.error(f"Error removing {duplicate}: {e}")

        results['deleted_files'] = deleted_count

    return results
