"""
Helpers module for various utility functions.
"""

import base64

from tqdm import tqdm
from pathlib import Path
import logging


def image_to_base64(image_path):
    """
    Convert an image file to base64 string.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Base64 encoded image string with data URI prefix
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # Add the appropriate data URI prefix for the image type
            file_ext = Path(image_path).suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }.get(file_ext, 'image/jpeg')
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logging.error(f"Error converting image to base64: {e}")
        return None


def cli_friendly_logging(iterable=None, desc=None, total=None, unit='it'):
    """
    Provides CLI-friendly progress reporting.

    This function has two modes:
    1. When passed an iterable, it wraps it with a progress bar
    2. When called with just a description, it returns a context manager that displays a spinner

    Args:
        iterable: The iterable to process (or None for message-only mode)
        desc (str, optional): Description for the progress bar or status message
        total (int, optional): Total number of items (calculated automatically if None)
        unit (str, optional): Unit name for items

    Returns:
        tqdm-wrapped iterable with progress bar or a context manager for status messages
    """
    if iterable is None:
        # Create a spinner for operations where we can't measure progress
        return tqdm(
            total=0,  # Indeterminate progress
            desc=desc,
            bar_format='{desc}',
            ncols=80,
            position=0,
            leave=True
        )
    else:
        # Normal progress bar for iterables
        if total is None and hasattr(iterable, '__len__'):
            total = len(iterable)

        return tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )


def status_message(message):
    """
    Display a status message with a spinner for operations without measurable progress.

    Args:
        message (str): The status message to display

    Returns:
        A context manager that will show a spinner while the operation is in progress
    """
    spinner = tqdm(
        total=0,  # Indeterminate progress
        desc=message,
        bar_format='{desc}',
        ncols=80,
        leave=True
    )
    return spinner
