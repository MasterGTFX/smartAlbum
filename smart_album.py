"""
Smart Album - A CLI tool to organize photos and videos into albums based on metadata and AI.
"""

import os
import argparse
from datetime import datetime, timedelta
import shutil
from pathlib import Path
import logging
from collections import Counter, defaultdict
import statistics
from itertools import groupby
from operator import itemgetter


from ai import rename_albums_with_ai
from config import (
    DEFAULT_LANGUAGE,
    DEFAULT_COLLAGE_THRESHOLD,
    DEFAULT_MAX_GAP_DAYS, DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
)
from helpers import cli_friendly_logging
from media_parsers import extract_metadata, create_collage


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def analyze_date_distribution(photo_dates, max_gap_days=3):
    """
    Analyze the distribution of photo dates to identify potential trips or events.
    
    Args:
        photo_dates (list): List of datetime objects representing photo dates
        max_gap_days (int): Maximum number of days between trips to merge them
        
    Returns:
        dict: Information about the date distribution and detected trips
    """
    if not photo_dates:
        return {'trips': []}

    # Sort dates
    sorted_dates = sorted(photo_dates)

    # Count photos per day
    daily_counts = Counter([date.date() for date in sorted_dates if date])

    # Calculate statistics for daily photo counts
    if len(daily_counts) > 0:
        counts = list(daily_counts.values())
        mean_count = statistics.mean(counts)
        try:
            stdev_count = statistics.stdev(counts)
        except statistics.StatisticsError:
            stdev_count = 0
    else:
        mean_count = 0
        stdev_count = 0

    # Lower threshold for considering a day as part of a trip (mean + 0.5 stdev or at least 3 photos)
    threshold = max(mean_count + (0.5 * stdev_count), 3)

    # Find days with photo counts above threshold
    trip_days = [day for day, count in daily_counts.items() if count >= threshold]

    # Group consecutive days into trips
    trips = []
    for k, g in groupby(enumerate(sorted(trip_days)), lambda x: x[0] - x[1].toordinal()):
        group = list(map(itemgetter(1), g))
        if len(group) >= 1:  # Consider even a single high-density day as a potential event
            start_date = group[0]
            end_date = group[-1]

            # Expand trip by one day on each side to capture related photos
            start_date = start_date - timedelta(days=1)
            end_date = end_date + timedelta(days=1)

            # Count photos in this trip
            trip_photos = sum(daily_counts[day] for day in daily_counts
                              if start_date <= day <= end_date)

            # Determine trip type based on duration
            duration = (end_date - start_date).days + 1
            if duration > 7:
                trip_type = "Vacation"
            elif duration > 2:
                trip_type = "Trip"
            else:
                trip_type = "Event"

            trips.append({
                'start_date': start_date,
                'end_date': end_date,
                'duration': duration,
                'photo_count': trip_photos,
                'type': trip_type
            })

    # Merge trips that are close to each other
    if trips and max_gap_days > 0:
        trips = merge_close_trips(trips, max_gap_days)

    return {
        'mean_photos_per_day': mean_count,
        'stdev_photos_per_day': stdev_count,
        'threshold': threshold,
        'trips': trips,
        'daily_counts': daily_counts
    }


def merge_close_trips(trips, max_gap_days):
    """
    Merge trips that are close to each other based on the maximum gap days.
    
    Args:
        trips (list): List of detected trips
        max_gap_days (int): Maximum number of days between trips to merge them
        
    Returns:
        list: List of merged trips
    """
    if not trips or len(trips) <= 1:
        return trips

    # Sort trips by start date
    sorted_trips = sorted(trips, key=lambda x: x['start_date'])

    # Initialize merged trips with the first trip
    merged_trips = [sorted_trips[0]]

    # Merge trips with small gaps
    for current_trip in sorted_trips[1:]:
        previous_trip = merged_trips[-1]

        # Calculate the gap between trips
        gap_days = (current_trip['start_date'] - previous_trip['end_date']).days

        # If the gap is small enough, merge the trips
        if gap_days <= max_gap_days:
            # Update end date if the current trip ends later
            if current_trip['end_date'] > previous_trip['end_date']:
                previous_trip['end_date'] = current_trip['end_date']

            # Combine photo counts
            previous_trip['photo_count'] += current_trip['photo_count']

            # Recalculate duration
            previous_trip['duration'] = (previous_trip['end_date'] - previous_trip['start_date']).days + 1

            # Update trip type based on new duration
            if previous_trip['duration'] > 7:
                previous_trip['type'] = "Vacation"
            elif previous_trip['duration'] > 2:
                previous_trip['type'] = "Trip"
            else:
                previous_trip['type'] = "Event"
        else:
            # If gap is too large, add as a separate trip
            merged_trips.append(current_trip)

    return merged_trips


def is_photo_in_trip(date, trips):
    """
    Check if a photo date falls within any detected trips.
    
    Args:
        date (datetime): The photo date
        trips (list): List of detected trips
        
    Returns:
        dict or None: Trip info if photo is in a trip, None otherwise
    """
    if not date:
        return None

    photo_date = date.date()

    for trip in trips:
        if trip['start_date'] <= photo_date <= trip['end_date']:
            return trip

    return None


def determine_album(metadata, trips=None):
    """
    Determine album name based on media metadata and trip detection.
    
    Args:
        metadata (dict): Media metadata
        trips (list, optional): List of detected trips
        
    Returns:
        str: Album name
    """
    if not metadata['date']:
        # If we have camera model but no date
        if metadata['camera_model']:
            return f"Camera-{metadata['camera_model'].replace(' ', '_')}"
        # Default album for media with no usable metadata
        else:
            if metadata['type'] == 'video':
                return "Unsorted-Videos"
            else:
                return "Unsorted"

    # Check if the media is part of a trip
    if trips:
        trip = is_photo_in_trip(metadata['date'], trips)
        if trip:
            start_str = trip['start_date'].strftime('%Y-%m-%d')
            end_str = trip['end_date'].strftime('%Y-%m-%d')
            if start_str == end_str:
                date_range = start_str
            else:
                date_range = f"{start_str}_to_{end_str}"
            return f"{trip['type']}-{date_range}"

    # Default: organize by year-month (same for both photos and videos)
    return metadata['date'].strftime('%Y-%m')


def process_media(input_dir, output_dir=None, copy=False, trip_detection=True, max_gap_days=DEFAULT_MAX_GAP_DAYS,
                  include_videos=True, create_collages=True, collage_threshold=DEFAULT_COLLAGE_THRESHOLD,
                  dont_use_ai=False, system_prompt=None, prompt=None, language=DEFAULT_LANGUAGE):
    """
    Process photos and videos in a directory and organize them into albums.
    
    Args:
        input_dir (str): Directory containing media files
        output_dir (str, optional): Directory for organized albums
        copy (bool): Copy files instead of moving them
        trip_detection (bool): Whether to detect trips and events
        max_gap_days (int): Maximum number of days between trips to merge them
        include_videos (bool): Whether to include video files
        create_collages (bool): Whether to create collages for each album
        collage_threshold (int): Maximum number of photos per collage
        dont_use_ai (bool): Flag to skip AI renaming
        system_prompt (str, optional): Custom system prompt for the AI
        prompt (str, optional): Custom prompt for the AI
        language (str): Language for album names (default: English)
    
    Returns:
        dict: Summary of processing results
    """
    input_dir = Path(input_dir)

    # If no output directory specified, create 'Albums' in the input directory
    if output_dir is None:
        output_dir = input_dir / 'Albums'
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Extensions for media files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}

    # Combine extensions based on include_videos setting
    if include_videos:
        media_extensions = image_extensions.union(video_extensions)
    else:
        media_extensions = image_extensions

    # Results tracking
    results = {
        'total_files': 0,
        'total_photos': 0,
        'total_videos': 0,
        'processed': 0,
        'failed': 0,
        'albums': {},
        'trips_detected': 0,
        'collages_created': 0
    }

    # First pass: Collect all media metadata for trip detection
    logging.info("First pass: Collecting media metadata for analysis...")
    media_metadata = []
    media_dates = []
    media_paths = []

    # Collect files first
    all_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            file_extension = Path(filename).suffix.lower()
            if file_extension in media_extensions:
                file_path = Path(root) / filename

                # Skip files in the output directory
                if str(output_dir) in str(file_path):
                    continue

                all_files.append((file_path, file_extension))

    # Process files with progress bar
    with cli_friendly_logging(all_files, desc="Collecting metadata", unit="file") as progress:
        for file_path, file_extension in progress:
            results['total_files'] += 1
            if file_extension in image_extensions:
                results['total_photos'] += 1
            elif file_extension in video_extensions:
                results['total_videos'] += 1

            try:
                metadata = extract_metadata(file_path)
                media_metadata.append(metadata)
                media_paths.append(file_path)
                if metadata['date']:
                    media_dates.append(metadata['date'])
            except Exception as e:
                logging.error(f"Failed to process {file_path} in first pass: {e}")
                results['failed'] += 1

    # Detect trips if enabled and we have enough media files with dates
    trips = []
    if trip_detection and len(media_dates) >= 3:  # Lower minimum media count
        logging.info("Analyzing media date distribution to detect trips and events...")
        date_analysis = analyze_date_distribution(media_dates, max_gap_days)
        trips = date_analysis['trips']
        daily_counts = date_analysis.get('daily_counts', Counter())

        if trips:
            results['trips_detected'] = len(trips)
            logging.info(f"Detected {len(trips)} trips/events (with max gap of {max_gap_days} days):")
            for idx, trip in enumerate(trips, 1):
                start_str = trip['start_date'].strftime('%Y-%m-%d')
                end_str = trip['end_date'].strftime('%Y-%m-%d')
                # Calculate peak days (days with most media)
                peak_days = sorted(
                    [(day, count) for day, count in daily_counts.items()
                     if trip['start_date'] <= day <= trip['end_date']],
                    key=lambda x: x[1], reverse=True
                )[:3]
                peak_days_str = ", ".join([f"{day.strftime('%Y-%m-%d')}: {count}" for day, count in peak_days])

                logging.info(f"  {idx}. {trip['type']} from {start_str} to {end_str} "
                             f"({trip['duration']} days, {trip['photo_count']} items)")
                logging.info(f"     Peak activity days: {peak_days_str}")

    # Dictionary to track images in each album (for collage creation)
    album_images = defaultdict(list)

    # Second pass: Organize media into albums
    logging.info("Second pass: Organizing media into albums...")
    media_items = list(zip(media_metadata, media_paths))
    with cli_friendly_logging(media_items, desc="Organizing media files", unit="file") as progress:
        for metadata, file_path in progress:
            file_type = "photo" if metadata['type'] == 'image' else "video"

            try:
                # Determine album with trip detection
                album_name = determine_album(metadata, trips)

                # Create album directory if it doesn't exist
                album_dir = output_dir / album_name
                album_dir.mkdir(exist_ok=True)

                # Destination path
                filename = os.path.basename(file_path)
                dest_path = album_dir / filename

                # Copy or move file
                if copy:
                    shutil.copy2(file_path, dest_path)
                    logging.debug(f"Copied to {dest_path}")
                else:
                    shutil.move(file_path, dest_path)
                    logging.debug(f"Moved to {dest_path}")

                # Track image files for collage creation
                if file_type == "photo":
                    album_images[album_name].append(str(dest_path))

                # Update results
                results['processed'] += 1
                if album_name not in results['albums']:
                    results['albums'][album_name] = 0
                results['albums'][album_name] += 1

            except Exception as e:
                logging.error(f"Failed to organize {file_path}: {e}")
                results['failed'] += 1

    # Third pass: Create collages for each album (if enabled or if AI processing is needed)
    if (create_collages or not dont_use_ai) and results['total_photos'] > 0:
        logging.info("Third pass: Creating collages for albums...")
        created_collage_paths = []  # Track collage paths for potential deletion

        with cli_friendly_logging(album_images.items(), desc="Creating album collages", unit="album") as progress:
            for album_name, images in progress:
                if not images:  # Skip albums with no images
                    continue

                # Create collages in batches if there are more than threshold images
                for i in range(0, len(images), collage_threshold):
                    batch = images[i:i + collage_threshold]
                    if not batch:
                        continue

                    # Suffix for multiple collages
                    suffix = f"_part{i // collage_threshold + 1}" if len(images) > collage_threshold else ""
                    collage_path = output_dir / album_name / f"collage{suffix}.jpg"

                    if create_collage(batch, collage_path):
                        results['collages_created'] += 1
                        if not create_collages:  # Track for deletion if only created for AI
                            created_collage_paths.append(collage_path)

    # Fourth pass: Use AI to rename albums (if enabled)
    if not dont_use_ai and results['collages_created'] > 0:
        logging.info("Fourth pass: Using AI to rename trip albums...")
        ai_results = rename_albums_with_ai(output_dir, DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, language)

        # Add AI results to overall results
        results['ai_renamed'] = ai_results.get('renamed', 0)
        results['ai_failed'] = ai_results.get('failed', 0)
        results['ai_skipped'] = ai_results.get('skipped', 0)

    # Delete collages if they were only created for AI processing
    if not create_collages and created_collage_paths:
        logging.info(f"Deleting {len(created_collage_paths)} collages created only for AI processing...")
        deleted_count = 0
        for collage_path in created_collage_paths:
            try:
                os.remove(collage_path)
                deleted_count += 1
            except Exception as e:
                logging.error(f"Failed to delete collage {collage_path}: {e}")
        logging.info(f"Deleted {deleted_count} collages")

    return results


def revert_albums(albums_dir, target_dir=None, copy=False):
    """
    Move all files from album directories back to the root directory.
    
    Args:
        albums_dir (str): Directory containing albums
        target_dir (str, optional): Target directory for files (default: parent of albums_dir)
        copy (bool): Copy files instead of moving them
    
    Returns:
        dict: Summary of results
    """
    albums_dir = Path(albums_dir)

    # If no target directory specified, use parent of albums directory
    if target_dir is None:
        target_dir = albums_dir.parent
    else:
        target_dir = Path(target_dir)

    # Create target directory if it doesn't exist
    target_dir.mkdir(exist_ok=True, parents=True)

    # Results tracking
    results = {
        'total_files': 0,
        'processed': 0,
        'failed': 0,
        'albums_found': 0,
        'collages_removed': 0,
        'skipped_files': 0
    }

    # Get all album directories
    album_dirs = [d for d in albums_dir.iterdir() if d.is_dir()]
    results['albums_found'] = len(album_dirs)

    logging.info(f"Found {len(album_dirs)} albums in {albums_dir}")

    # Process each album
    for album_dir in cli_friendly_logging(album_dirs, desc="Processing albums", unit="album"):
        logging.info(f"Processing album: {album_dir.name}")

        # Get all files in the album
        files = [f for f in album_dir.glob('*') if f.is_file()]
        logging.info(f"  Found {len(files)} files")

        # Process each file
        for file_path in cli_friendly_logging(files, desc=f"  Files in {album_dir.name}", unit="file"):
            results['total_files'] += 1

            # Check if file is a collage
            is_collage = file_path.stem.startswith('collage') and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}

            # Handle collage files differently - skip them and remove
            if is_collage:
                logging.info(f"  Skipping collage file: {file_path.name}")
                results['skipped_files'] += 1

                # Remove collage file if not copying
                if not copy:
                    try:
                        os.remove(file_path)
                        logging.info(f"  Removed collage file: {file_path.name}")
                        results['collages_removed'] += 1
                    except Exception as e:
                        logging.error(f"  Failed to remove collage file {file_path}: {e}")
                        results['failed'] += 1
                continue

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

                # Copy or move file
                if copy:
                    shutil.copy2(file_path, dest_path)
                    logging.info(f"  Copied {file_path.name} to {dest_path}")
                else:
                    shutil.move(file_path, dest_path)
                    logging.info(f"  Moved {file_path.name} to {dest_path}")

                results['processed'] += 1

            except Exception as e:
                logging.error(f"  Failed to process {file_path}: {e}")
                results['failed'] += 1

    # Remove empty album directories if not copying
    if not copy:
        for album_dir in cli_friendly_logging(album_dirs, desc="Cleaning empty directories", unit="dir"):
            try:
                # Check if directory is empty
                if not any(album_dir.iterdir()):
                    album_dir.rmdir()
                    logging.info(f"Removed empty album directory: {album_dir}")
            except Exception as e:
                logging.error(f"Failed to remove directory {album_dir}: {e}")

    return results


def main():
    """Main function for the CLI script."""
    parser = argparse.ArgumentParser(description='Organize photos and videos into albums based on metadata and AI.')
    parser.add_argument('input_dir', help='Directory containing media files')
    parser.add_argument('--output-dir', '-o', help='Output directory for albums (default: INPUT_DIR/Albums)')
    parser.add_argument('--copy', '-c', action='store_true', help='Copy files instead of moving them')
    parser.add_argument('--no-trip-detection', action='store_true',
                        help='Disable trip and event detection, use simple year-month organization')
    parser.add_argument('--max-gap-days', type=int, default=DEFAULT_MAX_GAP_DAYS,
                        help=f'Maximum number of days between trips to merge them (default: {DEFAULT_MAX_GAP_DAYS})')
    parser.add_argument('--no-videos', action='store_true',
                        help='Skip video files and only process photos')
    parser.add_argument('--no-collages', action='store_true',
                        help='Disable automatic collage creation for albums')
    parser.add_argument('--collage-threshold', type=int, default=DEFAULT_COLLAGE_THRESHOLD,
                        help=f'Maximum number of photos per collage (default: {DEFAULT_COLLAGE_THRESHOLD})')
    parser.add_argument('--revert', action='store_true',
                        help='Revert organization by moving files from albums back to root directory')
    parser.add_argument('--no_ai', action='store_true',
                        help='Use AI to rename trip albums based on collage images')
    parser.add_argument('--system-prompt', type=str,
                        help='Custom system prompt for the AI')
    parser.add_argument('--prompt', type=str,
                        help='Custom prompt for the AI')
    parser.add_argument('--language', type=str, default=DEFAULT_LANGUAGE,
                        help=f'Language for album names (default: {DEFAULT_LANGUAGE})')

    args = parser.parse_args()

    # Handle revert mode
    if args.revert:
        albums_dir = args.output_dir if args.output_dir else Path(args.input_dir) / 'Albums'
        print(f"Reverting organization by moving files from {albums_dir} back to {args.input_dir}")

        results = revert_albums(albums_dir, args.input_dir, args.copy)

        # Print summary
        print("\nRevert operation complete!")
        print(f"Albums found: {results['albums_found']}")
        print(f"Total files: {results['total_files']}")
        print(f"Files processed: {results['processed']}")

        # Show skipped collages information
        if 'skipped_files' in results and results['skipped_files'] > 0:
            print(f"Collages skipped: {results['skipped_files']}")

        if not args.copy and 'collages_removed' in results and results['collages_removed'] > 0:
            print(f"Collages removed: {results['collages_removed']}")

        print(f"Files failed: {results['failed']}")

        if args.copy:
            print("Files were copied (originals remain in albums)")
        else:
            print("Files were moved (empty album directories were removed)")

    # Handle organization mode
    else:
        include_videos = not args.no_videos
        create_collages = not args.no_collages
        media_type = "photos and videos" if include_videos else "photos only"
        print(f"Processing {media_type} in {args.input_dir}")

        trip_detection = not args.no_trip_detection

        if trip_detection:
            print(f"Trip and event detection is enabled (max gap between trips: {args.max_gap_days} days)")
            print(f"Trips that are {args.max_gap_days} days or less apart will be merged into single trips")
        else:
            print("Using simple year-month organization (trip detection disabled)")

        if create_collages:
            print(f"Collage creation is enabled (max {args.collage_threshold} photos per collage)")
        else:
            print("Collage creation is disabled")

        results = process_media(args.input_dir, args.output_dir, args.copy,
                                trip_detection, args.max_gap_days, include_videos,
                                create_collages, args.collage_threshold,
                                args.no_ai, args.system_prompt, args.prompt,
                                args.language)

        # Print summary
        print("\nProcessing complete!")
        print(f"Total files found: {results['total_files']}")
        if include_videos:
            print(f"Photos: {results['total_photos']}, Videos: {results['total_videos']}")
        else:
            print(f"Photos: {results['total_photos']}")
        print(f"Files processed: {results['processed']}")
        print(f"Files failed: {results['failed']}")

        if trip_detection and 'trips_detected' in results:
            print(f"Trips/Events detected: {results['trips_detected']}")

        if create_collages and 'collages_created' in results:
            print(f"Collages created: {results['collages_created']}")

        if not args.no_ai and 'ai_renamed' in results:
            print(f"Albums renamed with AI: {results['ai_renamed']}")
            if results.get('ai_failed', 0) > 0:
                print(f"AI renaming failures: {results['ai_failed']}")

        if results['albums']:
            print("\nAlbums created:")
            for album, count in sorted(results['albums'].items()):
                print(f"  - {album}: {count} files")


if __name__ == "__main__":
    main()
