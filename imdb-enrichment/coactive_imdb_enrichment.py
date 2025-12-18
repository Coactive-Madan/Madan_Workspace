#!/usr/bin/env python3
"""
Coactive IMDB Metadata Updater
This script fetches video assets from Coactive, retrieves IMDB metadata, and updates the assets.
"""

import os
import re
import requests
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse


class CoactiveIMDBUpdater:
    """Handles fetching videos from Coactive and updating with IMDB metadata"""
    
    def __init__(self, api_key: str, dataset_id: str):
        """
        Initialize the updater
        
        Args:
            api_key: Coactive API key (will be exchanged for session token)
            dataset_id: Coactive dataset ID
        """
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.base_url = "https://app.coactive.ai/api/v1"
        self.session_token = None
        self.headers = {}
        
        # Get session token on initialization
        self._authenticate()
    
    def _authenticate(self):
        """
        Exchange API key for session token using curl (Python requests has SSL issues on this system)
        """
        print("Authenticating with Coactive API...")
        
        import subprocess
        import json
        
        # Use curl to get the token (curl works, python requests doesn't due to LibreSSL/TLS issues)
        cmd = [
            'curl', '-s', '-X', 'POST',
            'https://api.coactive.ai/api/v0/login',
            '-H', f'Authorization: Bearer {self.api_key}',
            '-H', 'Content-Type: application/json',
            '-d', '{"grant_type": "refresh_token"}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                self.session_token = data.get("access_token") or data.get("token")
                self.headers = {"Authorization": f"Bearer {self.session_token}"}
                print("✓ Authentication successful\n")
            except json.JSONDecodeError:
                print(f"✗ Authentication failed: Invalid JSON response")
                print(f"  Response: {result.stdout}")
                raise Exception(f"Failed to authenticate with Coactive API")
        else:
            print(f"✗ Authentication failed: curl error")
            print(f"  Error: {result.stderr}")
            raise Exception(f"Failed to authenticate with Coactive API")
        
    def get_videos(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Fetch videos from Coactive dataset
        
        Args:
            limit: Maximum number of videos to fetch per request
            offset: Starting offset for pagination
            
        Returns:
            List of video objects
        """
        url = f"{self.base_url}/datasets/{self.dataset_id}/videos"
        params = {"limit": limit, "offset": offset}
        
        print(f"Fetching videos from Coactive (offset: {offset}, limit: {limit})...")
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            videos = data.get("data", [])
            print(f"✓ Retrieved {len(videos)} videos")
            return videos
        else:
            print(f"✗ Error fetching videos: {response.status_code}")
            print(response.text)
            return []
    
    def get_all_videos(self) -> List[Dict]:
        """
        Fetch all videos from the dataset using pagination
        
        Returns:
            List of all video objects
        """
        all_videos = []
        offset = 0
        limit = 100
        
        while True:
            videos = self.get_videos(limit=limit, offset=offset)
            if not videos:
                break
            all_videos.extend(videos)
            offset += limit
            
            # Break if we got fewer videos than the limit (last page)
            if len(videos) < limit:
                break
        
        print(f"\n✓ Total videos fetched: {len(all_videos)}\n")
        return all_videos
    
    def extract_title_from_path(self, path: str) -> Optional[str]:
        """
        Extract movie/show title from S3 path
        
        Args:
            path: S3 path to video file
            
        Returns:
            Extracted title or None
        """
        # Extract filename from path
        filename = os.path.basename(path)
        
        # Remove extension
        title = os.path.splitext(filename)[0]
        
        # Clean up common patterns (underscores, dashes, etc.)
        title = title.replace("_", " ").replace("-", " ")
        
        # Remove year patterns like (2023) or [2023]
        title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?\s*', ' ', title)
        
        # Remove common suffixes like "trailer", "clip", "Official", etc.
        # Match the pattern like "MovieTitle - Official Clip - Scene Name"
        # and extract just "MovieTitle"
        title = re.sub(r'\s*(official\s+)?(clip|trailer|promo|teaser|HD|1080p|720p|4K).*', '', title, flags=re.IGNORECASE)
        
        # If there's a dash, try to get the first part (usually the movie title)
        if ' - ' in title:
            parts = title.split(' - ')
            # Filter out parts that are too short or look like technical specs
            valid_parts = [p for p in parts if len(p) > 3 and not re.match(r'^[\dXx]+$', p)]
            if valid_parts:
                title = valid_parts[0]
        
        # Clean up extra spaces
        title = ' '.join(title.split())
        
        return title if title else None
    
    def search_imdb(self, title: str) -> Optional[Dict]:
        """
        Search for a title on IMDB and return metadata
        
        Args:
            title: Movie or show title to search for
            
        Returns:
            Dictionary containing IMDB metadata
        """
        try:
            from imdbinfo import search_title, get_movie
            
            print(f"  Searching IMDB for: '{title}'")
            results = search_title(title)
            
            if results and results.titles:
                # Get the first result
                first_result = results.titles[0]
                movie = get_movie(first_result.imdb_id)
                
                # Extract metadata - handle missing attributes gracefully
                metadata = {
                    "imdb_id": getattr(movie, 'imdb_id', None),
                    "imdb_title": getattr(movie, 'title', None),
                    "imdb_year": getattr(movie, 'year', None),
                    "imdb_rating": getattr(movie, 'rating', None),
                    "imdb_genres": ", ".join(movie.genres) if hasattr(movie, 'genres') and movie.genres else None,
                    "imdb_plot": getattr(movie, 'plot', None),
                    "imdb_directors": ", ".join([d.name for d in movie.directors]) if hasattr(movie, 'directors') and movie.directors else None,
                    "imdb_cast": ", ".join([a.name for a in movie.cast[:5]]) if hasattr(movie, 'cast') and movie.cast else None,  # Top 5 actors
                    "imdb_url": f"https://www.imdb.com/title/{movie.imdb_id}/" if hasattr(movie, 'imdb_id') else None
                }
                
                print(f"  ✓ Found: {movie.title} ({movie.year if hasattr(movie, 'year') else 'N/A'}) - Rating: {movie.rating if hasattr(movie, 'rating') else 'N/A'}")
                return metadata
            else:
                print(f"  ✗ No IMDB results found for '{title}'")
                return None
                
        except ImportError:
            print("  ✗ Error: 'imdbinfo' library not installed. Run: pip install imdbinfo")
            return None
        except Exception as e:
            print(f"  ✗ Error searching IMDB: {str(e)}")
            return None
    
    def update_video_metadata(self, video_id: str, metadata: Dict) -> bool:
        """
        Update metadata for a video asset in Coactive
        
        Args:
            video_id: Coactive video ID
            metadata: Dictionary of metadata to add/update
            
        Returns:
            True if successful, False otherwise
        """
        import subprocess
        import json
        
        # Use the ingestion metadata endpoint
        url = "https://api.coactive.ai/api/v1/ingestion/metadata"
        
        # Remove None values from metadata
        clean_metadata = {k: v for k, v in metadata.items() if v is not None}
        
        payload = {
            "dataset_id": self.dataset_id,
            "update_assets": [
                {
                    "asset_type": "video",
                    "asset_id": video_id,
                    "metadata": clean_metadata
                }
            ],
            "update_type": "upsert"
        }
        
        # Use curl to make the request (to avoid SSL issues)
        cmd = [
            'curl', '-s', '-X', 'POST',
            url,
            '-H', f'Authorization: {self.headers["Authorization"]}',
            '-H', 'Content-Type: application/json',
            '-d', json.dumps(payload),
            '-w', '\n%{http_code}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract status code from end of output
            output_lines = result.stdout.strip().split('\n')
            status_code = output_lines[-1] if output_lines else ''
            response_body = '\n'.join(output_lines[:-1]) if len(output_lines) > 1 else ''
            
            if status_code in ['200', '201', '202']:
                print(f"  ✓ Metadata updated successfully")
                return True
            else:
                print(f"  ✗ Error updating metadata: {status_code}")
                print(f"     {response_body}")
                return False
        else:
            print(f"  ✗ Error updating metadata: curl failed")
            print(f"     {result.stderr}")
            return False
    
    def process_videos(self, dry_run: bool = False):
        """
        Main processing function: fetch videos, get IMDB data, update metadata
        
        Args:
            dry_run: If True, don't actually update metadata (just simulate)
        """
        print("=" * 70)
        print("COACTIVE IMDB METADATA UPDATER")
        print("=" * 70)
        print(f"Dataset ID: {self.dataset_id}")
        print(f"Dry Run: {dry_run}")
        print("=" * 70 + "\n")
        
        # Fetch all videos
        videos = self.get_all_videos()
        
        if not videos:
            print("No videos found. Exiting.")
            return
        
        # Process each video
        success_count = 0
        failed_count = 0
        
        for idx, video in enumerate(videos, 1):
            print(f"\n[{idx}/{len(videos)}] Processing video...")
            
            # Extract video info
            video_id = video.get("coactiveVideoId") or video.get("coactive_video_id") or video.get("id")
            path = video.get("path", "")
            
            print(f"  Video ID: {video_id}")
            print(f"  Path: {path}")
            
            # Extract title from path
            title = self.extract_title_from_path(path)
            
            if not title:
                print(f"  ✗ Could not extract title from path")
                failed_count += 1
                continue
            
            print(f"  Extracted Title: '{title}'")
            
            # Search IMDB
            imdb_metadata = self.search_imdb(title)
            
            if not imdb_metadata:
                failed_count += 1
                continue
            
            # Update metadata in Coactive
            if dry_run:
                print(f"  [DRY RUN] Would update metadata:")
                for key, value in imdb_metadata.items():
                    print(f"    {key}: {value}")
                success_count += 1
            else:
                if self.update_video_metadata(video_id, imdb_metadata):
                    success_count += 1
                else:
                    failed_count += 1
            
            # Rate limiting - be nice to IMDB
            time.sleep(1)
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total videos processed: {len(videos)}")
        print(f"Successfully updated: {success_count}")
        print(f"Failed: {failed_count}")
        print("=" * 70)


def main():
    """Main entry point"""
    # Configuration - set these values
    API_KEY = os.environ.get("COACTIVE_API_KEY", "")
    DATASET_ID = os.environ.get("COACTIVE_DATASET_ID", "")
    
    # Check for required configuration
    if not API_KEY:
        print("Error: COACTIVE_API_KEY environment variable not set")
        print("Usage: export COACTIVE_API_KEY='your_api_key_here'")
        return
    
    if not DATASET_ID:
        print("Error: COACTIVE_DATASET_ID environment variable not set")
        print("Usage: export COACTIVE_DATASET_ID='your_dataset_id_here'")
        return
    
    # Create updater and process videos
    updater = CoactiveIMDBUpdater(API_KEY, DATASET_ID)
    
    # Set dry_run=True to test without updating
    updater.process_videos(dry_run=False)


if __name__ == "__main__":
    main()

