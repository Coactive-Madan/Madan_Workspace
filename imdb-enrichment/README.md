# Coactive IMDB Metadata Enrichment

Automatically enrich Coactive video assets with IMDB metadata by extracting titles from file paths and fetching movie/show information.

## Overview

This script:
1. Fetches all videos from a Coactive dataset
2. Extracts movie/show titles from file paths
3. Searches IMDB for matching content
4. Updates Coactive assets with rich metadata

## Metadata Fields Added

| Field | Description |
|-------|-------------|
| `imdb_id` | IMDB identifier (e.g., tt1234567) |
| `imdb_title` | Official title from IMDB |
| `imdb_year` | Release year |
| `imdb_rating` | IMDB rating (1-10) |
| `imdb_genres` | Comma-separated genres |
| `imdb_plot` | Plot summary |
| `imdb_directors` | Director names |
| `imdb_cast` | Top 5 cast members |
| `imdb_url` | Link to IMDB page |

## Usage

### Environment Variables

```bash
export COACTIVE_API_KEY="your_refresh_token_here"
export COACTIVE_DATASET_ID="your_dataset_uuid_here"
```

### Run

```bash
python3 coactive_imdb_enrichment.py
```

### Dry Run (Test without updating)

Edit the script and set `dry_run=True`:

```python
updater.process_videos(dry_run=True)
```

## Title Extraction

The script intelligently extracts titles from file paths:

| Input Path | Extracted Title |
|------------|-----------------|
| `s3://bucket/The Matrix (1999).mp4` | `The Matrix` |
| `s3://bucket/Inception_Official_Trailer_HD.mp4` | `Inception` |
| `s3://bucket/Pulp-Fiction-1994-clip.mp4` | `Pulp Fiction` |

**Patterns handled:**
- Year patterns: `(2023)`, `[2023]`, `2023`
- Suffixes: `trailer`, `clip`, `promo`, `HD`, `1080p`, `4K`
- Separators: underscores, dashes, spaces

## Requirements

```bash
pip install imdbinfo requests
```

## API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v0/login` | Exchange token for JWT |
| `GET /api/v1/datasets/{id}/videos` | Fetch video list |
| `POST /api/v1/ingestion/metadata` | Update asset metadata |

## Class API

```python
from coactive_imdb_enrichment import CoactiveIMDBUpdater

# Initialize
updater = CoactiveIMDBUpdater(
    api_key="your_token",
    dataset_id="your_dataset_id"
)

# Get all videos
videos = updater.get_all_videos()

# Search IMDB for a title
metadata = updater.search_imdb("The Matrix")

# Update video metadata
updater.update_video_metadata(video_id, metadata)

# Process all videos
updater.process_videos(dry_run=False)
```

## Output Example

```
======================================================================
COACTIVE IMDB METADATA UPDATER
======================================================================
Dataset ID: abc123-def456-...
Dry Run: False
======================================================================

Fetching videos from Coactive (offset: 0, limit: 100)...
✓ Retrieved 25 videos

[1/25] Processing video...
  Video ID: vid-123
  Path: s3://bucket/The_Matrix_1999_trailer.mp4
  Extracted Title: 'The Matrix'
  Searching IMDB for: 'The Matrix'
  ✓ Found: The Matrix (1999) - Rating: 8.7
  ✓ Metadata updated successfully
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `imdbinfo not installed` | `pip install imdbinfo` |
| No IMDB results | Title extraction may need adjustment |
| SSL errors | Script uses curl to bypass SSL issues |
| Rate limiting | 1 second delay between requests |

## Author

Madan - Coactive AI  
December 2025

