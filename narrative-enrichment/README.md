# Coactive Narrative Metadata Enrichment

Generate AI-powered narrative metadata for videos in a Coactive dataset using the Video Narrative Metadata API.

## Overview

This script automatically enriches videos with:
- **Summary** - AI-generated comprehensive summary
- **Description** - Detailed video description
- **Genre** - Classification (Awards Show, Documentary, Talk Show, etc.)
- **Mood** - Emotional tone (Celebratory, Emotional, Exciting, etc.)
- **Subject** - Topic classification (Music, Celebrity, Awards, etc.)
- **Format** - Content format (Speech, Performance, Interview, etc.)
- **Keyframe Captions** - AI-generated captions for key frames (async)

## Important: Metadata Setup

**For genre, mood, subject, and format to work, you must first create the classification values.**

The API needs to know what categories are available before it can classify videos. Run with `--setup-metadata` on first use:

```bash
# First time setup - creates classification categories
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --setup-metadata
```

This creates default values like:
- **Genre**: Awards Show, Documentary, Reality TV, Talk Show, Music Video
- **Mood**: Celebratory, Emotional, Exciting, Inspiring, Nostalgic
- **Subject**: Music, Celebrity, Awards, Fashion, Entertainment Industry
- **Format**: Speech, Performance, Interview, Highlight Reel, Behind The Scenes

## Usage

```bash
# Setup metadata values and run enrichment
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --setup-metadata

# Run enrichment only (after metadata is already configured)
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN

# Setup metadata values only (no enrichment)
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --setup-only

# Process a single video
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --video-id VIDEO_UUID

# Custom summary intent
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --intent "Focus on action scenes"

# Limit number of videos
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --limit 10
```

## Options

| Option | Description |
|--------|-------------|
| `-d, --dataset-id` | Coactive Dataset ID (required) |
| `-t, --token` | Coactive Refresh Token (required) |
| `--setup-metadata` | Create classification values before enrichment |
| `--setup-only` | Only create metadata values, don't enrich |
| `-v, --video-id` | Process only this specific video |
| `-i, --intent` | Custom summary intent |
| `-l, --limit` | Max videos to process (default: 100) |
| `--delay` | Delay between videos in seconds (default: 0.5) |

## API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v0/login` | Exchange refresh token for JWT |
| `POST /api/v0/video-narrative-metadata/metadata` | Create classification values |
| `GET /api/v0/video-narrative-metadata/metadata` | Get existing classification values |
| `POST /api/v0/video-summarization/.../summarize` | Generate video summary |
| `POST /api/v0/video-narrative-metadata/.../description` | Get description |
| `POST /api/v0/video-narrative-metadata/.../genre` | Classify genre |
| `POST /api/v0/video-narrative-metadata/.../mood` | Classify mood |
| `POST /api/v0/video-narrative-metadata/.../subject` | Classify subject |
| `POST /api/v0/video-narrative-metadata/.../format` | Classify format |
| `POST /api/v0/video-summarization/.../caption-keyframes` | Trigger keyframe captioning |
| `POST /api/v1/ingestion/metadata` | Update video metadata |

## Custom Metadata Values

You can customize the classification values by modifying `DEFAULT_METADATA_VALUES` in the script:

```python
DEFAULT_METADATA_VALUES = {
    "genre": [
        {
            "name": "Your Genre",
            "description": "Description of this genre",
            "examples": ["Example 1", "Example 2"]
        },
        # ... more genres
    ],
    "mood": [...],
    "subject": [...],
    "format": [...],
}
```

## Output

Each video will have metadata fields added:
- `video_narrative_summary`
- `video_narrative_description`
- `video_narrative_genre`
- `video_narrative_mood`
- `video_narrative_subject`
- `video_narrative_format`
- `video_narrative_keyframes_requested`
- `video_narrative_metadata_source`
- `video_narrative_metadata_generated_at`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Genre/Mood/Subject/Format return empty | Run with `--setup-metadata` first |
| "No metadata values found" | Classification values not created yet |
| "Internal Server Error" on summary | Dataset may need encoder configured |
| Auth failed | Token expired, get a new one |

## Requirements

- Python 3.7+
- `curl` (used for API calls)
- Coactive account with API access

## Author

Madan - Coactive AI  
December 2025
