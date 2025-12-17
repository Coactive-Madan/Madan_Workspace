# Coactive Video Narrative Metadata Enrichment

A Python script to automatically enrich video assets in Coactive with AI-generated narrative metadata using the Coactive Video Narrative APIs.

## Features

- **Video Summary** - Comprehensive plot/content summary with customizable intent
- **Video Description** - Marketing-style description for catalogs
- **Genre Classification** - Automatic genre detection (Drama, Comedy, Action, etc.)
- **Mood Analysis** - Emotional tone (Intense, Heartwarming, Humorous, etc.)
- **Subject Extraction** - Main themes and topics
- **Format Detection** - Presentation style (Feature Film, Documentary, Stand-up Special, etc.)
- **Keyframe Captioning** - AI captions for video keyframes (async)

## Requirements

- Python 3.7+
- `curl` command-line tool
- Coactive API access (Personal Token)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/coactive-narrative-enrichment.git
cd coactive-narrative-enrichment
```

No additional dependencies required - uses only Python standard library and curl.

## Usage

### Basic Usage

```bash
python3 coactive_narrative_enrichment.py -d DATASET_ID -t YOUR_TOKEN
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--dataset-id` | `-d` | Coactive Dataset ID (required) |
| `--token` | `-t` | Coactive Personal/Refresh Token (required) |
| `--video-id` | `-v` | Process only a specific video |
| `--intent` | `-i` | Custom summary intent/prompt |
| `--limit` | `-l` | Max videos to process (default: 100) |
| `--delay` | | Delay between videos in seconds (default: 1.0) |

### Examples

```bash
# Enrich all videos in a dataset
python3 coactive_narrative_enrichment.py \
  -d d2fae475-4ebd-46ac-8bad-af2c5a784b43 \
  -t YOUR_COACTIVE_TOKEN

# Process a single video
python3 coactive_narrative_enrichment.py \
  -d DATASET_ID -t TOKEN \
  -v a0c6b150-3606-4e08-ae37-3efe37615868

# Custom summary focus
python3 coactive_narrative_enrichment.py \
  -d DATASET_ID -t TOKEN \
  --intent "Focus on action scenes and character development"

# Limit to first 10 videos
python3 coactive_narrative_enrichment.py \
  -d DATASET_ID -t TOKEN \
  --limit 10
```

## Output

The script adds the following metadata fields to each video:

| Field | Description |
|-------|-------------|
| `video_narrative_summary` | Comprehensive content summary |
| `video_narrative_description` | Marketing-style description |
| `video_narrative_genre` | Genre classification |
| `video_narrative_mood` | Emotional tone |
| `video_narrative_subject` | Main themes/topics |
| `video_narrative_format` | Presentation style |
| `video_narrative_keyframes_requested` | Keyframe captioning status |
| `video_narrative_metadata_source` | API source identifier |
| `video_narrative_metadata_generated_at` | ISO timestamp |

## API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v0/login` | Authentication |
| `GET /api/v1/datasets/{id}` | Dataset info |
| `GET /api/v1/datasets/{id}/videos` | List videos |
| `POST /api/v0/video-summarization/.../summarize` | Generate summary |
| `POST /api/v0/video-narrative-metadata/.../description` | Generate description |
| `POST /api/v0/video-narrative-metadata/.../genre` | Detect genre |
| `POST /api/v0/video-narrative-metadata/.../mood` | Analyze mood |
| `POST /api/v0/video-narrative-metadata/.../subject` | Extract subjects |
| `POST /api/v0/video-narrative-metadata/.../format` | Detect format |
| `POST /api/v0/video-summarization/.../caption-keyframes` | Caption keyframes |
| `POST /api/v1/ingestion/metadata` | Update metadata |

## Sample Output

```
================================================================================
🎬 COACTIVE NARRATIVE METADATA ENRICHMENT
================================================================================
Dataset: d2fae475-4ebd-46ac-8bad-af2c5a784b43
Time: 2025-12-15 14:28:55
================================================================================
✅ Authenticated
📹 Found 5 videos

[1/5] 🎬 3:10 to Yuma
    ID: a0c6b150-3606-4e08-ae37-3efe37615868
    📝 Summary...
       ✅ Got summary (1305 chars)
    📄 Description...
       ✅ Got description
    🎬 Genre...
       ✅ Drama, Western
    🎭 Mood...
       ✅ Intense, Dramatic, Tense
    📚 Subject...
       ✅ Redemption, Survival, Justice, Family
    🎞️ Format...
       ✅ Feature Film
    🖼️ Keyframes...
       ✅ Captions already exist (1825 keyframes)
    💾 Updating (9 fields)...
       ✅ Metadata updated!

================================================================================
📊 ENRICHMENT SUMMARY
================================================================================
✅ Successfully enriched: 5/5
```

## Getting Your Coactive Token

1. Log in to [Coactive](https://app.coactive.ai)
2. Go to **Settings** → **Credentials**
3. Copy your **Personal Token**

## License

MIT License

## Author

Created for Lionsgate POC - December 2025

