# Video Scraper Pipeline — Coactive-Compliant

Reusable pipeline to **search, download, validate, and prepare** video datasets from YouTube for ingestion into [Coactive](https://coactive.ai). All outputs comply with Coactive's accepted media formats out of the box.

## What It Does

```
targets.json  -->  [Search]  -->  [Download]  -->  [Validate]  -->  [Extract Frames]  -->  [Manifest CSV]
  (config)        (yt-dlp)     (H.264/AAC)     (ffprobe)        (ffmpeg jpg)          (Coactive ingest)
```

| Phase | Description | Output |
|-------|-------------|--------|
| 1. Search | YouTube search via yt-dlp | Video URLs |
| 2. Download | Coactive-compliant format | `.mp4` (H.264, AAC, 720p, 30fps) |
| 3. Validate | ffprobe checks codec, resolution, fps, duration | Pass/fail report |
| 3b. Re-encode | Fix non-compliant files | Converted `.mp4` |
| 4. Frames | Extract jpg frames at interval | `.jpg` (>=360px) |
| 5. Manifest | Generate CSV for Coactive batch ingestion | `video_manifest.csv` |

## Quick Start

```bash
# Install dependencies
pip install yt-dlp
brew install ffmpeg  # or: apt-get install ffmpeg

# Clone and setup
git clone git@github.com:Coactive-Madan/video-scraper-pipeline.git
cd video-scraper-pipeline

# Create example targets.json
python3 pipeline.py --init

# Edit targets.json with your search queries
# Then preview what will be found
python3 pipeline.py --dry-run

# Run the full pipeline
python3 pipeline.py
```

## Usage

```bash
# Full pipeline (all tiers)
python3 pipeline.py

# Single tier
python3 pipeline.py --tier 1

# Search only (no downloads)
python3 pipeline.py --search-only

# Preview without downloading
python3 pipeline.py --dry-run

# Validate existing files
python3 pipeline.py --validate-only

# Fix non-compliant files
python3 pipeline.py --validate-only --reencode

# Extract jpg frames (1 per 5 seconds)
python3 pipeline.py --extract-frames

# Custom frame interval
python3 pipeline.py --extract-frames --frame-interval 10

# Generate manifest with your S3 prefix
python3 pipeline.py --manifest-only --s3-prefix s3://my-bucket/my-prefix

# Download at 360p instead of 720p (faster ingestion)
python3 pipeline.py --max-height 360
```

## Configuration

All video targets are defined in `targets.json`. Run `--init` to create a starter template, then edit it:

```json
{
  "targets": [
    {
      "campaign": "Product_Launch",
      "search_query": "\"my brand\" product launch ad 2025",
      "product_line": "Product_A",
      "tier": 1,
      "category": "brand_ads",
      "max_results": 3,
      "url": null
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `campaign` | Name for this group of videos |
| `search_query` | YouTube search string (use quotes for exact match) |
| `product_line` | Metadata tag for manifest |
| `tier` | Priority level (filter with `--tier N`) |
| `category` | Subfolder name under `raw_videos/` and `frames/` |
| `max_results` | How many search results to download |
| `url` | Direct YouTube URL — skips search if provided |

## Output Structure

```
project/
├── pipeline.py              # Main script
├── targets.json             # Video targets config (you edit this)
├── downloaded.txt           # yt-dlp archive (prevents re-downloads)
├── raw_videos/
│   ├── brand_ads/           # Videos organized by category
│   └── competitor_ads/
├── converted/               # Re-encoded non-compliant files
├── frames/
│   ├── brand_ads/           # Extracted .jpg frames
│   └── competitor_ads/
├── metadata/
│   └── video_manifest.csv   # Coactive CSV ingest format
└── logs/                    # Timestamped pipeline run logs
```

## Coactive Compliance

Downloads are automatically formatted to meet Coactive's requirements:

| Spec | Accepted | This Pipeline |
|------|----------|---------------|
| Container | .mp4, .mkv, .webm, .flv, .avi | .mp4 |
| Video Codec | H.264, H.265 | H.264 (libx264) |
| Audio Codec | AAC, MP3 | AAC |
| Resolution | 240p - 1080p | 720p (configurable) |
| Frame Rate | Up to 60fps | 30fps |
| Duration | 5s - 5 hours | Validated |
| Frame Format | .jpg, .png, .webp (>=350px) | .jpg (>=360px) |

## Manifest CSV

The generated `video_manifest.csv` follows Coactive's CSV ingest format:

```csv
source_path,asset_type,campaign,product_line,tier,category,date_published,duration_sec,width,height,video_codec,fps,compliant
s3://my-bucket/brand_ads/video.mp4,video,Product_Launch,Product_A,1,brand_ads,2025-01-15,35,1280,720,h264,30.0,True
```

- `source_path` is the required column (set via `--s3-prefix`)
- All other columns are treated as string metadata by Coactive
- Headers use snake_case (no spaces, no special characters)

## Re-downloading & Incremental Runs

The `downloaded.txt` file tracks what's already been downloaded (yt-dlp archive). Running the pipeline again will skip previously downloaded videos. Delete this file to force re-download.

## Prerequisites

| Tool | Install | Purpose |
|------|---------|---------|
| Python 3.9+ | -- | Script runtime |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | `pip install yt-dlp` | YouTube search & download |
| [ffmpeg](https://ffmpeg.org/) | `brew install ffmpeg` | Video processing, frame extraction |

## License

Internal use only.
