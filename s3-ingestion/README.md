# S3 Ingestion Tools

Utilities for copying assets from Google Drive to S3 and ingesting them into Coactive datasets.

## Tools

| Script | Purpose |
|--------|---------|
| `gdrive-to-s3.sh` | Copy/sync files from Google Drive to S3 via rclone |
| `coactive_ingest.py` | Ingest S3 assets into a Coactive dataset |

## Prerequisites

- **rclone** — `brew install rclone`
- **Python 3** with `requests` — `pip3 install requests`
- Configured rclone remotes for Google Drive (`gdrive`) and S3 (`s3`) — run `rclone config`

## Google Drive → S3 Copy

### Quick Start

```bash
# Dry run (preview, no changes)
./gdrive-to-s3.sh -s "My Folder" -b my-bucket -d dest/path --dry-run

# Copy
./gdrive-to-s3.sh -s "My Folder" -b my-bucket -d dest/path
```

### Using a Shared Folder by ID

For shared Google Drive folders, use rclone directly with `--drive-root-folder-id`:

```bash
rclone copy \
  "gdrive:" \
  "s3:my-bucket/dest/path/" \
  --drive-root-folder-id "FOLDER_ID_FROM_URL" \
  --transfers 16 \
  --checksum \
  --retries 3 \
  --progress
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-s, --source` | Google Drive source path | required |
| `-b, --bucket` | S3 bucket name | required |
| `-d, --dest` | S3 destination prefix | `""` |
| `-t, --transfers` | Parallel transfers | 16 |
| `-n, --dry-run` | Preview only | false |
| `--sync` | Mirror mode (deletes extras on S3) | false |

### Low Memory Mode

For resource-constrained environments:

```bash
rclone copy "gdrive:" "s3:bucket/path/" \
  --drive-root-folder-id "FOLDER_ID" \
  --transfers 4 \
  --checkers 2 \
  --buffer-size 0 \
  --s3-upload-concurrency 2
```

## Coactive Dataset Ingestion

### Quick Start

```bash
# With token as argument
python3 coactive_ingest.py \
  --dataset-id "YOUR_DATASET_ID" \
  --s3-path "s3://bucket/prefix/" \
  --max-depth 2 \
  --token "YOUR_COACTIVE_API_TOKEN"

# With environment variable
export COACTIVE_API_TOKEN="your-token"
python3 coactive_ingest.py \
  --dataset-id "YOUR_DATASET_ID" \
  --s3-path "s3://bucket/prefix/"
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset-id` | Coactive dataset UUID | required |
| `--s3-path` | S3 path to ingest | required |
| `--max-depth` | Max directory depth to crawl | 10 |
| `--token` | Coactive API token (or use `COACTIVE_API_TOKEN` env var) | required |

### Choosing `max-depth`

| Folder Structure | Recommended |
|-----------------|-------------|
| `s3://bucket/videos/*.mp4` | `--max-depth 1` |
| `s3://bucket/campaign/videos/*.mp4` | `--max-depth 2` |
| Deep nested folders | `--max-depth 10` (default) |

## End-to-End Example

```bash
# 1. Copy from Google Drive shared folder to S3
rclone copy "gdrive:" "s3:coactive-datasets-production/poc/MyProject/" \
  --drive-root-folder-id "1MzuoeufOX3s_CNzZHzp16iq92OpKQdNd" \
  --transfers 16 --checksum --retries 3 --progress

# 2. Ingest into Coactive
python3 coactive_ingest.py \
  --dataset-id "7114766b-7828-4dce-87ce-34e352ef3c06" \
  --s3-path "s3://coactive-datasets-production/poc/MyProject/" \
  --max-depth 2 \
  --token "$COACTIVE_API_TOKEN"
```

## Author

Madan - Coactive AI
