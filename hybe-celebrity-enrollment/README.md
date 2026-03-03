# Celebrity Detection + Dynamic Tags Toolkit

Reusable scripts for enrolling persons into Coactive's Celebrity Detection system and joining face detections with Dynamic Tag (DT) scores.

## Scripts

| Script | Purpose |
|--------|---------|
| `celebrity_enrollment.py` | Enroll persons (faces) into Celebrity Detection |
| `celeb_dt_workflow.py` | Fetch faces + DT scores, join them, generate SQL queries |
| `config.example.json` | Template config file — copy and fill in for your client |

## Quick Start

```bash
# 1. Copy and edit the config file
cp config.example.json config.json
# Edit config.json with your API key, dataset ID, celebrities, and DT tags

# 2. Enroll persons
python3 celebrity_enrollment.py --config config.json

# 3. Wait for backfill (~1-2 hours)

# 4. Check face detections
python3 celebrity_enrollment.py --config config.json --check-faces

# 5. (Optional) Set up Dynamic Tags
python3 celeb_dt_workflow.py --config config.json --dt-setup

# 6. Run the join workflow (after DT scoring completes)
python3 celeb_dt_workflow.py --config config.json
```

## Config File

Create a `config.json` based on `config.example.json`:

```json
{
  "client_name": "Your Client",
  "coactive_api_key": "your-api-key",
  "base_url": "https://api.coactive.ai",
  "dataset_id": "your-dataset-uuid",
  "seed_images_dir": "./seed_images",

  "celebrities": [
    {
      "name": "Person Name",
      "aliases": ["Alt Name", "Nickname"],
      "image_pattern": "PersonName*.png",
      "max_images": 18
    }
  ],

  "dt_groups": [
    {
      "name": "Content Classification",
      "tags": {
        "Tag Name": "Description of what this tag detects in visual content"
      }
    }
  ]
}
```

Alternatively, set environment variables:

```bash
export COACTIVE_API_KEY="your-api-key"
export COACTIVE_DATASET_ID="your-dataset-uuid"
export SEED_IMAGES_DIR="./seed_images"
```

## Celebrity Enrollment

### Enrollment Flow (4-step)

1. **Create draft** — `POST /api/v0/celebrity-detection/enroll`
2. **Upload seed images** — `POST /api/v0/celebrity-detection/upload`
3. **Update draft** — `PATCH /api/v0/celebrity-detection/enroll/{person_id}` (attach upload_ids + aliases)
4. **Finalize** — `POST /api/v0/celebrity-detection/enroll/{person_id}/finalize` (triggers backfill)

### Commands

```bash
# Enroll all persons from config
python3 celebrity_enrollment.py --config config.json

# List enrolled persons
python3 celebrity_enrollment.py --config config.json --list-only

# Check face detection counts after backfill
python3 celebrity_enrollment.py --config config.json --check-faces
```

### Seed Image Requirements

- **Format**: JPEG or PNG
- **Quality**: Clear, well-lit face shots
- **Variety**: Multiple angles, lighting conditions, expressions
- **Count**: 5-18 images per person (more = better accuracy)
- **Naming**: Follow the `image_pattern` glob defined per celebrity in config

## Celebrity + DT Workflow

### Workflow Phases

1. **Fetch faces** — Get face detections for enrolled persons in the dataset
2. **DT setup** — Create Dynamic Tag group with text-prompted tags (if not already created)
3. **Fetch DT scores** — Get scoring-preview scores (0-1 range) per tag per keyframe
4. **Join** — Inner join faces and DT scores on `image_id` = `asset_id`
5. **Output** — SQL queries for Coactive UI + summary statistics

### Commands

```bash
# Full workflow (faces + DT scores + join + SQL queries)
python3 celeb_dt_workflow.py --config config.json

# Fetch faces only (verify detections after backfill)
python3 celeb_dt_workflow.py --config config.json --faces-only

# Create DT group + tags (one-time setup)
python3 celeb_dt_workflow.py --config config.json --dt-setup

# Use a specific DT group ID
python3 celeb_dt_workflow.py --config config.json --dt-group-id abc-123

# Export joined data as JSON
python3 celeb_dt_workflow.py --config config.json --output results.json
```

### Join Logic

The join matches celebrity faces with DT scores on `image_id`:
- Celebrity Detection returns face detections with `image_id` (keyframe UUID)
- Dynamic Tags return scores with `asset_id` (same keyframe UUID)
- Each joined row = one (celebrity, tag) pair for a given keyframe

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v0/login` | POST | Exchange API key for JWT |
| `/api/v0/celebrity-detection/enroll` | POST | Create draft person |
| `/api/v0/celebrity-detection/upload` | POST | Upload seed images |
| `/api/v0/celebrity-detection/enroll/{id}` | PATCH | Update draft with uploads |
| `/api/v0/celebrity-detection/enroll/{id}/finalize` | POST | Finalize enrollment |
| `/api/v0/celebrity-detection/persons` | GET | List enrolled persons |
| `/api/v0/celebrity-detection/faces-with-person` | GET | Get face detections |
| `/api/v0/dynamic-tags/groups` | GET/POST | List/create DT groups |
| `/api/v0/dynamic-tags/groups/{id}/versions/latest` | GET | Get group version |
| `/api/v3/dynamic-tags/groups/{id}/versions/{vid}/publish` | POST | Publish group |

## Requirements

```
requests>=2.28.0
```

## Author

Madan - Coactive AI
