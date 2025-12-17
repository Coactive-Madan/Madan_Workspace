# HYBE Celebrity Enrollment

Enroll K-pop artists into Coactive's Celebrity Detection system for automatic face recognition in video content.

## Overview

This script automates the enrollment of HYBE artists (e.g., ENHYPEN members) into the Coactive Celebrity Detection system. Once enrolled, the system can automatically detect and identify these artists in video content.

## Features

- **JWT Authentication**: Secure API key exchange for Coactive API access
- **Base64 Image Encoding**: Automatically encodes seed face images
- **Multiple Artist Support**: Configure multiple artists for batch enrollment
- **Status Monitoring**: Check enrollment and backfill job progress
- **Detection Queries**: Search for detected faces in processed content

## Usage

```bash
# Configure the script with your credentials
export COACTIVE_API_KEY="your-api-key"

# Run the enrollment
python hybe_celebrity_enrollment.py
```

## Configuration

Edit the script to configure:

| Variable | Description |
|----------|-------------|
| `COACTIVE_API_KEY` | Your Coactive API key |
| `DATASET_ID` | Target dataset for celebrity detection |
| `SEED_IMAGES_DIR` | Directory containing face seed images |
| `ARTISTS` | List of artists to enroll |

### Artist Configuration

```python
ARTISTS = [
    {
        "name": "Park Sunghoon",
        "group": "ENHYPEN",
        "image_pattern": "Sunghoon_*.jpg",
        "max_images": 5
    },
    # Add more artists...
]
```

## Seed Image Requirements

- **Format**: JPEG or PNG
- **Quality**: Clear, well-lit face shots
- **Variety**: Multiple angles recommended
- **Naming**: Follow pattern defined in artist config

## Workflow

1. **Enrollment**: Upload seed face images to Coactive
2. **Backfill**: System scans existing content (1-2 hours)
3. **Detection**: Query `/images_with_person` for matches
4. **Verification**: Review detections in Coactive UI

## API Endpoints Used

- `POST /v1/auth/token` - Get JWT token
- `POST /v1/datasets/{id}/celebrities` - Enroll celebrity
- `GET /v1/datasets/{id}/celebrities` - List enrolled celebrities
- `GET /v1/datasets/{id}/images_with_person` - Query detections

## Post-Enrollment

After backfill completes:
- Verify detections in Coactive UI Roster Tab
- Use dynamic tags for behavior detection
- Build search queries for artist appearances

## Requirements

```
requests>=2.28.0
```

## Author

Madan - Coactive AI

