# Coactive Video Narrative Metadata API - Classification Guide

## What is the Narrative Metadata API?

The Coactive Video Narrative Metadata API analyzes video content using AI to automatically generate metadata. It includes several capabilities:

| Capability | What it does | Output |
|------------|--------------|--------|
| **Summarization** | Generates a text summary of the video | Free text |
| **Description** | Generates a detailed description | Free text |
| **Classification** | Categorizes video into YOUR defined taxonomy | Category matches |
| **Entity Extraction** | Identifies people, celebrities, brands | List of entities |
| **Keyframe Captioning** | Extracts key frames with descriptions | Images + captions |
| **Segment Detection** | Identifies scenes/chapters with timestamps | Timestamped segments |

---

## Classification: How It Works

Classification lets you define **your own taxonomy** and the API will automatically categorize videos into your categories.

**Supported classification types:**

| Type | Purpose | Example Categories |
|------|---------|-------------------|
| `mood` | Emotional tone of the content | Dark & Enigmatic, Uplifting & Romantic |
| `subject` | Theme or topic | Romance, Heroic Conflicts, Societal Issues |
| `genre` | Content type/category | Drama, Comedy, Action, Documentary |
| `format` | Production style | Scripted Series, Reality TV, Film |

---

## Step-by-Step Guide

### Step 1: Define Your Taxonomy (One-Time Setup)

Before classifying videos, you must tell Coactive what categories exist. For each category, provide:

- **Name** - The category label
- **Description** - What this category means
- **Examples** - Sample content that fits this category

**API Request:**
```bash
curl -X POST "https://api.coactive.ai/api/v0/video-narrative-metadata/metadata" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "your-dataset-id",
    "metadata_type": "mood",
    "name": "Dark & Enigmatic",
    "description": "Mysterious, shadowy, and intense tones blending noir, supernatural, and gothic elements",
    "examples": ["noir thriller scene", "gothic horror atmosphere", "supernatural mystery"]
  }'
```

Repeat this for each category in your taxonomy.

### Step 2: Classify Videos

Once your taxonomy is defined, classify videos by passing the list of possible values:

**API Request:**
```bash
curl -X POST "https://api.coactive.ai/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/mood" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "values": [
      "Dark & Enigmatic",
      "Heroic & Patriotic",
      "Quirky & Comedic",
      "Uplifting & Romantic"
    ]
  }'
```

**Response:**
```json
{
  "moods": ["Dark & Enigmatic"]
}
```

The API analyzes the video and returns which categories match the content.

### Step 3: Store Results as Metadata

Update the video's metadata with the classification results:

```bash
curl -X POST "https://api.coactive.ai/api/v1/ingestion/metadata" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "your-dataset-id",
    "update_assets": [{
      "asset_type": "video",
      "asset_id": "video-id",
      "metadata": {
        "video_narrative_mood": "Dark & Enigmatic",
        "video_narrative_subject": "Heroic & Epic Conflicts"
      }
    }],
    "update_type": "upsert"
  }'
```

---

## Best Practices for Classification Taxonomies

| Guideline | Recommendation |
|-----------|----------------|
| **Number of categories** | 5-15 per type (optimal) |
| **Category names** | Clear, distinct, non-overlapping |
| **Descriptions** | Specific enough to differentiate |
| **Examples** | 2-3 concrete examples per category |

### ❌ Too Granular (Avoid)

```
342 individual moods: "bittersweet", "melancholic", "nostalgic", "wistful", "sad"...
```

**Problem:** These overlap too much - the AI can't reliably distinguish between them.

### ✅ Right Level (Recommended)

```
10 mood categories: "Dark & Enigmatic", "Nostalgic & Dramatic", "Uplifting & Romantic"...
```

**Why it works:** Distinct categories with clear boundaries that the AI can confidently match.

---

## Example Taxonomy: NBCU Emotional Congruence

This taxonomy is designed for TV/Film content to enable emotional congruence in ad placement.

### Mood Categories (10)

| Category | Description |
|----------|-------------|
| **Dark & Enigmatic** | Mysterious, shadowy, noir, supernatural, gothic |
| **Dark Comedy & Gritty Humor** | Irreverent, slapstick, edgy, boundary-pushing comedy |
| **Heroic & Patriotic** | Epic, brave, nationalist, larger-than-life narratives |
| **Mystical & Whimsical** | Surreal, magical realism, dreamlike, psychedelic |
| **Nostalgic & Dramatic** | Emotional, melancholic, sentimental, reflective |
| **Quirky & Comedic** | Lighthearted, goofy, satirical, unconventional |
| **Suspenseful & Adventurous** | High-energy, action, mystery, exotic danger |
| **Thought-Provoking & Intellectual** | Psychological, philosophical, abstract |
| **Thrilling & Sinister** | Intense suspense, fear, danger, creepy |
| **Uplifting & Romantic** | Warm, hopeful, joyful, celebratory |

### Theme Categories (10)

| Category | Description |
|----------|-------------|
| **Coming-of-Age & Life Journeys** | Personal growth, maturity, self-discovery |
| **Dreams & Aspirations** | Hope, fortune, rags-to-riches, pursuing goals |
| **Heroic & Epic Conflicts** | Good vs evil, survival, high-stakes battles |
| **Identity & Hidden Agendas** | Secrets, revenge, mistaken identity, justice |
| **Morality & Inner Conflict** | Ethical dilemmas, redemption, moral gray areas |
| **Partnership & Comedy** | Buddy dynamics, teamwork, camaraderie |
| **Perseverance & Personal Struggle** | Resilience, comebacks, overcoming obstacles |
| **Romance & Emotional Turmoil** | Love, relationships, heartbreak, passion |
| **Societal & Cultural Issues** | Race, class, human rights, environment |
| **Transformation & Identity** | Reinvention, change, discovering true self |

### Genre Categories (6)

| Category | Description |
|----------|-------------|
| **Drama** | Serious narrative fiction with realistic characters and emotional themes |
| **Comedy** | Light-hearted content designed to amuse through humor |
| **Action & Adventure** | High-energy content with physical feats and exciting sequences |
| **Horror & Thriller** | Content designed to frighten or create tension |
| **Sci-Fi & Fantasy** | Speculative fiction with futuristic or magical elements |
| **Documentary & Reality** | Non-fiction content documenting real events or people |

### Format Categories (5)

| Category | Description |
|----------|-------------|
| **Scripted Series** | Episodic fictional content with ongoing narrative |
| **Unscripted & Reality** | Non-scripted content featuring real people |
| **Film & Movie** | Feature-length theatrical or streaming films |
| **News & Sports** | Live or recorded news and sports coverage |
| **Short-Form & Clips** | Brief content including trailers and promos |

---

## Python Code Sample

Here's a complete Python example for setting up and running classification:

```python
import requests
import json

API_BASE = "https://api.coactive.ai"
TOKEN = "your-api-token"
DATASET_ID = "your-dataset-id"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# =============================================================================
# STEP 1: Define taxonomy categories (one-time setup)
# =============================================================================

mood_categories = [
    {
        "name": "Dark & Enigmatic",
        "description": "Mysterious, shadowy, and intense tones blending noir, supernatural, and gothic elements",
        "examples": ["noir thriller scene", "gothic horror atmosphere", "supernatural mystery"]
    },
    {
        "name": "Uplifting & Romantic",
        "description": "Warm, hopeful, and emotionally positive tones celebrating love, joy, and human connection",
        "examples": ["romantic confession", "heartwarming reunion", "triumphant celebration"]
    },
    # ... add remaining categories
]

def setup_taxonomy(metadata_type, categories):
    """Define classification categories for a metadata type."""
    for category in categories:
        payload = {
            "dataset_id": DATASET_ID,
            "metadata_type": metadata_type,
            "name": category["name"],
            "description": category["description"],
            "examples": category["examples"]
        }
        response = requests.post(
            f"{API_BASE}/api/v0/video-narrative-metadata/metadata",
            headers=headers,
            json=payload
        )
        if response.ok:
            print(f"✅ Created: {category['name']}")
        else:
            print(f"❌ Failed: {category['name']} - {response.text}")

# Run once to set up taxonomy
# setup_taxonomy("mood", mood_categories)

# =============================================================================
# STEP 2: Classify videos
# =============================================================================

def classify_video(video_id, metadata_type, values):
    """Classify a video against defined categories."""
    response = requests.post(
        f"{API_BASE}/api/v0/video-narrative-metadata/datasets/{DATASET_ID}/videos/{video_id}/{metadata_type}",
        headers=headers,
        json={"values": values}
    )
    if response.ok:
        return response.json()
    return None

# Get classification
mood_values = ["Dark & Enigmatic", "Uplifting & Romantic", "Quirky & Comedic"]
result = classify_video("video-uuid", "mood", mood_values)
print(f"Detected moods: {result.get('moods', [])}")

# =============================================================================
# STEP 3: Save results to video metadata
# =============================================================================

def save_metadata(video_id, metadata):
    """Update video with classification results."""
    payload = {
        "dataset_id": DATASET_ID,
        "update_assets": [{
            "asset_type": "video",
            "asset_id": video_id,
            "metadata": metadata
        }],
        "update_type": "upsert"
    }
    response = requests.post(
        f"{API_BASE}/api/v1/ingestion/metadata",
        headers=headers,
        json=payload
    )
    return response.ok

# Save classification results
save_metadata("video-uuid", {
    "video_narrative_mood": "Dark & Enigmatic",
    "video_narrative_subject": "Heroic & Epic Conflicts"
})
```

---

## Using the Enrichment Script

The `coactive_narrative_enrichment.py` script automates this entire process:

```bash
# First time: Set up taxonomy categories
python3 coactive_narrative_enrichment.py \
  -d YOUR_DATASET_ID \
  -t YOUR_API_TOKEN \
  --setup-metadata

# Run classification on all videos
python3 coactive_narrative_enrichment.py \
  -d YOUR_DATASET_ID \
  -t YOUR_API_TOKEN

# Run with parallel processing (faster)
python3 coactive_narrative_enrichment.py \
  -d YOUR_DATASET_ID \
  -t YOUR_API_TOKEN \
  --workers 5
```

### Script Options

| Flag | Description |
|------|-------------|
| `-d, --dataset-id` | Coactive Dataset ID (required) |
| `-t, --token` | Coactive API Token (required) |
| `--setup-metadata` | Create taxonomy categories before enrichment |
| `--setup-only` | Only create categories, don't run enrichment |
| `--segments` | Enable scene/chapter detection |
| `--entities` | Enable entity extraction (people, brands) |
| `--workers N` | Number of parallel workers (default: 1) |
| `--video-id` | Process only a specific video |
| `--intent` | Custom summary intent |

---

## API Reference

### Classification Endpoints

| Action | Method | Endpoint |
|--------|--------|----------|
| Define category | POST | `/api/v0/video-narrative-metadata/metadata` |
| List categories | GET | `/api/v0/video-narrative-metadata/metadata?metadata_type={type}` |
| Classify mood | POST | `/api/v0/video-narrative-metadata/datasets/{id}/videos/{id}/mood` |
| Classify subject | POST | `/api/v0/video-narrative-metadata/datasets/{id}/videos/{id}/subject` |
| Classify genre | POST | `/api/v0/video-narrative-metadata/datasets/{id}/videos/{id}/genre` |
| Classify format | POST | `/api/v0/video-narrative-metadata/datasets/{id}/videos/{id}/format` |

### Other Narrative Metadata Endpoints

| Action | Method | Endpoint |
|--------|--------|----------|
| Get summary | POST | `/api/v0/video-summarization/datasets/{id}/videos/{id}/summarize` |
| Get description | POST | `/api/v0/video-narrative-metadata/datasets/{id}/videos/{id}/description` |
| Caption keyframes | POST | `/api/v0/video-summarization/datasets/{id}/videos/{id}/caption-keyframes` |

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DEFINE TAXONOMY (once per dataset)                          │
│     └── POST /metadata for each category                        │
│         • mood: 10 categories                                   │
│         • subject: 10 categories                                │
│         • genre: 6 categories                                   │
│         • format: 5 categories                                  │
│                                                                  │
│  2. CLASSIFY VIDEOS (for each video)                            │
│     └── POST /videos/{id}/mood                                  │
│     └── POST /videos/{id}/subject                               │
│     └── POST /videos/{id}/genre                                 │
│     └── POST /videos/{id}/format                                │
│                                                                  │
│  3. STORE RESULTS                                                │
│     └── POST /ingestion/metadata                                │
│         • video_narrative_mood                                  │
│         • video_narrative_subject                               │
│         • video_narrative_genre                                 │
│         • video_narrative_format                                │
│                                                                  │
│  4. USE METADATA                                                 │
│     └── Filter in Coactive UI                                   │
│     └── Power ad placement (emotional congruence)               │
│     └── Enable content discovery                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Questions?

For API documentation: https://docs.coactive.ai/api-reference

For support: Contact your Coactive account team
