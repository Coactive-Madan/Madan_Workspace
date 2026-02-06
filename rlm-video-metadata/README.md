# RLM Video Metadata Enrichment

An intelligent video metadata enrichment pipeline using **Recursive Language Model (RLM)** patterns to optimize API usage when processing large video datasets.

## Overview

Traditional video metadata enrichment runs every enrichment type on every video, which is wasteful and slow. This pipeline uses a 3-wave approach:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WAVE 1: Foundation (runs on all videos)                                     │
│   - Quick summary (cheap, fast, required for clustering)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ WAVE 2: RLM Analysis (samples corpus)                                       │
│   - Sample 10-20% of videos                                                 │
│   - Cluster by content type (movies, TV, news, etc.)                        │
│   - Determine which enrichments are worth running per cluster               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ WAVE 3: Targeted Enrichment (runs only what's needed)                       │
│   - Process by cluster with shared context                                  │
│   - Skip enrichments that don't apply (e.g., no entities on news)           │
│   - Share detected entities across related videos                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Benefits

- **Reduced API Calls**: Only run enrichments that make sense for each content type
- **Faster Processing**: Skip unnecessary work based on corpus analysis
- **Smarter Context**: Share detected entities across related videos
- **Cost Savings**: Typically 30-50% fewer API calls compared to naive approach

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rlm-video-metadata.git
cd rlm-video-metadata

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Implement the VideoAPIClient

The pipeline is platform-agnostic. Implement the `VideoAPIClient` abstract class for your video platform:

```python
from rlm_video_enrichment import VideoAPIClient, Config

class MyPlatformClient(VideoAPIClient):
    def __init__(self, config: Config):
        self.api_key = config.api_key
        self.base_url = config.api_base_url

    def get_videos(self, dataset_id: str, limit: int = 500):
        # Fetch videos from your platform
        pass

    def get_summary(self, video_id: str, intent: str = ""):
        # Get video summary/description
        pass

    def get_classification(self, video_id: str, classification_type: str, values=None):
        # Get genre, mood, subject, format classifications
        pass

    def get_entities(self, video_id: str):
        # Extract people, objects, brands from video
        pass

    def get_segments(self, video_id: str):
        # Get video segments/chapters
        pass

    def save_metadata(self, video_id: str, metadata: dict):
        # Save enriched metadata back to platform
        pass
```

### 2. Configure the Pipeline

Create a `config.json`:

```json
{
    "api_base_url": "https://api.your-platform.com",
    "api_key": "your-api-key",
    "llm_api_key": "your-openai-key",
    "llm_model": "gpt-4o-mini",
    "batch_size": 500,
    "rate_limit_delay": 0.3,
    "sample_size_for_analysis": 20
}
```

Or use environment variables:

```bash
export VIDEO_API_BASE_URL="https://api.your-platform.com"
export VIDEO_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"  # Optional, enables RLM analysis
```

### 3. Run the Pipeline

```bash
# Full enrichment
python rlm_video_enrichment.py --config config.json --dataset-id YOUR_DATASET_ID

# Analyze only (see what clusters would be created)
python rlm_video_enrichment.py --config config.json --dataset-id YOUR_DATASET_ID --analyze-only

# Force specific enrichments
python rlm_video_enrichment.py --config config.json --dataset-id YOUR_DATASET_ID \
    --force-enrichments summary,genre,entities

# Dry run (no saves)
python rlm_video_enrichment.py --config config.json --dataset-id YOUR_DATASET_ID --no-save
```

## Available Enrichment Types

| Type | Description | Best For |
|------|-------------|----------|
| `summary` | Full narrative summary | All content |
| `description` | Detailed description | All content |
| `genre` | Genre classification | Entertainment, mixed libraries |
| `mood` | Emotional tone | Entertainment, music |
| `subject` | Topic classification | News, educational |
| `format` | Content format | Varied content types |
| `entities` | People/celebrity extraction | Entertainment, news |
| `segments` | Scene/chapter detection | Long-form content |
| `keyframes` | Key frame extraction | Visual analysis |

## How RLM Analysis Works

When `OPENAI_API_KEY` is set, the pipeline uses an LLM to:

1. **Sample** 10-20% of videos from the dataset
2. **Analyze** content patterns and types
3. **Cluster** videos by similarity (news, entertainment, music, etc.)
4. **Plan** which enrichments to run for each cluster

Example RLM output:
```json
{
  "clusters": {
    "entertainment": {
      "video_patterns": ["movie", "film", "show"],
      "enrichments": ["summary", "genre", "mood", "entities"],
      "reasoning": "Entertainment content benefits from genre/mood classification and entity extraction"
    },
    "news": {
      "video_patterns": ["news", "report", "breaking"],
      "enrichments": ["summary", "subject"],
      "reasoning": "News content needs subject classification but not mood/genre"
    }
  }
}
```

Without an LLM key, the pipeline falls back to keyword-based heuristic clustering.

## Programmatic Usage

```python
from rlm_video_enrichment import (
    RLMEnrichmentPipeline,
    Config,
    EnrichmentType
)

# Your custom API client
from my_platform import MyPlatformClient

# Setup
config = Config.from_file("config.json")
api_client = MyPlatformClient(config)
pipeline = RLMEnrichmentPipeline(api_client, config)

# Run
result = pipeline.run(
    dataset_id="my-dataset-123",
    limit=100,
    analyze_only=False,
    force_enrichments=None,  # or [EnrichmentType.SUMMARY, EnrichmentType.GENRE]
    save_results=True
)

print(f"Processed {result['success_count']} videos in {result['elapsed_seconds']:.1f}s")
```

## CLI Options

```
usage: rlm_video_enrichment.py [-h] --dataset-id DATASET_ID [--config CONFIG]
                                [--limit LIMIT] [--analyze-only] [--skip-wave1]
                                [--skip-wave2] [--force-enrichments FORCE_ENRICHMENTS]
                                [--no-save] [--verbose] [--nbcu-taxonomy]

Options:
  --dataset-id, -d    Dataset ID to process (required)
  --config, -c        Path to config JSON file
  --limit, -l         Max videos to process (default: 500)
  --analyze-only      Only analyze corpus, no enrichment
  --skip-wave1        Skip Wave 1 (summaries)
  --skip-wave2        Skip Wave 2 (analysis)
  --force-enrichments Comma-separated: summary,genre,entities
  --no-save           Dry run - do not save results
  --verbose, -v       Verbose logging
  --nbcu-taxonomy     Use NBCU Emotional Congruence taxonomy (see below)
```

---

## NBCU Emotional Congruence Taxonomy

Use `--nbcu-taxonomy` flag to enable optimized classification for TV/Film content, designed for ad placement and emotional congruence use cases.

```bash
# Run with NBCU taxonomy
python rlm_video_enrichment.py --dataset-id abc123 --nbcu-taxonomy

# Or set in config.json
{
    "use_nbcu_taxonomy": true
}

# Or via environment variable
export USE_NBCU_TAXONOMY=true
```

### NBCU Taxonomy Categories

**10 Mood Categories:**
| Category | Description |
|----------|-------------|
| Dark & Enigmatic | Noir, supernatural, gothic, mysterious |
| Dark Comedy & Gritty Humor | Irreverent, edgy, boundary-pushing comedy |
| Heroic & Patriotic | Epic, brave, larger-than-life narratives |
| Mystical & Whimsical | Magical realism, dreamlike, surreal |
| Nostalgic & Dramatic | Emotional, melancholic, sentimental |
| Quirky & Comedic | Lighthearted, goofy, satirical |
| Suspenseful & Adventurous | High-energy action, mystery, danger |
| Thought-Provoking & Intellectual | Psychological, philosophical |
| Thrilling & Sinister | Intense suspense, fear, creepy |
| Uplifting & Romantic | Warm, hopeful, joyful |

**10 Theme Categories:**
| Category | Description |
|----------|-------------|
| Coming-of-Age & Life Journeys | Personal growth, self-discovery |
| Dreams & Aspirations | Pursuit of fortune, rags-to-riches |
| Heroic & Epic Conflicts | Good vs evil, survival, high stakes |
| Identity & Hidden Agendas | Secrets, revenge, mistaken identity |
| Morality & Inner Conflict | Ethical dilemmas, redemption |
| Partnership & Comedy | Buddy dynamics, teamwork |
| Perseverance & Personal Struggle | Resilience, comebacks |
| Romance & Emotional Turmoil | Love struggles, relationships |
| Societal & Cultural Issues | Race, class, environment |
| Transformation & Identity | Reinvention, self-discovery |

**8 Genre Categories:** Drama, Comedy, Action & Adventure, Horror & Thriller, Sci-Fi & Fantasy, Reality & Competition, Crime & Legal, Period & Historical

**5 Format Categories:** Scripted Drama Series, Scripted Comedy Series, Reality Competition, Animated Series, Limited Series & Prestige TV

### NBCU-Optimized Clustering

When `--nbcu-taxonomy` is enabled, the pipeline automatically clusters content by type:

| Cluster | Content Types | Enrichments | Rationale |
|---------|---------------|-------------|-----------|
| `scripted_drama` | Medical, legal, crime dramas | Full + entities + segments | Celebrity detection, scene-level targeting |
| `scripted_comedy` | Sitcoms, animated comedy | Full (no segments) | Mood focus, less structured narrative |
| `reality_competition` | Competition shows, dating | Full (no entities) | No scripted celebrities |
| `action_adventure` | Superhero, action thrillers | Full + entities + segments | Full ad placement support |
| `period_historical` | Historical dramas | Full + entities | Historical figure detection |
| `horror_thriller` | Horror, suspense | Full + segments | Scene-level ad safety |

---

## Requirements

- Python 3.8+
- `openai` (optional, for RLM analysis)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a PR.
