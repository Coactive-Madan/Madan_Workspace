# Madan Workspace

Collection of Coactive AI utility scripts and tools.

## Projects

| Directory | Description |
|-----------|-------------|
| [narrative-enrichment](./narrative-enrichment) | Generate AI metadata (summary, genre, mood, keyframes) |
| [rlm-video-metadata](./rlm-video-metadata) | RLM-optimized video enrichment (platform-agnostic) |
| [imdb-enrichment](./imdb-enrichment) | Enrich assets with IMDB metadata |
| [hybe-celebrity-enrollment](./hybe-celebrity-enrollment) | Enroll celebrities for face detection |
| [audio-processing](./audio-processing) | Audio downmix and track processing tools |
| [s3-ingestion](./s3-ingestion) | Google Drive → S3 copy + Coactive dataset ingestion |
| [dynamic-tags](./dynamic-tags) | Generate & push optimized prompts for Dynamic Tags (v3 API) |

## Quick Start

### Narrative Enrichment
```bash
cd narrative-enrichment
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --setup-metadata
```

### RLM Video Metadata (Platform-Agnostic)
```bash
cd rlm-video-metadata
# Configure your video platform API
cp config.example.json config.json
python3 rlm_video_enrichment.py --config config.json --dataset-id YOUR_DATASET_ID
```

### IMDB Enrichment
```bash
cd imdb-enrichment
export COACTIVE_API_KEY="your_token"
export COACTIVE_DATASET_ID="your_dataset_id"
python3 coactive_imdb_enrichment.py
```

### Celebrity Enrollment
```bash
cd hybe-celebrity-enrollment
python3 hybe_celebrity_enrollment.py
```

### Audio Processing
```bash
cd audio-processing
python3 process_audio_downmix.py
```

### Dynamic Tags (Generate & Push)
```bash
cd dynamic-tags
export COACTIVE_API_TOKEN="your_token"
export OPENAI_API_KEY="your_openai_key"

# End-to-end: fetch tags → generate prompts → push → publish
python3 dynamic_tags.py run \
    --group-url "https://app.coactive.ai/dynamic-tags/groups/<gid>/versions/<vid>" \
    --publish

# Or two-step: generate offline, then push
python3 dynamic_tags.py generate --input tags.json --output prompts.json
python3 dynamic_tags.py push --group-url "..." --prompts prompts.json --publish
```

## Author

Madan - Coactive AI
