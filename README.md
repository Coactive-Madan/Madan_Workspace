# Madan Workspace

Collection of Coactive AI utility scripts and tools.

## Projects

| Directory | Description |
|-----------|-------------|
| [narrative-enrichment](./narrative-enrichment) | Generate AI metadata (summary, genre, mood, keyframes) |
| [imdb-enrichment](./imdb-enrichment) | Enrich assets with IMDB metadata |
| [hybe-celebrity-enrollment](./hybe-celebrity-enrollment) | Enroll celebrities for face detection |
| [audio-processing](./audio-processing) | Audio downmix and track processing tools |

## Quick Start

### Narrative Enrichment
```bash
cd narrative-enrichment
python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --setup-metadata
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

## Author

Madan - Coactive AI
