# Audio Processing Scripts

Tools for processing audio in video files for Coactive ingestion, including downmixing 5.1 surround to stereo and stripping unwanted audio tracks.

## Scripts

### 1. `process_audio_downmix.py`

Batch process audio files to downmix 5.1/5.0 surround sound to stereo for Nielsen dataset ingestion.

**Features:**
- Analyzes audio channel configuration using ffprobe
- Skips already-stereo files
- Downmixes multi-channel audio to stereo
- Uploads processed files to S3

**Usage:**
```bash
python process_audio_downmix.py
```

**Configuration:**
| Variable | Description |
|----------|-------------|
| `INPUT_DIR` | Directory containing source audio files |
| `OUTPUT_DIR` | Directory for processed output |
| `FFMPEG` | Path to ffmpeg binary |
| `S3_BUCKET` | S3 destination for uploads |
| `DRY_RUN` | Set `False` to actually process |

---

### 2. `process_failed_audio.py`

Process failed video uploads by fixing audio track issues:
- Strip Spanish audio tracks
- Downmix 5.1 surround to stereo
- Re-upload corrected files to S3

**Features:**
- Reads failed uploads from CSV export
- Full audio analysis with ffprobe
- Intelligent track selection
- S3 download/upload automation

**Usage:**
```bash
# Edit configuration, then run
python process_failed_audio.py
```

**Configuration:**
| Variable | Description |
|----------|-------------|
| `CSV_FILE` | Path to failed uploads CSV |
| `WORK_DIR` | Temporary directory for downloads |
| `OUTPUT_DIR` | Directory for processed files |
| `S3_BUCKET` | Source S3 bucket |
| `OUTPUT_S3_PREFIX` | Prefix for processed files |
| `DRY_RUN` | Set `False` to process |
| `MAX_FILES` | Limit files for testing |

## Audio Processing Logic

```
Input Analysis:
├── Stereo (2.0) with Spanish track → Strip Spanish only
├── 5.1 Surround → Downmix to stereo
└── 5.1 with Spanish → Downmix AND strip Spanish
```

## FFmpeg Commands Used

**Downmix to stereo:**
```bash
ffmpeg -i input.m4a -ac 2 -c:a aac -b:a 192k output.m4a
```

**Strip audio track:**
```bash
ffmpeg -i input.ts -map 0:v -map 0:a:0 -c:v copy -c:a copy output.ts
```

## Requirements

### System Dependencies
- **ffmpeg** - Audio/video processing
- **ffprobe** - Media analysis
- **AWS CLI** - S3 operations

### Python
```
No external Python packages required (uses stdlib only)
```

### AWS Configuration
```bash
# Configure AWS CLI with credentials
aws configure

# Verify access
aws s3 ls s3://your-bucket/
```

## Output

**Stats tracked:**
- Files already stereo (skipped)
- Files downmixed
- Files uploaded to S3
- Processing errors

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ffmpeg not found | Update `FFMPEG_PATH` in script |
| S3 permission denied | Check AWS credentials & bucket policy |
| Processing fails | Check ffmpeg stderr output |

## Author

Madan - December 2025

