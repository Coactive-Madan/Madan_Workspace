#!/usr/bin/env python3
"""
Audio Processing Script - Downmix 5.1/5.0 to Stereo
Processes extracted audio files from Nielsen dataset and re-uploads to S3
"""

import os
import subprocess
import json
from pathlib import Path

# Configuration
INPUT_DIR = "/Users/madan/Downloads/audio_processing"
OUTPUT_DIR = "/Users/madan/Downloads/audio_processing_output"
FFMPEG = "/Users/madan/Downloads/ffmpeg"
FFPROBE = "/Users/madan/Downloads/ffprobe"

S3_BUCKET = "s3://coactive-ingestion-transient-production/org_sgVH8OmDi0mpgIxc/coactive/Nielsen_Data_V2_1764649381520/extracted_audio/"

DRY_RUN = False  # Set to False to actually process and upload

def get_audio_channels(filepath):
    """Get number of audio channels in file."""
    try:
        result = subprocess.run([
            FFPROBE, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels",
            "-of", "json",
            filepath
        ], capture_output=True, text=True)
        
        data = json.loads(result.stdout)
        if data.get("streams"):
            return data["streams"][0].get("channels", 0)
        return 0
    except Exception as e:
        print(f"  ⚠️ Error analyzing {filepath}: {e}")
        return 0

def downmix_to_stereo(input_path, output_path):
    """Downmix audio to stereo using FFmpeg."""
    try:
        cmd = [
            FFMPEG, "-y",
            "-i", input_path,
            "-ac", "2",  # Convert to 2 channels (stereo)
            "-c:a", "aac",  # AAC codec
            "-b:a", "192k",  # 192kbps bitrate
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"  ❌ FFmpeg error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def upload_to_s3(local_path, s3_path):
    """Upload file to S3."""
    try:
        cmd = ["aws", "s3", "cp", local_path, s3_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"  ❌ S3 upload error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ Upload exception: {e}")
        return False

def main():
    print("=" * 60)
    print("🎵 Nielsen Audio Downmix Processor")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"S3 destination: {S3_BUCKET}")
    print(f"Dry run: {DRY_RUN}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all m4a files
    input_files = list(Path(INPUT_DIR).glob("*.m4a"))
    print(f"\n📁 Found {len(input_files)} audio files\n")
    
    stats = {
        "already_stereo": 0,
        "downmixed": 0,
        "uploaded": 0,
        "errors": 0
    }
    
    for i, input_path in enumerate(input_files, 1):
        filename = input_path.name
        print(f"[{i}/{len(input_files)}] Processing: {filename}")
        
        # Check channel count
        channels = get_audio_channels(str(input_path))
        print(f"  📊 Channels: {channels}")
        
        if channels <= 2:
            print(f"  ✅ Already stereo/mono - skipping")
            stats["already_stereo"] += 1
            continue
        
        # Need to downmix
        output_path = Path(OUTPUT_DIR) / filename
        
        if DRY_RUN:
            print(f"  🔄 [DRY RUN] Would downmix {channels}ch → 2ch")
            print(f"  📤 [DRY RUN] Would upload to S3")
            stats["downmixed"] += 1
            stats["uploaded"] += 1
        else:
            # Downmix
            print(f"  🔄 Downmixing {channels}ch → 2ch...")
            if downmix_to_stereo(str(input_path), str(output_path)):
                stats["downmixed"] += 1
                
                # Verify output
                new_channels = get_audio_channels(str(output_path))
                print(f"  ✓ Output channels: {new_channels}")
                
                # Upload to S3
                print(f"  📤 Uploading to S3...")
                s3_dest = S3_BUCKET + filename
                if upload_to_s3(str(output_path), s3_dest):
                    print(f"  ✅ Uploaded successfully")
                    stats["uploaded"] += 1
                else:
                    stats["errors"] += 1
            else:
                stats["errors"] += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Already stereo (skipped): {stats['already_stereo']}")
    print(f"Downmixed to stereo:      {stats['downmixed']}")
    print(f"Uploaded to S3:           {stats['uploaded']}")
    print(f"Errors:                   {stats['errors']}")
    print("=" * 60)

if __name__ == "__main__":
    main()




