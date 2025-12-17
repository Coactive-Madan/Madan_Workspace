#!/usr/bin/env python3
"""
Process Failed Audio Files
===========================
Downloads failed files from S3, checks audio configuration, and:
- If stereo (2.0): strips Spanish audio track
- If 5.1: downmixes to stereo AND strips Spanish track

Requires:
- AWS CLI configured with access to the S3 bucket
- ffmpeg and ffprobe in the same directory

Author: Madan
Date: December 2025
"""

import subprocess
import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
FFMPEG_PATH = "/Users/madan/Downloads/ffmpeg"
FFPROBE_PATH = "/Users/madan/Downloads/ffprobe"
CSV_FILE = "/Users/madan/Downloads/results-6ed5f51a-3a2d-4744-bd78-8376f9d446a2.csv"
WORK_DIR = "/Users/madan/Downloads/audio_processing"
OUTPUT_DIR = "/Users/madan/Downloads/audio_processed"

# S3 Configuration
S3_BUCKET = "coactive-ai-video-samples"
OUTPUT_S3_BUCKET = "coactive-ai-video-samples"  # Same bucket, different prefix
OUTPUT_S3_PREFIX = "processed/"

# Processing Options
DRY_RUN = True  # Set to False to actually process files
MAX_FILES = 3   # Process only first N files for testing (set to None for all)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def get_audio_info(filepath: str) -> Optional[Dict]:
    """
    Use ffprobe to get audio track information.
    
    Returns dict with:
    - streams: list of audio streams with channel info, language
    - is_stereo: True if any stream is stereo
    - is_51: True if any stream is 5.1
    - spanish_tracks: list of indices for Spanish audio tracks
    """
    cmd = [
        FFPROBE_PATH,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",
        filepath
    ]
    
    exit_code, stdout, stderr = run_command(cmd)
    
    if exit_code != 0:
        print(f"  ❌ ffprobe failed: {stderr}")
        return None
    
    try:
        data = json.loads(stdout)
        streams = data.get("streams", [])
        
        info = {
            "streams": [],
            "is_stereo": False,
            "is_51": False,
            "spanish_tracks": [],
            "english_tracks": [],
            "total_audio_tracks": len(streams)
        }
        
        for i, stream in enumerate(streams):
            channels = stream.get("channels", 0)
            channel_layout = stream.get("channel_layout", "unknown")
            
            # Get language from tags
            tags = stream.get("tags", {})
            language = tags.get("language", "").lower()
            
            stream_info = {
                "index": stream.get("index"),
                "stream_index": i,
                "channels": channels,
                "channel_layout": channel_layout,
                "language": language,
                "codec": stream.get("codec_name", "unknown")
            }
            
            info["streams"].append(stream_info)
            
            # Check if stereo or 5.1
            if channels == 2:
                info["is_stereo"] = True
            elif channels >= 5:  # 5.1 has 6 channels
                info["is_51"] = True
            
            # Track Spanish and English
            if "spa" in language or "esp" in language or language == "es":
                info["spanish_tracks"].append(i)
            elif "eng" in language or language == "en" or language == "":
                info["english_tracks"].append(i)
        
        return info
        
    except json.JSONDecodeError as e:
        print(f"  ❌ Failed to parse ffprobe output: {e}")
        return None


def process_audio(input_file: str, output_file: str, audio_info: Dict) -> bool:
    """
    Process audio based on configuration:
    - Stereo (2.0): Just strip Spanish track
    - 5.1: Downmix to stereo AND strip Spanish track
    """
    cmd = [FFMPEG_PATH, "-y", "-i", input_file]
    
    # Build mapping command
    # Always copy video
    cmd.extend(["-map", "0:v"])
    
    has_spanish = len(audio_info["spanish_tracks"]) > 0
    has_51 = audio_info["is_51"]
    
    if has_51:
        # 5.1 audio: downmix to stereo
        print(f"  🔊 Found 5.1 audio - will downmix to stereo")
        
        # Map non-Spanish audio tracks and downmix
        for stream in audio_info["streams"]:
            if stream["stream_index"] not in audio_info["spanish_tracks"]:
                stream_idx = stream["stream_index"]
                cmd.extend(["-map", f"0:a:{stream_idx}"])
        
        # Downmix to stereo
        cmd.extend(["-ac", "2"])
        
    else:
        # Stereo: just map non-Spanish tracks
        print(f"  🔊 Stereo audio - stripping Spanish track")
        
        for stream in audio_info["streams"]:
            if stream["stream_index"] not in audio_info["spanish_tracks"]:
                stream_idx = stream["stream_index"]
                cmd.extend(["-map", f"0:a:{stream_idx}"])
    
    # Copy codecs (no re-encoding for video)
    cmd.extend(["-c:v", "copy"])
    
    # Audio encoding (needed for downmix)
    if has_51:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.extend(["-c:a", "copy"])
    
    cmd.append(output_file)
    
    print(f"  📝 FFmpeg command: {' '.join(cmd[:10])}...")
    
    if DRY_RUN:
        print(f"  [DRY RUN] Would run ffmpeg")
        return True
    
    exit_code, stdout, stderr = run_command(cmd, capture_output=True)
    
    if exit_code != 0:
        print(f"  ❌ FFmpeg failed: {stderr[-500:] if stderr else 'Unknown error'}")
        return False
    
    return True


def extract_unique_s3_paths() -> List[str]:
    """Extract unique S3 paths from the failed uploads CSV."""
    paths = set()
    
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get('source_path', '').strip()
            if path and path.startswith('s3://'):
                paths.add(path)
    
    return sorted(list(paths))


def s3_path_to_local(s3_path: str) -> Tuple[str, str]:
    """Convert S3 path to local paths for download and processed output."""
    # s3://bucket/filename.ts -> filename.ts
    filename = s3_path.split('/')[-1]
    
    input_path = os.path.join(WORK_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    return input_path, output_path


def download_from_s3(s3_path: str, local_path: str) -> bool:
    """Download file from S3."""
    print(f"  ⬇️  Downloading from S3...")
    
    if DRY_RUN:
        print(f"  [DRY RUN] Would download {s3_path}")
        return True
    
    cmd = ["aws", "s3", "cp", s3_path, local_path]
    exit_code, stdout, stderr = run_command(cmd)
    
    if exit_code != 0:
        print(f"  ❌ Download failed: {stderr}")
        return False
    
    print(f"  ✓ Downloaded to {local_path}")
    return True


def upload_to_s3(local_path: str, s3_path: str) -> bool:
    """Upload processed file back to S3."""
    print(f"  ⬆️  Uploading to S3...")
    
    if DRY_RUN:
        print(f"  [DRY RUN] Would upload to {s3_path}")
        return True
    
    cmd = ["aws", "s3", "cp", local_path, s3_path]
    exit_code, stdout, stderr = run_command(cmd)
    
    if exit_code != 0:
        print(f"  ❌ Upload failed: {stderr}")
        return False
    
    print(f"  ✓ Uploaded to {s3_path}")
    return True


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_single_file(s3_path: str) -> Dict:
    """Process a single file from S3."""
    result = {
        "s3_path": s3_path,
        "status": "unknown",
        "audio_info": None,
        "action": None
    }
    
    print(f"\n📹 Processing: {s3_path}")
    
    # Get local paths
    input_path, output_path = s3_path_to_local(s3_path)
    
    # Download from S3
    if not download_from_s3(s3_path, input_path):
        result["status"] = "download_failed"
        return result
    
    # Analyze audio (for dry run, we need a real file, so skip if dry run)
    if DRY_RUN:
        print(f"  [DRY RUN] Would analyze audio with ffprobe")
        result["status"] = "dry_run"
        result["action"] = "would_process"
        return result
    
    # Get audio info
    audio_info = get_audio_info(input_path)
    
    if audio_info is None:
        result["status"] = "probe_failed"
        return result
    
    result["audio_info"] = audio_info
    
    # Print audio analysis
    print(f"  📊 Audio Analysis:")
    print(f"     Total audio tracks: {audio_info['total_audio_tracks']}")
    print(f"     Is stereo (2.0): {audio_info['is_stereo']}")
    print(f"     Is 5.1: {audio_info['is_51']}")
    print(f"     Spanish tracks: {audio_info['spanish_tracks']}")
    print(f"     English tracks: {audio_info['english_tracks']}")
    
    for i, stream in enumerate(audio_info["streams"]):
        print(f"     Track {i}: {stream['channels']} ch ({stream['channel_layout']}), lang={stream['language']}")
    
    # Determine action
    if not audio_info["spanish_tracks"]:
        print(f"  ℹ️  No Spanish track found - checking if downmix needed")
        if audio_info["is_51"]:
            result["action"] = "downmix_only"
        else:
            result["action"] = "no_change"
            result["status"] = "no_action_needed"
            return result
    else:
        if audio_info["is_51"]:
            result["action"] = "downmix_and_strip_spanish"
        else:
            result["action"] = "strip_spanish"
    
    # Process audio
    success = process_audio(input_path, output_path, audio_info)
    
    if success:
        result["status"] = "processed"
        
        # Upload processed file
        output_s3_path = f"s3://{OUTPUT_S3_BUCKET}/{OUTPUT_S3_PREFIX}{os.path.basename(output_path)}"
        if upload_to_s3(output_path, output_s3_path):
            result["status"] = "uploaded"
            result["output_s3_path"] = output_s3_path
    else:
        result["status"] = "processing_failed"
    
    return result


def main():
    """Main execution."""
    print("=" * 80)
    print("AUDIO PROCESSING FOR FAILED UPLOADS")
    print("Strip Spanish tracks & downmix 5.1 to stereo")
    print("=" * 80)
    print()
    
    if DRY_RUN:
        print("⚠️  DRY RUN MODE - No files will be downloaded or processed")
        print()
    
    # Create directories
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check ffmpeg/ffprobe
    if not os.path.exists(FFMPEG_PATH):
        print(f"❌ ffmpeg not found at {FFMPEG_PATH}")
        return
    if not os.path.exists(FFPROBE_PATH):
        print(f"❌ ffprobe not found at {FFPROBE_PATH}")
        return
    
    print(f"✓ ffmpeg: {FFMPEG_PATH}")
    print(f"✓ ffprobe: {FFPROBE_PATH}")
    
    # Check AWS CLI
    exit_code, stdout, stderr = run_command(["aws", "--version"])
    if exit_code != 0:
        print("❌ AWS CLI not found - please install and configure it")
        print("   Install: pip install awscli")
        print("   Configure: aws configure")
        return
    
    print(f"✓ AWS CLI available")
    print()
    
    # Extract failed S3 paths
    print(f"📂 Reading failed uploads from: {CSV_FILE}")
    s3_paths = extract_unique_s3_paths()
    print(f"✓ Found {len(s3_paths)} unique failed S3 paths")
    print()
    
    # Limit for testing
    if MAX_FILES:
        s3_paths = s3_paths[:MAX_FILES]
        print(f"📊 Processing first {len(s3_paths)} files for testing")
        print()
    
    # Process each file
    results = []
    
    print("=" * 80)
    print("PROCESSING FILES")
    print("=" * 80)
    
    for idx, s3_path in enumerate(s3_paths, 1):
        print(f"\n[{idx}/{len(s3_paths)}]")
        result = process_single_file(s3_path)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\n📊 Results:")
    for status, count in sorted(status_counts.items()):
        print(f"   {status}: {count}")
    
    print(f"\n📁 Work directory: {WORK_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    if DRY_RUN:
        print("\n⚠️  This was a DRY RUN - no files were actually processed")
        print("   Set DRY_RUN = False to process files for real")
    
    print("=" * 80)


if __name__ == "__main__":
    main()




