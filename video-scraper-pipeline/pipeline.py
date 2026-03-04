#!/usr/bin/env python3
"""
YouTube Video Scraper Pipeline — Coactive-Compliant
====================================================
Generic, reusable pipeline to search, download, validate, and prepare
video datasets for ingestion into Coactive.

All outputs comply with Coactive's accepted media formats:
  - Video: H.264 / AAC / .mp4 / <=720p / 30fps
  - Frames: .jpg / >=350x350px

Usage:
    python3 pipeline.py --init                  # Create example targets.json
    python3 pipeline.py --dry-run               # Preview without downloading
    python3 pipeline.py                         # Full pipeline (all tiers)
    python3 pipeline.py --tier 1                # Single tier only
    python3 pipeline.py --search-only           # List found URLs
    python3 pipeline.py --validate-only         # Validate existing files
    python3 pipeline.py --extract-frames        # Extract jpg frames
    python3 pipeline.py --manifest-only         # Regenerate CSV manifest
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────────────
# All paths are relative to the script location. Override via targets.json
# or CLI flags — no hardcoded bucket names or credentials.

BASE_DIR = Path(__file__).parent.resolve()
RAW_DIR = BASE_DIR / "raw_videos"
FRAMES_DIR = BASE_DIR / "frames"
CONVERTED_DIR = BASE_DIR / "converted"
METADATA_DIR = BASE_DIR / "metadata"
LOG_DIR = BASE_DIR / "logs"
ARCHIVE_FILE = BASE_DIR / "downloaded.txt"
TARGETS_FILE = BASE_DIR / "targets.json"

# Coactive-compliant download settings
MAX_HEIGHT = 720
VIDEO_CODEC = "avc1"  # H.264
AUDIO_CODEC = "mp4a"  # AAC
TARGET_FPS = 30
CONTAINER = "mp4"

# Frame extraction defaults
FRAME_INTERVAL_SEC = 5   # 1 frame every N seconds
FRAME_MIN_PX = 360       # minimum dimension for frames

# yt-dlp format string for Coactive compliance
YT_DLP_FORMAT = (
    f"bestvideo[height<={MAX_HEIGHT}][vcodec^={VIDEO_CODEC}]"
    f"+bestaudio[acodec^={AUDIO_CODEC}]"
    f"/best[height<={MAX_HEIGHT}]"
)

# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class VideoTarget:
    """A single video search/download target."""
    campaign: str
    search_query: str
    product_line: str
    tier: int
    category: str       # subfolder name (e.g., "brand_ads", "competitor_ads")
    max_results: int = 3
    url: Optional[str] = None  # direct URL — skips search if provided

# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("pipeline")


log = setup_logging()

# ─── Target Loading ──────────────────────────────────────────────────────────

def load_targets(tier_filter: Optional[int] = None) -> list[VideoTarget]:
    """Load video targets from targets.json."""
    if not TARGETS_FILE.exists():
        log.error(f"Targets file not found: {TARGETS_FILE}")
        log.info("Run with --init to create an example targets.json")
        sys.exit(1)

    with open(TARGETS_FILE) as f:
        data = json.load(f)

    targets = []
    for entry in data["targets"]:
        t = VideoTarget(**entry)
        if tier_filter is None or t.tier == tier_filter:
            targets.append(t)

    log.info(f"Loaded {len(targets)} targets" +
             (f" (tier {tier_filter})" if tier_filter else ""))
    return targets

# ─── Phase 1: Search ────────────────────────────────────────────────────────

def search_videos(target: VideoTarget, dry_run: bool = False) -> list[dict]:
    """Search YouTube for videos matching a target query. Returns metadata."""
    log.info(f"Searching: '{target.search_query}' (max {target.max_results})")

    cmd = [
        "yt-dlp",
        f"ytsearch{target.max_results}:{target.search_query}",
        "--dump-json",
        "--no-download",
        "--flat-playlist",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            log.warning(f"Search failed for '{target.search_query}': {result.stderr[:200]}")
            return []

        videos = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    info = json.loads(line)
                    videos.append({
                        "url": (
                            info.get("url")
                            or info.get("webpage_url")
                            or f"https://www.youtube.com/watch?v={info.get('id', '')}"
                        ),
                        "title": info.get("title", "Unknown"),
                        "duration": info.get("duration", 0),
                        "uploader": info.get("uploader", "Unknown"),
                        "upload_date": info.get("upload_date", ""),
                        "id": info.get("id", ""),
                    })
                except json.JSONDecodeError:
                    continue

        log.info(f"  Found {len(videos)} results")
        if dry_run:
            for v in videos:
                log.info(f"    -> {v['title']} ({v['duration']}s) -- {v['url']}")

        return videos

    except subprocess.TimeoutExpired:
        log.warning(f"Search timed out for '{target.search_query}'")
        return []


def search_all(targets: list[VideoTarget], dry_run: bool = False) -> dict[str, list[dict]]:
    """Search for all targets. Returns {campaign: [video_info]}."""
    results = {}
    for target in targets:
        if target.url:
            results[target.campaign] = [{
                "url": target.url,
                "title": target.campaign,
                "duration": 0,
                "uploader": "Direct URL",
                "upload_date": "",
                "id": "",
            }]
        else:
            found = search_videos(target, dry_run=dry_run)
            results[target.campaign] = found
    return results

# ─── Phase 2: Download ──────────────────────────────────────────────────────

def download_video(url: str, output_dir: Path, dry_run: bool = False) -> Optional[Path]:
    """Download a single video in Coactive-compliant format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", YT_DLP_FORMAT,
        "--merge-output-format", CONTAINER,
        "--postprocessor-args", f"ffmpeg:-c:v libx264 -c:a aac -r {TARGET_FPS}",
        "--download-archive", str(ARCHIVE_FILE),
        "-o", output_template,
        "--no-overwrites",
        "--restrict-filenames",
        "--write-info-json",
        url,
    ]

    if dry_run:
        log.info(f"  [DRY RUN] Would download: {url}")
        return None

    log.info(f"  Downloading: {url}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            if "has already been recorded" in result.stdout:
                log.info("  Already downloaded (skipping)")
                return None
            log.warning(f"  Download failed: {result.stderr[:300]}")
            return None

        # Find the downloaded file
        for line in result.stdout.split("\n"):
            if "[Merger]" in line or "[download]" in line:
                match = re.search(r'Merging formats into "(.+?)"', line)
                if match:
                    return Path(match.group(1))
                match = re.search(r'Destination: (.+?)$', line)
                if match:
                    return Path(match.group(1))

        # Fallback: newest mp4 in output dir
        mp4s = sorted(output_dir.glob("*.mp4"), key=os.path.getmtime, reverse=True)
        if mp4s:
            return mp4s[0]

        return None

    except subprocess.TimeoutExpired:
        log.warning(f"  Download timed out for {url}")
        return None


def download_all(
    targets: list[VideoTarget],
    search_results: dict[str, list[dict]],
    dry_run: bool = False,
) -> list[tuple[VideoTarget, Path]]:
    """Download all found videos. Returns [(target, filepath)]."""
    downloaded = []

    for target in targets:
        videos = search_results.get(target.campaign, [])
        output_dir = RAW_DIR / target.category

        for video in videos:
            filepath = download_video(video["url"], output_dir, dry_run=dry_run)
            if filepath:
                downloaded.append((target, filepath))

    log.info(f"Downloaded {len(downloaded)} new videos")
    return downloaded

# ─── Phase 3: Validate ──────────────────────────────────────────────────────

def probe_video(filepath: Path) -> Optional[dict]:
    """Use ffprobe to extract video metadata."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=codec_name,codec_type,width,height,r_frame_rate,duration",
        "-show_entries", "format=duration,filename",
        "-of", "json",
        str(filepath),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log.warning(f"ffprobe failed for {filepath}: {result.stderr[:200]}")
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        log.warning(f"ffprobe error for {filepath}: {e}")
        return None


def parse_fps(r_frame_rate: str) -> float:
    """Parse fractional fps like '30/1' or '30000/1001'."""
    try:
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            return round(float(num) / float(den), 2)
        return float(r_frame_rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def validate_video(filepath: Path) -> dict:
    """Validate a video against Coactive's accepted media specs."""
    report = {
        "file": str(filepath),
        "exists": filepath.exists(),
        "compliant": False,
        "issues": [],
        "video_codec": None,
        "audio_codec": None,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration": 0.0,
    }

    if not report["exists"]:
        report["issues"].append("File does not exist")
        return report

    probe = probe_video(filepath)
    if not probe:
        report["issues"].append("Could not probe file")
        return report

    streams = probe.get("streams", [])
    fmt = probe.get("format", {})

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            report["video_codec"] = stream.get("codec_name", "")
            report["width"] = int(stream.get("width", 0))
            report["height"] = int(stream.get("height", 0))
            fps_str = stream.get("r_frame_rate", "0/1")
            report["fps"] = parse_fps(fps_str)

            if report["video_codec"] not in ("h264", "hevc", "h265"):
                report["issues"].append(
                    f"Video codec '{report['video_codec']}' -- expected h264 or hevc"
                )
            if report["height"] > 1080:
                report["issues"].append(f"Resolution {report['height']}p exceeds 1080p max")
            elif report["height"] < 240:
                report["issues"].append(f"Resolution {report['height']}p below 240p min")
            if report["fps"] > 60:
                report["issues"].append(f"FPS {report['fps']} exceeds 60fps max")

        elif codec_type == "audio":
            report["audio_codec"] = stream.get("codec_name", "")
            if report["audio_codec"] not in ("aac", "mp3"):
                report["issues"].append(
                    f"Audio codec '{report['audio_codec']}' -- expected aac or mp3"
                )

    # Duration
    duration = float(fmt.get("duration", 0))
    if duration == 0:
        for s in streams:
            d = s.get("duration")
            if d:
                duration = float(d)
                break
    report["duration"] = duration

    if duration < 5:
        report["issues"].append(f"Duration {duration:.1f}s below 5s minimum")
    elif duration > 18000:
        report["issues"].append(f"Duration {duration:.1f}s exceeds 5h maximum")

    # Container
    if not str(filepath).endswith((".mp4", ".mkv", ".webm", ".flv", ".avi", ".mpg", ".ts")):
        report["issues"].append(f"Unsupported container: {filepath.suffix}")

    report["compliant"] = len(report["issues"]) == 0
    return report


def validate_all(directory: Optional[Path] = None) -> list[dict]:
    """Validate all mp4 files in the raw_videos directory."""
    search_dir = directory or RAW_DIR
    reports = []

    for mp4 in sorted(search_dir.rglob("*.mp4")):
        report = validate_video(mp4)
        status = "PASS" if report["compliant"] else "FAIL"
        icon = "PASS" if report["compliant"] else "FAIL"
        log.info(
            f"  [{icon}] {mp4.name} "
            f"({report['video_codec']}, {report['width']}x{report['height']}, "
            f"{report['fps']}fps, {report['duration']:.0f}s)"
        )
        if report["issues"]:
            for issue in report["issues"]:
                log.info(f"       !  {issue}")
        reports.append(report)

    passed = sum(1 for r in reports if r["compliant"])
    log.info(f"\nValidation: {passed}/{len(reports)} compliant")
    return reports

# ─── Phase 3b: Re-encode Non-Compliant ──────────────────────────────────────

def reencode_video(filepath: Path) -> Optional[Path]:
    """Re-encode a non-compliant video to Coactive-compliant format."""
    output = CONVERTED_DIR / filepath.name
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

    if output.exists():
        log.info(f"  Already converted: {output.name}")
        return output

    cmd = [
        "ffmpeg", "-i", str(filepath),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-r", str(TARGET_FPS),
        "-vf", f"scale='min(1280,iw)':'min({MAX_HEIGHT},ih)':force_original_aspect_ratio=decrease",
        "-movflags", "+faststart",
        "-y",
        str(output),
    ]

    log.info(f"  Re-encoding: {filepath.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.warning(f"  Re-encode failed: {result.stderr[:300]}")
            return None
        log.info(f"  Converted: {output.name}")
        return output
    except subprocess.TimeoutExpired:
        log.warning(f"  Re-encode timed out for {filepath.name}")
        return None

# ─── Phase 4: Frame Extraction ───────────────────────────────────────────────

def extract_frames(filepath: Path, output_dir: Path) -> int:
    """Extract frames from a video at configured interval. Returns frame count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = filepath.stem

    existing = list(output_dir.glob(f"{stem}_*.jpg"))
    if existing:
        log.info(f"  Frames already exist for {filepath.name} ({len(existing)} frames)")
        return len(existing)

    cmd = [
        "ffmpeg", "-i", str(filepath),
        "-vf", f"fps=1/{FRAME_INTERVAL_SEC},scale='max({FRAME_MIN_PX},iw)':'max({FRAME_MIN_PX},ih)'",
        "-q:v", "2",
        "-y",
        str(output_dir / f"{stem}_%04d.jpg"),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.warning(f"  Frame extraction failed for {filepath.name}: {result.stderr[:200]}")
            return 0

        frames = list(output_dir.glob(f"{stem}_*.jpg"))
        log.info(f"  Extracted {len(frames)} frames from {filepath.name}")
        return len(frames)
    except subprocess.TimeoutExpired:
        log.warning(f"  Frame extraction timed out for {filepath.name}")
        return 0


def extract_all_frames():
    """Extract frames from all downloaded videos, preserving subfolder structure."""
    total_frames = 0
    if not RAW_DIR.exists():
        log.warning(f"No raw_videos directory found at {RAW_DIR}")
        return 0

    for subdir in sorted(RAW_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        dst_dir = FRAMES_DIR / subdir.name
        for mp4 in sorted(subdir.glob("*.mp4")):
            count = extract_frames(mp4, dst_dir)
            total_frames += count

    log.info(f"\nTotal frames extracted: {total_frames}")
    return total_frames

# ─── Phase 5: Manifest Generation ───────────────────────────────────────────

def infer_campaign(filepath: Path, targets: list[VideoTarget]) -> tuple[str, str, int]:
    """Try to match a file to a target for metadata. Returns (campaign, product_line, tier)."""
    name_lower = filepath.stem.lower().replace("_", " ").replace("-", " ")
    parent = filepath.parent.name

    for target in targets:
        campaign_words = target.campaign.lower().replace("_", " ").split()
        if any(word in name_lower for word in campaign_words if len(word) > 3):
            return target.campaign, target.product_line, target.tier

    # Fallback: use filename and tier 0
    return filepath.stem, "Unknown", 0


def generate_manifest(
    targets: list[VideoTarget],
    s3_prefix: str = "s3://YOUR-BUCKET/YOUR-PREFIX",
) -> Path:
    """Generate Coactive CSV manifest from all downloaded videos."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = METADATA_DIR / "video_manifest.csv"

    rows = []
    for mp4 in sorted(RAW_DIR.rglob("*.mp4")):
        report = validate_video(mp4)
        campaign, product_line, tier = infer_campaign(mp4, targets)
        category = mp4.parent.name

        # Build S3 path
        relative = mp4.relative_to(RAW_DIR)
        s3_path = f"{s3_prefix}/{relative}"

        # Try to get upload date from yt-dlp .info.json sidecar
        info_json = mp4.with_suffix(".mp4.info.json")
        if not info_json.exists():
            info_json = mp4.with_suffix(".info.json")

        date_published = ""
        if info_json.exists():
            try:
                with open(info_json) as f:
                    info = json.load(f)
                raw_date = info.get("upload_date", "")
                if raw_date and len(raw_date) == 8:
                    date_published = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            except (json.JSONDecodeError, KeyError):
                pass

        rows.append({
            "source_path": s3_path,
            "asset_type": "video",
            "campaign": campaign.replace(" ", "_"),
            "product_line": product_line.replace(" ", "_"),
            "tier": tier,
            "category": category,
            "date_published": date_published,
            "duration_sec": round(report["duration"]),
            "width": report["width"],
            "height": report["height"],
            "video_codec": report["video_codec"] or "",
            "fps": report["fps"],
            "compliant": report["compliant"],
        })

    if not rows:
        log.warning("No videos found -- manifest is empty")
        return manifest_path

    fieldnames = list(rows[0].keys())
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"\nManifest written: {manifest_path} ({len(rows)} entries)")
    return manifest_path

# ─── Init: Create Example Targets ────────────────────────────────────────────

def create_example_targets():
    """Create an example targets.json to get started."""
    targets = {
        "description": "Example video targets -- edit search queries, tiers, and categories for your use case",
        "targets": [
            # ── Tier 1: Primary ──
            {
                "campaign": "Brand_Hero_Spot",
                "search_query": "\"your brand\" official ad 2025",
                "product_line": "Brand",
                "tier": 1,
                "category": "brand_ads",
                "max_results": 3
            },
            {
                "campaign": "Product_Launch",
                "search_query": "\"your product\" launch ad official",
                "product_line": "Product_A",
                "tier": 1,
                "category": "brand_ads",
                "max_results": 2
            },
            # ── Tier 2: Supplemental ──
            {
                "campaign": "Seasonal_Campaign",
                "search_query": "\"your brand\" holiday ad 2025",
                "product_line": "Seasonal",
                "tier": 2,
                "category": "brand_ads",
                "max_results": 2
            },
            # ── Tier 3: Competitive ──
            {
                "campaign": "Competitor_A",
                "search_query": "competitor brand official ad 2025",
                "product_line": "Competitor",
                "tier": 3,
                "category": "competitor_ads",
                "max_results": 2
            },
            # ── Direct URL example ──
            {
                "campaign": "Specific_Video",
                "search_query": "",
                "product_line": "Brand",
                "tier": 1,
                "category": "brand_ads",
                "max_results": 1,
                "url": "https://www.youtube.com/watch?v=REPLACE_ME"
            },
        ],
    }

    with open(TARGETS_FILE, "w") as f:
        json.dump(targets, f, indent=2)
    log.info(f"Created example targets file: {TARGETS_FILE}")
    log.info("Edit targets.json with your search queries, then run the pipeline.")

# ─── Summary Report ─────────────────────────────────────────────────────────

def print_summary(validations: list[dict], manifest_path: Path):
    """Print a final pipeline summary."""
    total = len(validations)
    passed = sum(1 for v in validations if v["compliant"])
    failed = total - passed

    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    print(f"  Videos processed:      {total}")
    print(f"  Coactive-compliant:    {passed}")
    if failed:
        print(f"  Non-compliant:         {failed}")
    print(f"  Manifest:              {manifest_path}")
    print(f"  Archive:               {ARCHIVE_FILE}")
    print("=" * 60 + "\n")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YouTube Video Scraper Pipeline -- Coactive-Compliant"
    )
    parser.add_argument("--tier", type=int,
                        help="Only process targets in this tier (e.g., 1, 2, 3)")
    parser.add_argument("--search-only", action="store_true",
                        help="Search and list URLs without downloading")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate existing downloads only")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Regenerate manifest CSV only")
    parser.add_argument("--extract-frames", action="store_true",
                        help="Extract jpg frames from downloaded videos")
    parser.add_argument("--reencode", action="store_true",
                        help="Re-encode non-compliant videos to H.264/AAC")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    parser.add_argument("--init", action="store_true",
                        help="Create an example targets.json")
    parser.add_argument("--s3-prefix", default="s3://YOUR-BUCKET/YOUR-PREFIX",
                        help="S3 prefix for manifest source_path column")
    parser.add_argument("--frame-interval", type=int, default=FRAME_INTERVAL_SEC,
                        help=f"Seconds between extracted frames (default: {FRAME_INTERVAL_SEC})")
    parser.add_argument("--max-height", type=int, default=MAX_HEIGHT,
                        help=f"Max video height in pixels (default: {MAX_HEIGHT})")

    args = parser.parse_args()

    # Apply CLI overrides to globals
    global FRAME_INTERVAL_SEC, MAX_HEIGHT, YT_DLP_FORMAT
    FRAME_INTERVAL_SEC = args.frame_interval
    MAX_HEIGHT = args.max_height
    YT_DLP_FORMAT = (
        f"bestvideo[height<={MAX_HEIGHT}][vcodec^={VIDEO_CODEC}]"
        f"+bestaudio[acodec^={AUDIO_CODEC}]"
        f"/best[height<={MAX_HEIGHT}]"
    )

    print("\nVideo Scraper Pipeline (Coactive-Compliant)")
    print(f"  Base directory: {BASE_DIR}\n")

    # ── Init ──
    if args.init:
        create_example_targets()
        return

    # ── Validate only ──
    if args.validate_only:
        log.info("Phase 3: Validating existing downloads...")
        reports = validate_all()
        if args.reencode:
            for report in reports:
                if not report["compliant"]:
                    reencode_video(Path(report["file"]))
        return

    # ── Manifest only ──
    if args.manifest_only:
        targets = load_targets(args.tier)
        log.info("Phase 5: Generating manifest...")
        manifest = generate_manifest(targets, s3_prefix=args.s3_prefix)
        log.info(f"Done: {manifest}")
        return

    # ── Frame extraction only ──
    if args.extract_frames:
        log.info("Phase 4: Extracting frames...")
        extract_all_frames()
        return

    # ── Full pipeline ──
    targets = load_targets(args.tier)

    # Phase 1: Search
    log.info("Phase 1: Searching for videos...")
    search_results = search_all(targets, dry_run=args.dry_run)

    total_found = sum(len(v) for v in search_results.values())
    log.info(f"Search complete: {total_found} videos found across {len(search_results)} campaigns\n")

    if args.search_only or args.dry_run:
        for campaign, videos in search_results.items():
            print(f"\n  {campaign}:")
            for v in videos:
                print(f"    {v['title']}")
                print(f"    {v['url']}  ({v['duration']}s)\n")
        if args.dry_run:
            log.info("[DRY RUN] No downloads performed.")
        return

    # Phase 2: Download
    log.info("Phase 2: Downloading videos...")
    download_all(targets, search_results, dry_run=args.dry_run)

    # Phase 3: Validate
    log.info("\nPhase 3: Validating downloads...")
    reports = validate_all()

    # Re-encode if requested
    if args.reencode:
        for report in reports:
            if not report["compliant"]:
                reencode_video(Path(report["file"]))

    # Phase 5: Generate manifest
    log.info("\nPhase 5: Generating manifest...")
    manifest = generate_manifest(targets, s3_prefix=args.s3_prefix)

    # Summary
    print_summary(reports, manifest)


if __name__ == "__main__":
    main()
