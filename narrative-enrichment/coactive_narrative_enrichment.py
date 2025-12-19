#!/usr/bin/env python3
"""
Coactive Video Narrative Metadata Enrichment
=============================================
Runs Coactive Video Narrative APIs on all videos in a dataset and updates metadata:
- Video Summary (with custom intent)
- Video Description
- Video Genre
- Video Mood
- Video Subject
- Video Format
- Video Segments (scene/chapter detection with timestamps)
- Caption Keyframes (async)

IMPORTANT: For genre, mood, subject, and format to work, you must first define
the possible values using --setup-metadata. This creates the classification 
categories that the API uses to categorize videos.

Usage:
    # First time setup - create metadata values
    python3 coactive_narrative_enrichment.py -d <DATASET_ID> -t <TOKEN> --setup-metadata
    
    # Run enrichment
    python3 coactive_narrative_enrichment.py -d <DATASET_ID> -t <TOKEN>
    
    # Run with segment detection (scene/chapter timestamps)
    python3 coactive_narrative_enrichment.py -d <DATASET_ID> -t <TOKEN> --segments

Examples:
    # Setup metadata values and run enrichment
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --setup-metadata
    
    # Run enrichment only (metadata values already exist)
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN
    
    # Run with segment/chapter detection
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --segments
    
    # Run with custom summary intent
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --intent "Summarize key action scenes"
    
    # Only process specific video
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --video-id VIDEO_UUID

Author: Madan
Date: December 2025
"""

import json
import subprocess
import time
import argparse
from datetime import datetime, timezone


# API Configuration
API_BASE_URL = "https://api.coactive.ai"
APP_BASE_URL = "https://app.coactive.ai"


# Default metadata values for classification
# These define what categories the API can classify videos into
DEFAULT_METADATA_VALUES = {
    "genre": [
        {"name": "Awards Show", "description": "Content featuring award ceremonies, acceptance speeches, and recognition events", "examples": ["Grammy Awards ceremony", "Artist receiving award on stage", "Acceptance speech"]},
        {"name": "Documentary", "description": "Non-fiction content documenting real events and people", "examples": ["Behind the scenes footage", "Event documentation"]},
        {"name": "Reality TV", "description": "Unscripted reality television content", "examples": ["Red carpet coverage", "Live event footage"]},
        {"name": "Talk Show", "description": "Interview and conversation format programming", "examples": ["Celebrity interview", "Artist Q&A"]},
        {"name": "Music Video", "description": "Music video or performance content", "examples": ["Artist performance", "Music clip"]},
    ],
    "mood": [
        {"name": "Celebratory", "description": "Joyful and celebratory atmosphere", "examples": ["Award wins", "Celebration moments"]},
        {"name": "Emotional", "description": "Touching and emotional moments", "examples": ["Heartfelt speeches", "Tearful acceptance"]},
        {"name": "Exciting", "description": "High energy and exciting content", "examples": ["Performance highlights", "Big reveals"]},
        {"name": "Inspiring", "description": "Uplifting and motivational content", "examples": ["Inspirational speeches", "Success stories"]},
        {"name": "Nostalgic", "description": "Content evoking memories and nostalgia", "examples": ["Retrospective moments", "Historical clips"]},
    ],
    "subject": [
        {"name": "Music", "description": "Content about music and musicians", "examples": ["Songs", "Albums", "Musical performances"]},
        {"name": "Celebrity", "description": "Celebrity-focused content", "examples": ["Famous artists", "Star appearances"]},
        {"name": "Awards", "description": "Award-related content", "examples": ["Grammy Awards", "Music awards"]},
        {"name": "Fashion", "description": "Fashion and style content", "examples": ["Red carpet fashion", "Designer outfits"]},
        {"name": "Entertainment Industry", "description": "Entertainment business content", "examples": ["Industry news", "Record labels"]},
    ],
    "format": [
        {"name": "Speech", "description": "Acceptance speeches and presentations", "examples": ["Award acceptance", "Thank you speech"]},
        {"name": "Performance", "description": "Live musical performances", "examples": ["Stage performance", "Live singing"]},
        {"name": "Interview", "description": "Interview format content", "examples": ["Red carpet interview", "Backstage Q&A"]},
        {"name": "Highlight Reel", "description": "Compilation and highlight content", "examples": ["Best moments", "Montage"]},
        {"name": "Behind The Scenes", "description": "Behind the scenes footage", "examples": ["Backstage footage", "Preparation clips"]},
    ],
}


def get_token(refresh_token):
    """Get JWT access token from refresh token."""
    cmd = [
        'curl', '-s', '-X', 'POST',
        f'{API_BASE_URL}/api/v0/login',
        '-H', f'Authorization: Bearer {refresh_token}',
        '-H', 'Content-Type: application/json',
        '-d', '{"grant_type": "refresh_token"}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            return data.get('access_token')
        except json.JSONDecodeError:
            print(f"❌ Failed to parse token response: {result.stdout}")
    return None


def api_post(token, endpoint, body=None):
    """Make POST request to Coactive API."""
    cmd = [
        'curl', '-s', '-X', 'POST',
        f'{API_BASE_URL}{endpoint}',
        '-H', f'Authorization: Bearer {token}',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps(body or {})
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout:
        try:
            return json.loads(result.stdout)
        except:
            return {"raw": result.stdout}
    return None


def api_get(token, endpoint):
    """Make GET request to Coactive API."""
    cmd = [
        'curl', '-s', '-X', 'GET',
        f'{API_BASE_URL}{endpoint}',
        '-H', f'Authorization: Bearer {token}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout:
        try:
            return json.loads(result.stdout)
        except:
            return None
    return None


def app_api_get(token, endpoint):
    """Make GET request to Coactive App API."""
    cmd = [
        'curl', '-s', '-X', 'GET',
        f'{APP_BASE_URL}{endpoint}',
        '-H', f'Authorization: Bearer {token}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout:
        try:
            return json.loads(result.stdout)
        except:
            return None
    return None


def get_existing_metadata_values(token, dataset_id, metadata_type):
    """Get existing metadata values for a type."""
    resp = api_get(token, f'/api/v0/video-narrative-metadata/metadata?dataset_id={dataset_id}&metadata_type={metadata_type}')
    if resp and 'items' in resp:
        return resp['items']
    return []


def create_metadata_value(token, dataset_id, metadata_type, name, description, examples):
    """Create a metadata value for classification."""
    payload = {
        "dataset_id": dataset_id,
        "metadata_type": metadata_type,
        "name": name,
        "description": description,
        "examples": examples
    }
    resp = api_post(token, '/api/v0/video-narrative-metadata/metadata', payload)
    if resp and ('item' in resp or 'id' in resp):
        return True, None
    return False, resp.get('detail', 'Unknown error') if resp else 'No response'


def setup_metadata_values(token, dataset_id, custom_values=None):
    """
    Setup metadata values for genre, mood, subject, and format.
    These values define what categories the API can classify videos into.
    
    Args:
        token: JWT access token
        dataset_id: Dataset ID
        custom_values: Optional dict of custom values (same structure as DEFAULT_METADATA_VALUES)
    
    Returns:
        dict with counts of created/existing/failed values per type
    """
    values = custom_values or DEFAULT_METADATA_VALUES
    results = {}
    
    print("\n📋 Setting up metadata classification values...")
    print("   (This defines what categories the API can classify videos into)\n")
    
    for metadata_type, items in values.items():
        print(f"   {metadata_type.upper()}:")
        
        # Check existing values
        existing = get_existing_metadata_values(token, dataset_id, metadata_type)
        existing_names = {item.get('name', '').lower() for item in existing}
        
        created = 0
        skipped = 0
        failed = 0
        
        for item in items:
            name = item['name']
            
            # Skip if already exists
            if name.lower() in existing_names:
                print(f"      ⏭️  {name} (already exists)")
                skipped += 1
                continue
            
            # Create new value
            success, error = create_metadata_value(
                token, dataset_id, metadata_type,
                name, item['description'], item['examples']
            )
            
            if success:
                print(f"      ✅ {name}")
                created += 1
            else:
                # Some values may be rejected if they don't match the type's scope
                print(f"      ⚠️  {name}: {str(error)[:50]}")
                failed += 1
        
        results[metadata_type] = {'created': created, 'skipped': skipped, 'failed': failed}
    
    print("\n   Summary:")
    for mt, counts in results.items():
        print(f"      {mt}: {counts['created']} created, {counts['skipped']} existing, {counts['failed']} failed")
    
    return results


def get_dataset_info(token, dataset_id):
    """Get dataset information."""
    return app_api_get(token, f'/api/v1/datasets/{dataset_id}')


def get_dataset_videos(token, dataset_id, limit=100):
    """Get all videos from dataset."""
    data = app_api_get(token, f'/api/v1/datasets/{dataset_id}/videos?limit={limit}')
    if data and 'data' in data:
        return data['data']
    return []


def get_narrative_metadata(token, dataset_id, video_id, summary_intent=None):
    """Get all narrative metadata for a video."""
    metadata = {}
    fields = []
    
    # Default summary intent
    if not summary_intent:
        summary_intent = "Provide a comprehensive summary of the video content, plot, key scenes, and themes"
    
    # Summary
    print("    📝 Summary...", end=" ", flush=True)
    resp = api_post(token, 
        f'/api/v0/video-summarization/datasets/{dataset_id}/videos/{video_id}/summarize',
        {"summary_intent": summary_intent}
    )
    if resp and 'summary' in resp:
        metadata['video_narrative_summary'] = resp['summary']
        fields.append('summary')
        print("✅")
    else:
        print("❌")
    
    # Description
    print("    📄 Description...", end=" ", flush=True)
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/description',
        {}
    )
    if resp and 'description' in resp:
        metadata['video_narrative_description'] = resp['description']
        fields.append('desc')
        print("✅")
    else:
        print("❌")
    
    # Genre
    print("    🎬 Genre...", end=" ", flush=True)
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/genre',
        {}
    )
    if resp and 'genres' in resp and resp['genres']:
        metadata['video_narrative_genre'] = ', '.join(resp['genres'])
        fields.append('genre')
        print(f"✅ {metadata['video_narrative_genre']}")
    else:
        print("❌ (run --setup-metadata first)")
    
    # Mood
    print("    🎭 Mood...", end=" ", flush=True)
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/mood',
        {}
    )
    if resp and 'moods' in resp and resp['moods']:
        metadata['video_narrative_mood'] = ', '.join(resp['moods'])
        fields.append('mood')
        print(f"✅ {metadata['video_narrative_mood']}")
    else:
        print("❌")
    
    # Subject
    print("    📚 Subject...", end=" ", flush=True)
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/subject',
        {}
    )
    if resp and 'subjects' in resp and resp['subjects']:
        metadata['video_narrative_subject'] = ', '.join(resp['subjects'])
        fields.append('subject')
        print(f"✅ {metadata['video_narrative_subject']}")
    else:
        print("❌")
    
    # Format
    print("    🎞️ Format...", end=" ", flush=True)
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/format',
        {}
    )
    if resp and 'formats' in resp and resp['formats']:
        metadata['video_narrative_format'] = ', '.join(resp['formats'])
        fields.append('format')
        print(f"✅ {metadata['video_narrative_format']}")
    else:
        print("❌")
    
    # Caption Keyframes (async)
    print("    🖼️ Keyframes...", end=" ", flush=True)
    resp = api_post(token, 
        f'/api/v0/video-summarization/datasets/{dataset_id}/videos/{video_id}/caption-keyframes',
        {}
    )
    if resp and ('message' in resp or 'keyframe_captions' in resp):
        metadata['video_narrative_keyframes_requested'] = 'true'
        fields.append('keyframes')
        print("✅ (triggered)")
    else:
        print("❌")
    
    # Add metadata source and timestamp
    if metadata:
        metadata['video_narrative_metadata_source'] = 'Coactive Video Narrative API'
        metadata['video_narrative_metadata_generated_at'] = datetime.now(timezone.utc).isoformat()
    
    return metadata, fields


def get_video_segments(token, dataset_id, video_id):
    """
    Detect scenes/chapters/segments in a video with timestamps.
    
    Returns a list of segments, each containing:
    - segment_start: Start time in seconds
    - segment_end: End time in seconds
    - description: What happens in this segment
    - transition_or_topic_change: Description of the transition
    """
    print("    🎬 Detecting segments...", end=" ", flush=True)
    
    segment_intent = (
        "Identify distinct scenes, segments, or chapters in this video. "
        "For each segment, describe what happens and note any transitions or topic changes."
    )
    
    resp = api_post(token, 
        f'/api/v0/video-summarization/datasets/{dataset_id}/videos/{video_id}/summarize',
        {"summary_intent": segment_intent}
    )
    
    # The API returns segments as a list when given this intent
    if resp:
        # Check if response is already a list of segments
        if isinstance(resp, list):
            print(f"✅ Found {len(resp)} segments")
            return resp
        # Check if summary contains segment data
        elif 'summary' in resp:
            summary = resp['summary']
            if isinstance(summary, list):
                print(f"✅ Found {len(summary)} segments")
                return summary
            elif isinstance(summary, str):
                # Try to parse as JSON if it's a string
                try:
                    segments = json.loads(summary)
                    if isinstance(segments, list):
                        print(f"✅ Found {len(segments)} segments")
                        return segments
                except:
                    pass
                # Return as single segment if it's just text
                print("✅ (text summary)")
                return [{"description": summary}]
    
    print("❌")
    return []


def format_segments_for_metadata(segments):
    """
    Format segments list for storage in metadata.
    Creates both a JSON string and a human-readable summary.
    """
    if not segments:
        return None, None
    
    # Create human-readable chapter list
    chapters = []
    for i, seg in enumerate(segments, 1):
        start = seg.get('segment_start', 0)
        end = seg.get('segment_end', 0)
        desc = seg.get('description', '')[:100]
        
        # Format timestamp as MM:SS
        start_str = f"{int(start//60)}:{int(start%60):02d}"
        end_str = f"{int(end//60)}:{int(end%60):02d}"
        
        chapters.append(f"[{start_str}-{end_str}] {desc}")
    
    chapters_text = "\n".join(chapters)
    segments_json = json.dumps(segments)
    
    return segments_json, chapters_text


def update_video_metadata(token, dataset_id, video_id, metadata):
    """Update video with narrative metadata."""
    payload = {
        "dataset_id": dataset_id,
        "update_assets": [{
            "asset_type": "video",
            "asset_id": video_id,
            "metadata": metadata
        }],
        "update_type": "upsert"
    }
    
    response = api_post(token, '/api/v1/ingestion/metadata', payload)
    
    if response:
        if response.get('status') == 'pending' or 'accepted' in str(response).lower():
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Run Coactive Narrative Metadata APIs on videos in a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time: setup metadata values then run enrichment
  python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --setup-metadata

  # Run enrichment only (metadata values already configured)
  python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN

  # Run with segment/chapter detection
  python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --segments

  # Process a single video
  python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --video-id VIDEO_ID

  # Custom summary intent
  python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --intent "Focus on action scenes"

Note: Genre, Mood, Subject, and Format require metadata values to be set up first.
      Use --setup-metadata on first run to create the classification categories.
        """
    )
    
    parser.add_argument('--dataset-id', '-d', required=True,
                        help='Coactive Dataset ID (UUID)')
    parser.add_argument('--token', '-t', required=True,
                        help='Coactive Refresh Token (Personal Token)')
    parser.add_argument('--setup-metadata', action='store_true',
                        help='Setup metadata values (genre, mood, subject, format) before enrichment')
    parser.add_argument('--setup-only', action='store_true',
                        help='Only setup metadata values, do not run enrichment')
    parser.add_argument('--segments', '-s', action='store_true',
                        help='Detect video segments/chapters with timestamps')
    parser.add_argument('--segments-only', action='store_true',
                        help='Only detect segments, skip other metadata')
    parser.add_argument('--video-id', '-v',
                        help='Process only this specific video ID')
    parser.add_argument('--intent', '-i',
                        help='Custom summary intent (default: comprehensive summary)')
    parser.add_argument('--limit', '-l', type=int, default=100,
                        help='Max number of videos to process (default: 100)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between videos in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎬 COACTIVE NARRATIVE METADATA ENRICHMENT")
    print("=" * 80)
    print(f"Dataset: {args.dataset_id}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.intent:
        print(f"Custom Intent: {args.intent}")
    if args.setup_metadata or args.setup_only:
        print("Setup Metadata: Yes")
    if args.segments or args.segments_only:
        print("Segment Detection: Yes")
    print("=" * 80)
    
    # Authenticate
    print("\n🔐 Authenticating...")
    token = get_token(args.token)
    if not token:
        print("❌ Authentication failed. Check your token.")
        return
    print("✅ Authenticated")
    
    # Get dataset info
    print("\n📊 Dataset Info:")
    ds_info = get_dataset_info(token, args.dataset_id)
    if ds_info:
        print(f"   Name: {ds_info.get('name', 'Unknown')}")
        print(f"   Status: {ds_info.get('status', 'Unknown')}")
        print(f"   Videos: {ds_info.get('videoCount', 0)}")
    
    # Setup metadata values if requested
    if args.setup_metadata or args.setup_only:
        setup_metadata_values(token, args.dataset_id)
        
        if args.setup_only:
            print("\n✅ Metadata setup complete. Run without --setup-only to enrich videos.")
            return
    
    # Get videos
    if args.video_id:
        # Process single video
        videos = [{'coactiveVideoId': args.video_id, 'path': args.video_id, 'metadata': {}}]
        print(f"\n📹 Processing single video: {args.video_id}")
    else:
        print(f"\n📹 Fetching videos...")
        videos = get_dataset_videos(token, args.dataset_id, args.limit)
        print(f"   Found {len(videos)} videos")
    
    if not videos:
        print("❌ No videos found")
        return
    
    # Process each video
    results = []
    
    for idx, video in enumerate(videos, 1):
        video_id = video.get('coactiveVideoId')
        path = video.get('path', '')
        filename = path.split('/')[-1] if path else video_id
        
        # Get title from existing metadata
        existing_meta = video.get('metadata', {})
        title = (existing_meta.get('original_title') or 
                 existing_meta.get('imdb_title') or 
                 existing_meta.get('extracted_title') or 
                 filename)[:50]
        
        print(f"\n[{idx}/{len(videos)}] 🎬 {title}")
        print(f"    ID: {video_id}")
        
        metadata = {}
        fields = []
        
        # Get segment/chapter detection if requested
        if args.segments or args.segments_only:
            segments = get_video_segments(token, args.dataset_id, video_id)
            if segments:
                segments_json, chapters_text = format_segments_for_metadata(segments)
                if segments_json:
                    metadata['video_segments'] = segments_json
                    metadata['video_chapters'] = chapters_text
                    metadata['video_segment_count'] = str(len(segments))
                    fields.append('segments')
        
        # Get other narrative metadata (unless segments-only mode)
        if not args.segments_only:
            narrative_metadata, narrative_fields = get_narrative_metadata(token, args.dataset_id, video_id, args.intent)
            metadata.update(narrative_metadata)
            fields.extend(narrative_fields)
        
        if len(fields) > 0:
            print(f"    💾 Saving {len(fields)} fields...")
            if update_video_metadata(token, args.dataset_id, video_id, metadata):
                print("       ✅ Saved!")
                results.append({'video': title, 'status': 'success', 'fields': fields})
            else:
                print("       ❌ Save failed")
                results.append({'video': title, 'status': 'save_failed', 'fields': fields})
        else:
            print("    ❌ No metadata retrieved")
            results.append({'video': title, 'status': 'no_metadata', 'fields': []})
        
        # Rate limiting
        if idx < len(videos):
            time.sleep(args.delay)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 ENRICHMENT SUMMARY")
    print("=" * 80)
    
    full_success = len([r for r in results if len(r.get('fields', [])) >= 5])
    partial = len([r for r in results if 0 < len(r.get('fields', [])) < 5])
    failed = len([r for r in results if len(r.get('fields', [])) == 0])
    
    print(f"✅ Full (5+ fields): {full_success}")
    print(f"⚠️  Partial: {partial}")
    print(f"❌ Failed: {failed}")
    
    # Collect all field types
    all_fields = set()
    for r in results:
        all_fields.update(r.get('fields', []))
    
    if all_fields:
        print(f"\n📋 Fields captured: {', '.join(sorted(all_fields))}")
    
    if 'genre' not in all_fields or 'mood' not in all_fields:
        print("\n💡 Tip: If genre/mood/subject/format are missing, run with --setup-metadata")
    
    print(f"\n🔗 View in Coactive: https://app.coactive.ai/datasets/{args.dataset_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()
