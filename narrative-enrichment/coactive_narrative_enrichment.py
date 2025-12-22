#!/usr/bin/env python3
"""
Coactive Video Narrative Metadata Enrichment
=============================================
Runs Coactive Video Narrative APIs on all videos in a dataset and updates metadata:
- Video Summary (with custom intent)
- Video Description
- Video Genre (classification)
- Video Mood (classification)
- Video Subject (classification)
- Video Format (classification)
- Video Segments (scene/chapter detection with timestamps)
- Video Entities (people/celebrity extraction)
- Caption Keyframes (async)

API Flow for Genre/Mood/Subject/Format:
1. POST to /api/v0/video-narrative-metadata/metadata to define possible values
2. POST to /api/v0/video-narrative-metadata/datasets/{id}/videos/{id}/{type} 
   with {"values": [...]} to classify the video into defined categories

IMPORTANT: For genre, mood, subject, and format classification to work:
1. First run with --setup-metadata to define the classification categories
2. The script will then POST with the defined values to classify videos
3. If video content doesn't match any defined values, classification returns empty

Usage:
    # First time setup - create metadata values
    python3 coactive_narrative_enrichment.py -d <DATASET_ID> -t <TOKEN> --setup-metadata
    
    # Run enrichment
    python3 coactive_narrative_enrichment.py -d <DATASET_ID> -t <TOKEN>
    
    # Run with segment detection (scene/chapter timestamps)
    python3 coactive_narrative_enrichment.py -d <DATASET_ID> -t <TOKEN> --segments
    
    # Run with entity extraction (people/celebrities)
    python3 coactive_narrative_enrichment.py -d <DATASET_ID> -t <TOKEN> --entities

Examples:
    # Setup metadata values and run enrichment
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --setup-metadata
    
    # Run enrichment only (metadata values already exist)
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN
    
    # Run with segment/chapter detection
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --segments
    
    # Run with entity extraction
    python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --entities
    
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
# Includes both entertainment and news content categories
DEFAULT_METADATA_VALUES = {
    "genre": [
        # Entertainment
        {"name": "Awards Show", "description": "Content featuring award ceremonies, acceptance speeches, and recognition events", "examples": ["Grammy Awards ceremony", "Artist receiving award on stage", "Acceptance speech"]},
        {"name": "Documentary", "description": "Non-fiction content documenting real events and people", "examples": ["Behind the scenes footage", "Event documentation"]},
        {"name": "Reality TV", "description": "Unscripted reality television content", "examples": ["Red carpet coverage", "Live event footage"]},
        {"name": "Talk Show", "description": "Interview and conversation format programming", "examples": ["Celebrity interview", "Artist Q&A"]},
        {"name": "Music Video", "description": "Music video or performance content", "examples": ["Artist performance", "Music clip"]},
        # News
        {"name": "News Program", "description": "Television news program genre", "examples": ["CBS Evening News", "NBC Nightly News", "Breaking news coverage"]},
        {"name": "Current Affairs", "description": "Current affairs programming covering news and public interest topics", "examples": ["60 Minutes", "Dateline", "News magazine"]},
    ],
    "mood": [
        # Entertainment
        {"name": "Celebratory", "description": "Joyful and celebratory atmosphere", "examples": ["Award wins", "Celebration moments"]},
        {"name": "Emotional", "description": "Touching and emotional moments", "examples": ["Heartfelt speeches", "Tearful acceptance"]},
        {"name": "Exciting", "description": "High energy and exciting content", "examples": ["Performance highlights", "Big reveals"]},
        {"name": "Inspiring", "description": "Uplifting and motivational content", "examples": ["Inspirational speeches", "Success stories"]},
        {"name": "Nostalgic", "description": "Content evoking memories and nostalgia", "examples": ["Retrospective moments", "Historical clips"]},
        # News
        {"name": "Serious", "description": "Serious and professional tone for important topics", "examples": ["Hard news", "Breaking news coverage"]},
    ],
    "subject": [
        # Entertainment
        {"name": "Music", "description": "Content about music and musicians", "examples": ["Songs", "Albums", "Musical performances"]},
        {"name": "Celebrity", "description": "Celebrity-focused content", "examples": ["Famous artists", "Star appearances"]},
        {"name": "Awards", "description": "Award-related content", "examples": ["Grammy Awards", "Music awards"]},
        {"name": "Fashion", "description": "Fashion and style content", "examples": ["Red carpet fashion", "Designer outfits"]},
        {"name": "Entertainment Industry", "description": "Entertainment business content", "examples": ["Industry news", "Record labels"]},
        # News
        {"name": "Current Events", "description": "Coverage of current news and events", "examples": ["Daily news", "Breaking stories", "World events"]},
        {"name": "Politics", "description": "Political news and government coverage", "examples": ["Government policy", "Elections", "Congress"]},
    ],
    "format": [
        # Entertainment
        {"name": "Speech", "description": "Acceptance speeches and presentations", "examples": ["Award acceptance", "Thank you speech"]},
        {"name": "Performance", "description": "Live musical performances", "examples": ["Stage performance", "Live singing"]},
        {"name": "Interview", "description": "Interview format content", "examples": ["Red carpet interview", "Backstage Q&A"]},
        {"name": "Highlight Reel", "description": "Compilation and highlight content", "examples": ["Best moments", "Montage"]},
        {"name": "Behind The Scenes", "description": "Behind the scenes footage", "examples": ["Backstage footage", "Preparation clips"]},
        # News
        {"name": "News Broadcast", "description": "Television news broadcast format with anchors and field reports", "examples": ["Evening news show", "News desk", "Field report"]},
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


def get_metadata_values_for_classification(token, metadata_type):
    """
    Get defined metadata values for use in classification.
    Returns a list of value names that can be passed to classification endpoints.
    """
    resp = api_get(token, f'/api/v0/video-narrative-metadata/metadata?metadata_type={metadata_type}')
    if resp and 'items' in resp:
        return [item.get('name') for item in resp['items'] if item.get('name')]
    return []


def get_narrative_metadata(token, dataset_id, video_id, summary_intent=None, classification_values=None):
    """
    Get all narrative metadata for a video.
    
    Args:
        token: JWT access token
        dataset_id: Dataset ID
        video_id: Video ID
        summary_intent: Custom intent for summary generation
        classification_values: Dict of {type: [values]} for genre/mood/subject/format.
                              If None, will fetch from API.
    """
    metadata = {}
    fields = []
    
    # Default summary intent
    if not summary_intent:
        summary_intent = "Provide a comprehensive summary of the video content, plot, key scenes, and themes"
    
    # Fetch classification values if not provided
    if classification_values is None:
        classification_values = {}
        for meta_type in ['genre', 'mood', 'subject', 'format']:
            values = get_metadata_values_for_classification(token, meta_type)
            if values:
                classification_values[meta_type] = values
    
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
    
    # Genre - POST with values for classification
    print("    🎬 Genre...", end=" ", flush=True)
    genre_values = classification_values.get('genre', [])
    if genre_values:
        resp = api_post(token, 
            f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/genre',
            {"values": genre_values}
        )
        if resp and 'genres' in resp and resp['genres']:
            metadata['video_narrative_genre'] = ', '.join(resp['genres'])
            fields.append('genre')
            print(f"✅ {metadata['video_narrative_genre']}")
        else:
            print("❌ (no match found)")
    else:
        print("❌ (run --setup-metadata first)")
    
    # Mood - POST with values for classification
    print("    🎭 Mood...", end=" ", flush=True)
    mood_values = classification_values.get('mood', [])
    if mood_values:
        resp = api_post(token, 
            f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/mood',
            {"values": mood_values}
        )
        if resp and 'moods' in resp and resp['moods']:
            metadata['video_narrative_mood'] = ', '.join(resp['moods'])
            fields.append('mood')
            print(f"✅ {metadata['video_narrative_mood']}")
        else:
            print("❌ (no match found)")
    else:
        print("❌ (run --setup-metadata first)")
    
    # Subject - POST with values for classification
    print("    📚 Subject...", end=" ", flush=True)
    subject_values = classification_values.get('subject', [])
    if subject_values:
        resp = api_post(token, 
            f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/subject',
            {"values": subject_values}
        )
        if resp and 'subjects' in resp and resp['subjects']:
            metadata['video_narrative_subject'] = ', '.join(resp['subjects'])
            fields.append('subject')
            print(f"✅ {metadata['video_narrative_subject']}")
        else:
            print("❌ (no match found)")
    else:
        print("❌ (run --setup-metadata first)")
    
    # Format - POST with values for classification
    print("    🎞️ Format...", end=" ", flush=True)
    format_values = classification_values.get('format', [])
    if format_values:
        resp = api_post(token, 
            f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/format',
            {"values": format_values}
        )
        if resp and 'formats' in resp and resp['formats']:
            metadata['video_narrative_format'] = ', '.join(resp['formats'])
            fields.append('format')
            print(f"✅ {metadata['video_narrative_format']}")
        else:
            print("❌ (no match found)")
    else:
        print("❌ (run --setup-metadata first)")
    
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


def get_video_entities(token, dataset_id, video_id):
    """
    Extract people, celebrities, artists, and notable entities from a video.
    
    Returns a list of entities, each containing:
    - name: Entity name
    - role: List of roles (speaker, performer, mentioned, audience, etc.)
    - context: Description of their appearance/mention
    """
    print("    🧑 Detecting entities...", end=" ", flush=True)
    
    entity_intent = (
        "Identify all people, celebrities, artists, and notable entities that appear "
        "or are mentioned in this video. For each person, note their role (speaker, "
        "performer, mentioned, audience) and any context about them."
    )
    
    resp = api_post(token, 
        f'/api/v0/video-summarization/datasets/{dataset_id}/videos/{video_id}/summarize',
        {"summary_intent": entity_intent}
    )
    
    if resp:
        # Check if response is already a list of entities
        if isinstance(resp, list):
            print(f"✅ Found {len(resp)} entities")
            return resp
        # Check if summary contains entity data
        elif 'summary' in resp:
            summary = resp['summary']
            if isinstance(summary, list):
                print(f"✅ Found {len(summary)} entities")
                return summary
            elif isinstance(summary, str):
                # Try to parse as JSON if it's a string
                try:
                    entities = json.loads(summary)
                    if isinstance(entities, list):
                        print(f"✅ Found {len(entities)} entities")
                        return entities
                except:
                    pass
                # Return as single entity if it's just text
                print("✅ (text description)")
                return [{"description": summary}]
    
    print("❌")
    return []


def format_entities_for_metadata(entities):
    """
    Format entities list for storage in metadata.
    Creates JSON string and a list of people names.
    """
    if not entities:
        return None, None
    
    # Extract just the names for quick reference
    people_names = []
    for entity in entities:
        name = entity.get('name', '')
        if name and not any(skip in name.lower() for skip in ['audience', 'unspecified', 'unknown']):
            # Clean up name
            clean_name = name.split('(')[0].strip()  # Remove parenthetical notes
            if clean_name and len(clean_name) > 1:
                people_names.append(clean_name)
    
    # Deduplicate while preserving order
    seen = set()
    unique_names = []
    for name in people_names:
        if name.lower() not in seen:
            seen.add(name.lower())
            unique_names.append(name)
    
    people_list = ", ".join(unique_names[:20])  # Limit to top 20
    entities_json = json.dumps(entities)
    
    return entities_json, people_list


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
    parser.add_argument('--entities', '-e', action='store_true',
                        help='Extract people, celebrities, and notable entities')
    parser.add_argument('--entities-only', action='store_true',
                        help='Only extract entities, skip other metadata')
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
    if args.entities or args.entities_only:
        print("Entity Extraction: Yes")
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
        
        # Get entity/people extraction if requested
        if args.entities or args.entities_only:
            entities = get_video_entities(token, args.dataset_id, video_id)
            if entities:
                entities_json, people_list = format_entities_for_metadata(entities)
                if entities_json:
                    metadata['video_entities'] = entities_json
                    metadata['video_people'] = people_list
                    metadata['video_entity_count'] = str(len(entities))
                    fields.append('entities')
        
        # Get other narrative metadata (unless segments-only or entities-only mode)
        if not args.segments_only and not args.entities_only:
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
