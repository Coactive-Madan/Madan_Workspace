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
- Caption Keyframes (async)

Usage:
    python3 coactive_narrative_enrichment.py --dataset-id <DATASET_ID> --token <REFRESH_TOKEN>
    python3 coactive_narrative_enrichment.py -d <DATASET_ID> -t <REFRESH_TOKEN>

Examples:
    # Run on a specific dataset
    python3 coactive_narrative_enrichment.py -d d2fae475-4ebd-46ac-8bad-af2c5a784b43 -t YOUR_TOKEN
    
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


def get_dataset_info(token, dataset_id):
    """Get dataset information."""
    return api_get(token, f'/api/v1/datasets/{dataset_id}')


def get_dataset_videos(token, dataset_id, limit=100):
    """Get all videos from dataset."""
    data = api_get(token, f'/api/v1/datasets/{dataset_id}/videos?limit={limit}')
    if data and 'data' in data:
        return data['data']
    return []


def get_narrative_metadata(token, dataset_id, video_id, summary_intent=None):
    """Get all narrative metadata for a video."""
    metadata = {}
    
    # Default summary intent
    if not summary_intent:
        summary_intent = "Provide a comprehensive summary of the video content, plot, key scenes, and themes"
    
    # Summary
    print("    📝 Summary...")
    resp = api_post(token, 
        f'/api/v0/video-summarization/datasets/{dataset_id}/videos/{video_id}/summarize',
        {"summary_intent": summary_intent}
    )
    if resp and 'summary' in resp:
        metadata['video_narrative_summary'] = resp['summary']
        print(f"       ✅ Got summary ({len(resp['summary'])} chars)")
    elif resp and 'detail' in resp:
        print(f"       ⚠️ {str(resp['detail'])[:60]}")
    
    # Description
    print("    📄 Description...")
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/description',
        {}
    )
    if resp and 'description' in resp:
        metadata['video_narrative_description'] = resp['description']
        print("       ✅ Got description")
    
    # Genre
    print("    🎬 Genre...")
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/genre',
        {}
    )
    if resp and 'genres' in resp:
        metadata['video_narrative_genre'] = ', '.join(resp['genres'])
        print(f"       ✅ {metadata['video_narrative_genre']}")
    
    # Mood
    print("    🎭 Mood...")
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/mood',
        {}
    )
    if resp and 'moods' in resp:
        metadata['video_narrative_mood'] = ', '.join(resp['moods'])
        print(f"       ✅ {metadata['video_narrative_mood']}")
    
    # Subject
    print("    📚 Subject...")
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/subject',
        {}
    )
    if resp and 'subjects' in resp:
        metadata['video_narrative_subject'] = ', '.join(resp['subjects'])
        print(f"       ✅ {metadata['video_narrative_subject']}")
    
    # Format
    print("    🎞️ Format...")
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/format',
        {}
    )
    if resp and 'formats' in resp:
        metadata['video_narrative_format'] = ', '.join(resp['formats'])
        print(f"       ✅ {metadata['video_narrative_format']}")
    
    # Caption Keyframes (async)
    print("    🖼️ Keyframes...")
    resp = api_post(token, 
        f'/api/v0/video-summarization/datasets/{dataset_id}/videos/{video_id}/caption-keyframes',
        {}
    )
    if resp and 'message' in resp:
        metadata['video_narrative_keyframes_requested'] = 'true'
        print(f"       ✅ {resp['message'][:60]}")
    
    # Add metadata source and timestamp
    metadata['video_narrative_metadata_source'] = 'Coactive Video Narrative API'
    metadata['video_narrative_metadata_generated_at'] = datetime.now(timezone.utc).isoformat()
    
    return metadata


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
  python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN
  python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --video-id VIDEO_ID
  python3 coactive_narrative_enrichment.py -d DATASET_ID -t TOKEN --intent "Focus on action scenes"
        """
    )
    
    parser.add_argument('--dataset-id', '-d', required=True,
                        help='Coactive Dataset ID (UUID)')
    parser.add_argument('--token', '-t', required=True,
                        help='Coactive Refresh Token (Personal Token)')
    parser.add_argument('--video-id', '-v',
                        help='Process only this specific video ID')
    parser.add_argument('--intent', '-i',
                        help='Custom summary intent (default: comprehensive summary)')
    parser.add_argument('--limit', '-l', type=int, default=100,
                        help='Max number of videos to process (default: 100)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between videos in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎬 COACTIVE NARRATIVE METADATA ENRICHMENT")
    print("=" * 80)
    print(f"Dataset: {args.dataset_id}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.intent:
        print(f"Custom Intent: {args.intent}")
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
                 filename)
        
        print(f"\n[{idx}/{len(videos)}] 🎬 {title}")
        print(f"    ID: {video_id}")
        
        # Get narrative metadata
        metadata = get_narrative_metadata(token, args.dataset_id, video_id, args.intent)
        
        # Check if we got meaningful metadata (more than just source and timestamp)
        meaningful_fields = len([k for k in metadata.keys() 
                                if k not in ['video_narrative_metadata_source', 
                                           'video_narrative_metadata_generated_at']])
        
        if meaningful_fields > 0:
            print(f"    💾 Updating ({len(metadata)} fields)...")
            if update_video_metadata(token, args.dataset_id, video_id, metadata):
                print("       ✅ Metadata updated!")
                results.append({'video': title, 'status': 'success', 'fields': len(metadata)})
            else:
                print("       ❌ Metadata update failed")
                results.append({'video': title, 'status': 'update_failed'})
        else:
            print("    ❌ No metadata retrieved from APIs")
            results.append({'video': title, 'status': 'no_metadata'})
        
        # Rate limiting
        if idx < len(videos):
            time.sleep(args.delay)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 ENRICHMENT SUMMARY")
    print("=" * 80)
    success = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] != 'success'])
    
    print(f"✅ Successfully enriched: {success}/{len(results)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(results)}")
    
    print()
    for r in results:
        status_icon = "✅" if r['status'] == 'success' else "❌"
        fields = f"({r.get('fields', 0)} fields)" if r['status'] == 'success' else f"({r['status']})"
        print(f"  {status_icon} {r['video']} {fields}")
    
    print(f"\n🔗 View in Coactive: https://app.coactive.ai/datasets/{args.dataset_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()

