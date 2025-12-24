#!/usr/bin/env python3
"""
Narrative Metadata Benchmark Script
====================================
Runs narrative metadata API on all videos in a dataset and tracks latency.

Author: Auto-generated
Date: December 2025
"""

import json
import subprocess
import time
from datetime import datetime, timezone

# Configuration from user
REFRESH_TOKEN = "4tIiilfr_SCNbL07QA8T6WptrMHcEsx0CbqQQ6P2mDoDf"
DATASET_ID = "8b86dab7-1b90-4e24-a58b-67f5998228d1"
API_BASE_URL = "https://api.coactive.ai"
APP_BASE_URL = "https://app.coactive.ai"

# Videos to skip (already processed)
SKIP_VIDEO_IDS = [
    "70824273-4bc7-4ea5-9307-97ee3d23e6ee",
    "a4825239-27c4-49ac-8e94-109bb4581a8f"
]


def get_access_token():
    """Exchange refresh token for access token."""
    cmd = [
        'curl', '-s', '-X', 'POST',
        f'{API_BASE_URL}/api/v0/login',
        '-H', f'Authorization: Bearer {REFRESH_TOKEN}',
        '-H', 'Content-Type: application/json',
        '-d', '{"grant_type": "refresh_token"}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            return data.get('access_token')
        except:
            pass
    return None


def api_get(token, endpoint, base_url=None):
    """Make GET request to Coactive API."""
    url = base_url or APP_BASE_URL
    cmd = [
        'curl', '-s', '-X', 'GET',
        f'{url}{endpoint}',
        '-H', f'Authorization: Bearer {token}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout:
        try:
            return json.loads(result.stdout)
        except:
            return {"raw": result.stdout}
    return None


def api_post(token, endpoint, body=None, base_url=None):
    """Make POST request to Coactive API."""
    url = base_url or API_BASE_URL
    cmd = [
        'curl', '-s', '-X', 'POST',
        f'{url}{endpoint}',
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


def get_dataset_videos(token, dataset_id):
    """Get all videos from dataset."""
    data = api_get(token, f'/api/v1/datasets/{dataset_id}/videos?limit=500')
    if data and 'data' in data:
        return data['data']
    return []


def get_narrative_metadata_with_timing(token, dataset_id, video_id, title):
    """Get all narrative metadata for a video with timing."""
    metadata = {}
    timings = {}
    total_start = time.time()
    
    # Summary
    print("    📝 Getting summary...", end=" ", flush=True)
    start = time.time()
    resp = api_post(token, 
        f'/api/v0/video-summarization/datasets/{dataset_id}/videos/{video_id}/summarize',
        {"summary_intent": "Provide a comprehensive summary of the video content, plot, key scenes, and themes"}
    )
    elapsed = time.time() - start
    timings['summary'] = elapsed
    if resp and 'summary' in resp:
        metadata['video_narrative_summary'] = resp['summary']
        print(f"✅ ({elapsed:.2f}s)")
    elif resp and 'detail' in resp:
        print(f"⚠️ {resp['detail'][:50]} ({elapsed:.2f}s)")
    else:
        print(f"❌ ({elapsed:.2f}s)")
    
    # Description
    print("    📄 Getting description...", end=" ", flush=True)
    start = time.time()
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/description',
        {}
    )
    elapsed = time.time() - start
    timings['description'] = elapsed
    if resp and 'description' in resp:
        metadata['video_narrative_description'] = resp['description']
        print(f"✅ ({elapsed:.2f}s)")
    else:
        print(f"❌ ({elapsed:.2f}s)")
    
    # Genre
    print("    🎬 Getting genre...", end=" ", flush=True)
    start = time.time()
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/genre',
        {}
    )
    elapsed = time.time() - start
    timings['genre'] = elapsed
    if resp and 'genres' in resp:
        metadata['video_narrative_genre'] = ', '.join(resp['genres'])
        print(f"✅ ({elapsed:.2f}s)")
    else:
        print(f"❌ ({elapsed:.2f}s)")
    
    # Mood
    print("    🎭 Getting mood...", end=" ", flush=True)
    start = time.time()
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/mood',
        {}
    )
    elapsed = time.time() - start
    timings['mood'] = elapsed
    if resp and 'moods' in resp:
        metadata['video_narrative_mood'] = ', '.join(resp['moods'])
        print(f"✅ ({elapsed:.2f}s)")
    else:
        print(f"❌ ({elapsed:.2f}s)")
    
    # Subject
    print("    📚 Getting subject...", end=" ", flush=True)
    start = time.time()
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/subject',
        {}
    )
    elapsed = time.time() - start
    timings['subject'] = elapsed
    if resp and 'subjects' in resp:
        metadata['video_narrative_subject'] = ', '.join(resp['subjects'])
        print(f"✅ ({elapsed:.2f}s)")
    else:
        print(f"❌ ({elapsed:.2f}s)")
    
    # Format
    print("    🎞️ Getting format...", end=" ", flush=True)
    start = time.time()
    resp = api_post(token, 
        f'/api/v0/video-narrative-metadata/datasets/{dataset_id}/videos/{video_id}/format',
        {}
    )
    elapsed = time.time() - start
    timings['format'] = elapsed
    if resp and 'formats' in resp:
        metadata['video_narrative_format'] = ', '.join(resp['formats'])
        print(f"✅ ({elapsed:.2f}s)")
    else:
        print(f"❌ ({elapsed:.2f}s)")
    
    # Caption Keyframes (async)
    print("    🖼️ Triggering keyframe captioning...", end=" ", flush=True)
    start = time.time()
    resp = api_post(token, 
        f'/api/v0/video-summarization/datasets/{dataset_id}/videos/{video_id}/caption-keyframes',
        {}
    )
    elapsed = time.time() - start
    timings['keyframes'] = elapsed
    if resp and 'message' in resp:
        metadata['video_narrative_keyframes_requested'] = 'true'
        print(f"✅ ({elapsed:.2f}s)")
    else:
        print(f"❌ ({elapsed:.2f}s)")
    
    # Add metadata source and timestamp
    metadata['video_narrative_metadata_source'] = 'Coactive Video Narrative API'
    metadata['video_narrative_metadata_generated_at'] = datetime.now(timezone.utc).isoformat()
    
    total_elapsed = time.time() - total_start
    timings['total'] = total_elapsed
    
    return metadata, timings


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
    print("=" * 80)
    print("🎬 NARRATIVE METADATA BENCHMARK")
    print("=" * 80)
    print(f"Dataset: {DATASET_ID}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Skipping videos: {SKIP_VIDEO_IDS}")
    print("=" * 80)
    
    # Authenticate
    print("\n🔐 Authenticating...")
    token = get_access_token()
    if not token:
        print("❌ Authentication failed")
        return
    print("✅ Got access token")
    
    overall_start = time.time()
    
    # Get all videos
    print("\n📊 Fetching videos from dataset...")
    videos = get_dataset_videos(token, DATASET_ID)
    
    if not videos:
        print("❌ No videos found or API error")
        return
    
    print(f"✅ Found {len(videos)} total videos")
    
    # Filter out skipped videos
    videos_to_process = [v for v in videos if v.get('coactiveVideoId') not in SKIP_VIDEO_IDS]
    skipped_count = len(videos) - len(videos_to_process)
    
    print(f"   Skipping {skipped_count} already processed videos")
    print(f"   Processing {len(videos_to_process)} videos\n")
    
    all_timings = []
    results = []
    
    for idx, video in enumerate(videos_to_process, 1):
        video_id = video.get('coactiveVideoId')
        path = video.get('path', '')
        filename = path.split('/')[-1] if path else 'Unknown'
        
        # Get title from metadata or filename
        existing_meta = video.get('metadata', {})
        title = existing_meta.get('original_title') or existing_meta.get('imdb_title') or filename
        
        print(f"\n[{idx}/{len(videos_to_process)}] 🎬 {title[:60]}")
        print(f"    Video ID: {video_id}")
        
        # Get narrative metadata with timing
        metadata, timings = get_narrative_metadata_with_timing(token, DATASET_ID, video_id, title)
        all_timings.append({'video': title[:40], 'timings': timings})
        
        if metadata:
            print(f"    ✅ Got {len(metadata)} metadata fields in {timings['total']:.2f}s")
            
            # Update video
            print("    💾 Updating metadata...", end=" ", flush=True)
            start = time.time()
            if update_video_metadata(token, DATASET_ID, video_id, metadata):
                print(f"✅ ({time.time() - start:.2f}s)")
                results.append({'video': title, 'status': 'success', 'fields': len(metadata), 'total_time': timings['total']})
            else:
                print(f"❌ ({time.time() - start:.2f}s)")
                results.append({'video': title, 'status': 'update_failed', 'total_time': timings['total']})
        else:
            print("    ❌ Failed to get narrative metadata")
            results.append({'video': title, 'status': 'api_failed', 'total_time': timings.get('total', 0)})
        
        # Small delay between videos
        time.sleep(1)
    
    overall_elapsed = time.time() - overall_start
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS")
    print("=" * 80)
    
    success = len([r for r in results if r['status'] == 'success'])
    print(f"✅ Successfully enriched: {success}/{len(results)}")
    print(f"⏱️  Total time: {overall_elapsed:.2f}s ({overall_elapsed/60:.1f} minutes)")
    
    if results:
        avg_time = sum(r.get('total_time', 0) for r in results) / len(results)
        print(f"⏱️  Average time per video: {avg_time:.2f}s")
    
    print("\n📋 Per-video results:")
    for r in results:
        status_icon = "✅" if r['status'] == 'success' else "❌"
        time_str = f"{r.get('total_time', 0):.2f}s"
        print(f"  {status_icon} {r['video'][:50]:50s} | {time_str:8s} | {r['status']}")
    
    print("\n📋 Per-endpoint average latencies:")
    if all_timings:
        # Calculate averages per endpoint
        endpoints = ['summary', 'description', 'genre', 'mood', 'subject', 'format', 'keyframes']
        for endpoint in endpoints:
            times = [t['timings'].get(endpoint, 0) for t in all_timings if endpoint in t['timings']]
            if times:
                avg = sum(times) / len(times)
                print(f"  {endpoint:15s}: avg {avg:.2f}s, min {min(times):.2f}s, max {max(times):.2f}s")
    
    print(f"\n🔗 View in Coactive: https://app.coactive.ai/datasets/{DATASET_ID}")
    print("=" * 80)


if __name__ == "__main__":
    main()

