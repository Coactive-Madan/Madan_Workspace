#!/usr/bin/env python3
"""
HYBE Celebrity Detection - Artist Enrollment Script
====================================================
This script enrolls HYBE artists (starting with Sunghoon from ENHYPEN) 
into the Coactive Celebrity Detection system.

Usage:
    python hybe_celebrity_enrollment.py
"""

import os
import base64
import json
import glob
import requests
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Coactive API Configuration
COACTIVE_API_KEY = "xhSWjZhY7kUeB3E9SJS8R8LgOP-P1V0Tklu0SdyvED747"
COACTIVE_BASE_URL = "https://api.coactive.ai"
COACTIVE_AUTH_URL = f"{COACTIVE_BASE_URL}/v1/auth/token"

# Dataset Information
DATASET_ID = "6f8cca8f-67d2-4212-ae0b-e4195194bad2"

# Directory containing seed images
SEED_IMAGES_DIR = "/Users/madan/Downloads"

# Artists to enroll (add more as needed)
ARTISTS = [
    {
        "name": "Park Sunghoon",
        "group": "ENHYPEN", 
        "image_pattern": "Sunghoon_*.jpg",
        "max_images": 5  # Use top 5 images for enrollment
    },
    # Future artists can be added here:
    # {
    #     "name": "Jay",
    #     "group": "ENHYPEN",
    #     "image_pattern": "Jay*.jpeg",
    #     "max_images": 5
    # },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_jwt_token(api_key: str) -> str:
    """Exchange API key for JWT token."""
    print("🔑 Exchanging API key for JWT token...")
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    try:
        response = requests.post(
            COACTIVE_AUTH_URL,
            headers=headers,
            json={}
        )
        response.raise_for_status()
        
        token_data = response.json()
        jwt_token = token_data.get("access_token") or token_data.get("token")
        
        if jwt_token:
            print("✅ JWT token obtained successfully!")
            return jwt_token
        else:
            # If the API key is used directly as Bearer token
            print("ℹ️  Using API key directly as authorization...")
            return api_key
            
    except requests.exceptions.HTTPError as e:
        print(f"⚠️  Token exchange returned {e.response.status_code}")
        print("ℹ️  Attempting to use API key directly...")
        return api_key
    except Exception as e:
        print(f"⚠️  Token exchange error: {e}")
        print("ℹ️  Attempting to use API key directly...")
        return api_key


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and encode it to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_artist_images(image_pattern: str, max_images: int = 5) -> list:
    """
    Find and load artist seed images.
    Returns list of (filename, base64_data) tuples.
    """
    pattern = os.path.join(SEED_IMAGES_DIR, image_pattern)
    image_files = sorted(glob.glob(pattern))
    
    if not image_files:
        print(f"⚠️  No images found matching pattern: {pattern}")
        return []
    
    print(f"📷 Found {len(image_files)} images, using top {min(len(image_files), max_images)}")
    
    images = []
    for img_path in image_files[:max_images]:
        filename = os.path.basename(img_path)
        try:
            base64_data = encode_image_to_base64(img_path)
            file_size_kb = os.path.getsize(img_path) / 1024
            print(f"   ✓ {filename} ({file_size_kb:.1f} KB)")
            images.append({
                "filename": filename,
                "base64": base64_data,
                "path": img_path
            })
        except Exception as e:
            print(f"   ✗ {filename} - Error: {e}")
    
    return images


def enroll_celebrity(token: str, artist_name: str, images: list, dataset_id: str) -> dict:
    """
    Enroll a celebrity in the Coactive Celebrity Detection system.
    This will trigger a backfill job to scan existing content.
    """
    print(f"\n🎭 Enrolling {artist_name}...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Prepare the enrollment payload
    # Note: Adjust endpoint and payload structure based on actual Coactive API docs
    enrollment_endpoints = [
        f"{COACTIVE_BASE_URL}/v1/datasets/{dataset_id}/celebrities",
        f"{COACTIVE_BASE_URL}/v1/celebrities/enroll",
        f"{COACTIVE_BASE_URL}/celebrity-detection/enroll",
    ]
    
    payload = {
        "person_name": artist_name,
        "dataset_id": dataset_id,
        "face_images": [img["base64"] for img in images],
        # Alternative formats the API might expect:
        "images": [
            {
                "filename": img["filename"],
                "data": img["base64"]
            }
            for img in images
        ]
    }
    
    # Try different endpoint patterns
    for endpoint in enrollment_endpoints:
        print(f"   Trying endpoint: {endpoint}")
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200 or response.status_code == 201:
                print(f"✅ Successfully enrolled {artist_name}!")
                return {
                    "success": True,
                    "endpoint": endpoint,
                    "response": response.json() if response.text else {}
                }
            elif response.status_code == 404:
                print(f"   → Endpoint not found, trying next...")
                continue
            else:
                print(f"   → Response {response.status_code}: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print(f"   → Request timed out")
        except Exception as e:
            print(f"   → Error: {e}")
    
    return {"success": False, "error": "All endpoints failed"}


def check_enrollment_status(token: str, dataset_id: str) -> dict:
    """Check the status of celebrity enrollments and backfill jobs."""
    print("\n📊 Checking enrollment status...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    endpoints = [
        f"{COACTIVE_BASE_URL}/v1/datasets/{dataset_id}/celebrities",
        f"{COACTIVE_BASE_URL}/v1/datasets/{dataset_id}/jobs",
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, headers=headers, timeout=30)
            if response.status_code == 200:
                print(f"✅ Status from {endpoint}:")
                data = response.json()
                print(json.dumps(data, indent=2)[:500])
                return data
        except Exception as e:
            print(f"   → Error checking {endpoint}: {e}")
    
    return {}


def list_detected_faces(token: str, dataset_id: str, person_name: str) -> dict:
    """Query for images containing the enrolled person."""
    print(f"\n🔍 Searching for {person_name} in dataset...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Try the images_with_person endpoint mentioned in the tech doc
    endpoints = [
        f"{COACTIVE_BASE_URL}/v1/datasets/{dataset_id}/images_with_person",
        f"{COACTIVE_BASE_URL}/v1/datasets/{dataset_id}/search/faces",
    ]
    
    params = {
        "person_name": person_name,
        "limit": 10
    }
    
    for endpoint in endpoints:
        try:
            response = requests.get(
                endpoint, 
                headers=headers, 
                params=params,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found results:")
                print(json.dumps(data, indent=2)[:500])
                return data
        except Exception as e:
            print(f"   → Error: {e}")
    
    return {}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("🎬 HYBE Celebrity Detection - Artist Enrollment")
    print("=" * 60)
    print(f"Dataset: {DATASET_ID}")
    print(f"Dataset URL: https://app.coactive.ai/datasets/{DATASET_ID}?tab=videos")
    print("=" * 60)
    
    # Step 1: Get JWT token
    token = get_jwt_token(COACTIVE_API_KEY)
    
    # Step 2: Process each artist
    for artist in ARTISTS:
        print(f"\n{'='*60}")
        print(f"Processing: {artist['name']} ({artist['group']})")
        print("=" * 60)
        
        # Load seed images
        images = get_artist_images(
            artist["image_pattern"],
            artist["max_images"]
        )
        
        if not images:
            print(f"❌ Skipping {artist['name']} - no images found")
            continue
        
        # Enroll the celebrity
        result = enroll_celebrity(
            token=token,
            artist_name=artist["name"],
            images=images,
            dataset_id=DATASET_ID
        )
        
        if result.get("success"):
            print(f"\n✅ {artist['name']} enrolled successfully!")
            print("ℹ️  Backfill job has been triggered.")
            print("⏳ Wait 1-2 hours for K-NN search to propagate.")
        else:
            print(f"\n⚠️  Enrollment may require UI or different API endpoint")
            print("   Consider using the Roster Tab in the Coactive UI")
    
    # Step 3: Check status (optional)
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    check_enrollment_status(token, DATASET_ID)
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("=" * 60)
    print("1. If API enrollment failed, use the Roster Tab in the Coactive UI")
    print("2. Wait 1-2 hours for backfill to complete")
    print("3. Verify with: /images_with_person endpoint")
    print("4. Once verified, proceed to Phase 3 (Dynamic Tags for behaviors)")
    print("=" * 60)


if __name__ == "__main__":
    main()


