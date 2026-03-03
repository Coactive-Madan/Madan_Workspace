#!/usr/bin/env python3
"""
Celebrity Detection - Person Enrollment Script
===============================================
Enrolls a person into the Coactive Celebrity Detection system using the
3-step API flow from the Postman collection:
  1. Create draft person  (POST /api/v0/celebrity-detection/enroll)
  2. Upload seed images    (POST /api/v0/celebrity-detection/upload)
  3. Update draft with upload_ids (PATCH /api/v0/celebrity-detection/enroll/{person_id})
  4. Finalize enrollment   (POST /api/v0/celebrity-detection/enroll/{person_id}/finalize)

Usage:
    COACTIVE_API_KEY="..." COACTIVE_DATASET_ID="..." SEED_IMAGES_DIR="..." python3 hybe_celebrity_enrollment.py
"""

import os
import base64
import json
import glob
import sys
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

COACTIVE_API_KEY = os.environ.get("COACTIVE_API_KEY", "")
COACTIVE_BASE_URL = "https://api.coactive.ai"
COACTIVE_AUTH_URL = f"{COACTIVE_BASE_URL}/api/v0/login"
CELEBRITY_BASE = f"{COACTIVE_BASE_URL}/api/v0/celebrity-detection"

DATASET_ID = os.environ.get("COACTIVE_DATASET_ID", "")
SEED_IMAGES_DIR = os.environ.get("SEED_IMAGES_DIR", "./seed_images")

ARTISTS = [
    {
        "name": "Stephen Curry",
        "aliases": ["Steph Curry", "Wardell Stephen Curry II"],
        "image_pattern": "Curry*.png",
        "max_images": 18,
    },
]


# =============================================================================
# AUTH
# =============================================================================

def get_jwt_token(api_key: str) -> str:
    """Exchange personal API token for JWT via /api/v0/login."""
    print("🔑 Exchanging personal token for JWT...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"grant_type": "refresh_token"}

    response = requests.post(COACTIVE_AUTH_URL, headers=headers, json=payload, verify=False, timeout=60)

    if response.status_code == 200:
        token_data = response.json()
        jwt_token = token_data.get("access_token") or token_data.get("token") or token_data.get("jwt")
        if jwt_token:
            print(f"✅ JWT token obtained: {jwt_token[:40]}...")
            return jwt_token

    raise ValueError(f"Failed to get JWT (HTTP {response.status_code}): {response.text[:300]}")


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and encode it to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_artist_images(image_pattern: str, max_images: int = 18) -> list:
    """Find and load artist seed images."""
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
                "path": img_path,
            })
        except Exception as e:
            print(f"   ✗ {filename} - Error: {e}")

    return images


# =============================================================================
# ENROLLMENT API (3-step flow from Postman collection)
# =============================================================================

def api_call(method, url, headers, **kwargs):
    """Make API call and return response, with verbose logging."""
    print(f"   → {method} {url}")
    resp = requests.request(method, url, headers=headers, verify=False, timeout=120, **kwargs)
    print(f"   ← HTTP {resp.status_code} ({len(resp.content)} bytes)")
    return resp


def step1_create_draft_person(token: str, person_name: str) -> str:
    """Step 1: Create a draft person entry. Returns person_id."""
    print(f"\n📝 Step 1: Creating draft person '{person_name}'...")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"person_name": person_name}

    resp = api_call("POST", f"{CELEBRITY_BASE}/enroll", headers, json=payload)

    if resp.status_code in (200, 201, 202):
        data = resp.json()
        person_id = data.get("person_id") or data.get("id") or data.get("draft_id")
        print(f"   ✅ Draft person created: {person_id}")
        print(f"   Response: {json.dumps(data, indent=2)[:500]}")
        return person_id
    else:
        print(f"   ❌ Failed: {resp.text[:500]}")
        return None


def step2_upload_images(token: str, images: list) -> list:
    """
    Step 2: Upload seed face images and get upload_ids.
    Try multiple upload approaches since the exact endpoint isn't in the collection.
    """
    print(f"\n📤 Step 2: Uploading {len(images)} seed images...")

    headers_json = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    headers_no_ct = {
        "Authorization": f"Bearer {token}",
    }

    upload_ids = []

    # Approach A: Try uploading all images as base64 in one call
    upload_endpoints = [
        f"{CELEBRITY_BASE}/upload",
        f"{CELEBRITY_BASE}/uploads",
        f"{COACTIVE_BASE_URL}/api/v0/uploads",
        f"{COACTIVE_BASE_URL}/api/v0/upload",
        f"{COACTIVE_BASE_URL}/api/v1/uploads",
    ]

    # Try batch base64 upload
    for endpoint in upload_endpoints:
        print(f"\n   Trying batch upload: {endpoint}")
        payload = {
            "images": [
                {"filename": img["filename"], "data": img["base64"]}
                for img in images
            ]
        }
        resp = api_call("POST", endpoint, headers_json, json=payload)
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            print(f"   ✅ Batch upload succeeded: {json.dumps(data, indent=2)[:300]}")
            ids = data.get("upload_ids") or data.get("ids") or data.get("file_ids") or []
            if ids:
                return ids
            # Maybe it's a list of objects
            if isinstance(data, list):
                return [item.get("id") or item.get("upload_id") for item in data if item.get("id") or item.get("upload_id")]
            break
        elif resp.status_code == 404:
            print(f"   → 404, trying next...")
            continue
        else:
            print(f"   → {resp.status_code}: {resp.text[:200]}")

    # Approach B: Try individual file uploads (multipart form)
    print(f"\n   Trying individual multipart uploads...")
    for endpoint in upload_endpoints:
        print(f"\n   Trying multipart: {endpoint}")
        first_img = images[0]
        with open(first_img["path"], "rb") as f:
            files = {"file": (first_img["filename"], f, "image/png")}
            resp = api_call("POST", endpoint, headers_no_ct, files=files)

        if resp.status_code in (200, 201, 202):
            data = resp.json()
            print(f"   ✅ Multipart upload works: {json.dumps(data, indent=2)[:300]}")
            upload_id = data.get("upload_id") or data.get("id") or data.get("file_id")
            if upload_id:
                upload_ids.append(upload_id)
                # Upload remaining images
                for img in images[1:]:
                    with open(img["path"], "rb") as f:
                        files = {"file": (img["filename"], f, "image/png")}
                        r2 = api_call("POST", endpoint, headers_no_ct, files=files)
                        if r2.status_code in (200, 201, 202):
                            d2 = r2.json()
                            uid = d2.get("upload_id") or d2.get("id") or d2.get("file_id")
                            if uid:
                                upload_ids.append(uid)
                return upload_ids
            break
        elif resp.status_code == 404:
            continue
        else:
            print(f"   → {resp.status_code}: {resp.text[:200]}")
            break

    # Approach C: Try uploading individual base64 images
    print(f"\n   Trying individual base64 uploads...")
    for endpoint in upload_endpoints:
        print(f"\n   Trying single base64: {endpoint}")
        first_img = images[0]
        payload = {
            "filename": first_img["filename"],
            "data": first_img["base64"],
            "image": first_img["base64"],
        }
        resp = api_call("POST", endpoint, headers_json, json=payload)
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            print(f"   ✅ Single base64 upload works: {json.dumps(data, indent=2)[:300]}")
            upload_id = data.get("upload_id") or data.get("id") or data.get("file_id")
            if upload_id:
                upload_ids.append(upload_id)
                for img in images[1:]:
                    payload = {"filename": img["filename"], "data": img["base64"], "image": img["base64"]}
                    r2 = api_call("POST", endpoint, headers_json, json=payload)
                    if r2.status_code in (200, 201, 202):
                        d2 = r2.json()
                        uid = d2.get("upload_id") or d2.get("id") or d2.get("file_id")
                        if uid:
                            upload_ids.append(uid)
                return upload_ids
            break
        elif resp.status_code == 404:
            continue
        else:
            print(f"   → {resp.status_code}: {resp.text[:200]}")

    print(f"\n   ⚠️  Could not find working upload endpoint. Got {len(upload_ids)} upload_ids.")
    return upload_ids


def step3_update_draft(token: str, person_id: str, upload_ids: list, aliases: list = None) -> bool:
    """Step 3: Update draft person with upload_ids and optional aliases."""
    print(f"\n📎 Step 3: Updating draft {person_id} with {len(upload_ids)} image(s)...")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"upload_ids": upload_ids}
    if aliases:
        payload["person_aliases"] = aliases

    resp = api_call("PATCH", f"{CELEBRITY_BASE}/enroll/{person_id}", headers, json=payload)

    if resp.status_code in (200, 201, 202):
        print(f"   ✅ Draft updated with images")
        print(f"   Response: {resp.text[:300]}")
        return True
    else:
        print(f"   ❌ Failed: {resp.text[:500]}")
        return False


def step4_finalize(token: str, person_id: str) -> bool:
    """Step 4: Finalize the enrollment (triggers backfill)."""
    print(f"\n🚀 Step 4: Finalizing enrollment for {person_id}...")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    resp = api_call("POST", f"{CELEBRITY_BASE}/enroll/{person_id}/finalize", headers)

    if resp.status_code in (200, 201, 202):
        print(f"   ✅ Enrollment finalized!")
        print(f"   Response: {resp.text[:500]}")
        return True
    else:
        print(f"   ❌ Failed: {resp.text[:500]}")
        return False


def try_legacy_enroll(token: str, person_name: str, upload_ids: list, aliases: list = None) -> bool:
    """Fallback: Try legacy enrollment endpoint (single-step)."""
    print(f"\n🔄 Trying legacy enrollment endpoint...")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "person_name": person_name,
        "upload_ids": upload_ids,
    }
    if aliases:
        payload["person_aliases"] = aliases

    resp = api_call("POST", f"{CELEBRITY_BASE}/enroll-legacy", headers, json=payload)

    if resp.status_code in (200, 201, 202):
        print(f"   ✅ Legacy enrollment succeeded!")
        print(f"   Response: {resp.text[:500]}")
        return True
    else:
        print(f"   ❌ Legacy endpoint failed: {resp.text[:500]}")
        return False


def get_persons(token: str) -> list:
    """List all enrolled persons."""
    print(f"\n📋 Listing enrolled persons...")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    resp = api_call("GET", f"{CELEBRITY_BASE}/persons", headers)

    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✅ Persons: {json.dumps(data, indent=2)[:800]}")
        return data if isinstance(data, list) else data.get("persons", data.get("items", []))
    else:
        print(f"   → {resp.status_code}: {resp.text[:300]}")
        return []


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🎬 Celebrity Detection - Person Enrollment")
    print("=" * 60)
    print(f"  Dataset: {DATASET_ID}")
    print(f"  Images:  {SEED_IMAGES_DIR}")
    print("=" * 60)

    if not COACTIVE_API_KEY:
        print("❌ COACTIVE_API_KEY not set")
        sys.exit(1)

    # Auth
    token = get_jwt_token(COACTIVE_API_KEY)

    # List existing persons
    get_persons(token)

    # Process each artist
    for artist in ARTISTS:
        print(f"\n{'='*60}")
        print(f"🏀 Processing: {artist['name']}")
        print("=" * 60)

        images = get_artist_images(artist["image_pattern"], artist["max_images"])
        if not images:
            print(f"❌ No images found, skipping")
            continue

        # Step 1: Create draft
        person_id = step1_create_draft_person(token, artist["name"])

        if person_id:
            # Step 2: Upload images
            upload_ids = step2_upload_images(token, images)

            if upload_ids:
                # Step 3: Update draft with upload_ids
                updated = step3_update_draft(
                    token, person_id, upload_ids,
                    aliases=artist.get("aliases")
                )

                if updated:
                    # Step 4: Finalize
                    step4_finalize(token, person_id)
                else:
                    print("⚠️  Draft update failed, trying legacy...")
                    try_legacy_enroll(token, artist["name"], upload_ids, artist.get("aliases"))
            else:
                print("⚠️  No upload_ids obtained. Upload endpoint may not be in this collection.")
                print("   Trying to finalize with just the draft (images may need UI upload)...")
                step4_finalize(token, person_id)
        else:
            print("⚠️  Could not create draft person.")

    # Final summary
    print(f"\n{'='*60}")
    print("📊 Final Status - Enrolled Persons")
    print("=" * 60)
    get_persons(token)
    print("=" * 60)


if __name__ == "__main__":
    main()
