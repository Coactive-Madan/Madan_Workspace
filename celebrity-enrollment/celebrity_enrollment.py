#!/usr/bin/env python3
"""
Celebrity Detection - Person Enrollment Script
===============================================
Enrolls persons into the Coactive Celebrity Detection system.
Config-driven and reusable across any client or use case.

Enrollment flow (4-step):
  1. Create draft person  (POST /api/v0/celebrity-detection/enroll)
  2. Upload seed images   (POST /api/v0/celebrity-detection/upload)
  3. Update draft with upload_ids + aliases (PATCH /api/v0/celebrity-detection/enroll/{person_id})
  4. Finalize enrollment  (POST /api/v0/celebrity-detection/enroll/{person_id}/finalize)

Usage:
    # With config file:
    python3 celebrity_enrollment.py --config config.json

    # With env vars:
    COACTIVE_API_KEY="..." COACTIVE_DATASET_ID="..." python3 celebrity_enrollment.py

    # List enrolled persons only:
    python3 celebrity_enrollment.py --config config.json --list-only
"""

import os
import argparse
import base64
import json
import glob
import sys
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from JSON file, falling back to environment variables.

    Config file schema:
    {
        "client_name": "Trajektory",
        "coactive_api_key": "...",
        "base_url": "https://api.coactive.ai",
        "dataset_id": "...",
        "seed_images_dir": "./seed_images",
        "celebrities": [
            {
                "name": "Person Name",
                "aliases": ["Alt Name"],
                "image_pattern": "Person*.png",
                "max_images": 18
            }
        ]
    }
    """
    config = {
        "client_name": "Default",
        "coactive_api_key": os.environ.get("COACTIVE_API_KEY", ""),
        "base_url": os.environ.get("COACTIVE_BASE_URL", "https://api.coactive.ai"),
        "dataset_id": os.environ.get("COACTIVE_DATASET_ID", ""),
        "seed_images_dir": os.environ.get("SEED_IMAGES_DIR", "./seed_images"),
        "celebrities": [],
    }

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            file_config = json.load(f)
        # File values override defaults (but not empty strings over env vars)
        for key in file_config:
            if file_config[key]:
                config[key] = file_config[key]
        print(f"  Config loaded from: {config_path}")
    else:
        print(f"  Config: using environment variables")

    # Validate required fields
    if not config["coactive_api_key"]:
        print("ERROR: coactive_api_key not set (use config file or COACTIVE_API_KEY env var)")
        sys.exit(1)

    return config


# =============================================================================
# AUTH
# =============================================================================

def get_jwt_token(config: dict) -> str:
    """Exchange personal API token for JWT via /api/v0/login."""
    print("  Authenticating...")

    auth_url = f"{config['base_url']}/api/v0/login"
    headers = {
        "Authorization": f"Bearer {config['coactive_api_key']}",
        "Content-Type": "application/json",
    }

    resp = requests.post(auth_url, headers=headers,
                         json={"grant_type": "refresh_token"},
                         verify=False, timeout=60)

    if resp.status_code == 200:
        token = (resp.json().get("access_token")
                 or resp.json().get("token")
                 or resp.json().get("jwt"))
        if token:
            print(f"  JWT obtained: {token[:40]}...")
            return token

    raise ValueError(f"Auth failed (HTTP {resp.status_code}): {resp.text[:300]}")


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and encode it to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_seed_images(seed_dir: str, image_pattern: str, max_images: int = 18) -> list:
    """Find and load seed images matching a glob pattern."""
    pattern = os.path.join(seed_dir, image_pattern)
    image_files = sorted(glob.glob(pattern))

    if not image_files:
        print(f"  WARNING: No images found matching: {pattern}")
        return []

    use_count = min(len(image_files), max_images)
    print(f"  Found {len(image_files)} images, using {use_count}")

    images = []
    for img_path in image_files[:max_images]:
        filename = os.path.basename(img_path)
        try:
            base64_data = encode_image_to_base64(img_path)
            file_size_kb = os.path.getsize(img_path) / 1024
            print(f"    {filename} ({file_size_kb:.1f} KB)")
            images.append({
                "filename": filename,
                "base64": base64_data,
                "path": img_path,
            })
        except Exception as e:
            print(f"    {filename} - Error: {e}")

    return images


# =============================================================================
# API HELPER
# =============================================================================

def api_call(method: str, url: str, headers: dict, **kwargs) -> requests.Response:
    """Make API call with logging."""
    print(f"    {method} {url}")
    resp = requests.request(method, url, headers=headers,
                            verify=False, timeout=120, **kwargs)
    print(f"    -> HTTP {resp.status_code} ({len(resp.content)} bytes)")
    return resp


# =============================================================================
# ENROLLMENT STEPS
# =============================================================================

def step1_create_draft(base_url: str, token: str, person_name: str) -> str:
    """Create a draft person entry. Returns person_id."""
    print(f"\n  Step 1: Create draft person '{person_name}'")

    celeb_base = f"{base_url}/api/v0/celebrity-detection"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    resp = api_call("POST", f"{celeb_base}/enroll", headers,
                    json={"person_name": person_name})

    if resp.status_code in (200, 201, 202):
        data = resp.json()
        person_id = data.get("person_id") or data.get("id") or data.get("draft_id")
        print(f"    Draft created: {person_id}")
        return person_id

    print(f"    FAILED: {resp.text[:500]}")
    return None


def step2_upload_images(base_url: str, token: str, images: list) -> list:
    """Upload seed images and return upload_ids. Tries multiple endpoints/formats."""
    print(f"\n  Step 2: Upload {len(images)} seed images")

    celeb_base = f"{base_url}/api/v0/celebrity-detection"
    headers_json = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    headers_multipart = {"Authorization": f"Bearer {token}"}

    upload_endpoints = [
        f"{celeb_base}/upload",
        f"{celeb_base}/uploads",
        f"{base_url}/api/v0/uploads",
        f"{base_url}/api/v0/upload",
        f"{base_url}/api/v1/uploads",
    ]

    # --- Approach A: Batch base64 upload ---
    for endpoint in upload_endpoints:
        payload = {"images": [{"filename": img["filename"], "data": img["base64"]} for img in images]}
        resp = api_call("POST", endpoint, headers_json, json=payload)
        if resp.status_code in (200, 201, 202):
            data = resp.json()
            ids = data.get("upload_ids") or data.get("ids") or data.get("file_ids") or []
            if ids:
                print(f"    Batch upload succeeded: {len(ids)} IDs")
                return ids
            if isinstance(data, list):
                return [item.get("id") or item.get("upload_id") for item in data
                        if item.get("id") or item.get("upload_id")]
            break
        elif resp.status_code == 404:
            continue

    # --- Approach B: Individual multipart form uploads ---
    for endpoint in upload_endpoints:
        first_img = images[0]
        with open(first_img["path"], "rb") as f:
            files = {"file": (first_img["filename"], f, "image/png")}
            resp = api_call("POST", endpoint, headers_multipart, files=files)

        if resp.status_code in (200, 201, 202):
            data = resp.json()
            upload_id = data.get("upload_id") or data.get("id") or data.get("file_id")
            if upload_id:
                upload_ids = [upload_id]
                for img in images[1:]:
                    with open(img["path"], "rb") as f:
                        files = {"file": (img["filename"], f, "image/png")}
                        r2 = api_call("POST", endpoint, headers_multipart, files=files)
                        if r2.status_code in (200, 201, 202):
                            uid = r2.json().get("upload_id") or r2.json().get("id") or r2.json().get("file_id")
                            if uid:
                                upload_ids.append(uid)
                print(f"    Multipart upload: {len(upload_ids)} IDs")
                return upload_ids
            break
        elif resp.status_code == 404:
            continue

    # --- Approach C: Individual base64 uploads ---
    for endpoint in upload_endpoints:
        first_img = images[0]
        payload = {"filename": first_img["filename"], "data": first_img["base64"], "image": first_img["base64"]}
        resp = api_call("POST", endpoint, headers_json, json=payload)
        if resp.status_code in (200, 201, 202):
            upload_id = resp.json().get("upload_id") or resp.json().get("id") or resp.json().get("file_id")
            if upload_id:
                upload_ids = [upload_id]
                for img in images[1:]:
                    payload = {"filename": img["filename"], "data": img["base64"], "image": img["base64"]}
                    r2 = api_call("POST", endpoint, headers_json, json=payload)
                    if r2.status_code in (200, 201, 202):
                        uid = r2.json().get("upload_id") or r2.json().get("id") or r2.json().get("file_id")
                        if uid:
                            upload_ids.append(uid)
                print(f"    Base64 upload: {len(upload_ids)} IDs")
                return upload_ids
            break
        elif resp.status_code == 404:
            continue

    print(f"    WARNING: No working upload endpoint found")
    return []


def step3_update_draft(base_url: str, token: str, person_id: str,
                       upload_ids: list, aliases: list = None) -> bool:
    """Update draft person with upload_ids and aliases."""
    print(f"\n  Step 3: Update draft {person_id} with {len(upload_ids)} images")

    celeb_base = f"{base_url}/api/v0/celebrity-detection"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"upload_ids": upload_ids}
    if aliases:
        payload["person_aliases"] = aliases

    resp = api_call("PATCH", f"{celeb_base}/enroll/{person_id}", headers, json=payload)

    if resp.status_code in (200, 201, 202):
        print(f"    Draft updated")
        return True

    print(f"    FAILED: {resp.text[:500]}")
    return False


def step4_finalize(base_url: str, token: str, person_id: str) -> bool:
    """Finalize enrollment (triggers backfill scanning)."""
    print(f"\n  Step 4: Finalize enrollment for {person_id}")

    celeb_base = f"{base_url}/api/v0/celebrity-detection"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    resp = api_call("POST", f"{celeb_base}/enroll/{person_id}/finalize", headers)

    if resp.status_code in (200, 201, 202):
        print(f"    Enrollment finalized (backfill triggered)")
        return True

    print(f"    FAILED: {resp.text[:500]}")
    return False


def try_legacy_enroll(base_url: str, token: str, person_name: str,
                      upload_ids: list, aliases: list = None) -> bool:
    """Fallback: Legacy single-step enrollment."""
    print(f"\n  Fallback: Legacy enrollment for '{person_name}'")

    celeb_base = f"{base_url}/api/v0/celebrity-detection"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"person_name": person_name, "upload_ids": upload_ids}
    if aliases:
        payload["person_aliases"] = aliases

    resp = api_call("POST", f"{celeb_base}/enroll-legacy", headers, json=payload)

    if resp.status_code in (200, 201, 202):
        print(f"    Legacy enrollment succeeded")
        return True

    print(f"    Legacy endpoint failed: {resp.text[:500]}")
    return False


# =============================================================================
# QUERY HELPERS
# =============================================================================

def list_persons(base_url: str, token: str) -> list:
    """List all enrolled persons."""
    celeb_base = f"{base_url}/api/v0/celebrity-detection"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    resp = api_call("GET", f"{celeb_base}/persons", headers)

    if resp.status_code == 200:
        data = resp.json()
        persons = data if isinstance(data, list) else data.get("persons", data.get("items", []))
        print(f"    {len(persons)} enrolled persons")
        for p in persons:
            name = p.get("person_name") or p.get("name", "unknown")
            pid = p.get("person_id") or p.get("id", "?")
            status = p.get("status", "?")
            print(f"      {name} (id={pid}, status={status})")
        return persons

    print(f"    Failed: {resp.status_code}")
    return []


def get_faces_for_person(base_url: str, token: str, person_name: str,
                         dataset_id: str = None) -> list:
    """Fetch face detections for an enrolled person."""
    celeb_base = f"{base_url}/api/v0/celebrity-detection"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    resp = requests.get(f"{celeb_base}/faces-with-person", headers=headers,
                        params={"person_name": person_name},
                        verify=False, timeout=120)

    if resp.status_code != 200:
        print(f"    {person_name}: HTTP {resp.status_code}")
        return []

    faces = resp.json().get("faces", [])

    if dataset_id:
        faces = [f for f in faces if f.get("dataset_id") == dataset_id]

    return faces


# =============================================================================
# ENROLL ONE PERSON (orchestrates steps 1-4)
# =============================================================================

def enroll_person(config: dict, token: str, celebrity: dict) -> dict:
    """
    Enroll a single person. Returns result dict with status.

    celebrity: {"name": str, "aliases": list, "image_pattern": str, "max_images": int}
    """
    name = celebrity["name"]
    base_url = config["base_url"]
    seed_dir = config["seed_images_dir"]

    result = {"name": name, "status": "failed", "person_id": None, "face_count": 0}

    # Load seed images
    images = load_seed_images(
        seed_dir,
        celebrity.get("image_pattern", f"{name.split()[0]}*.png"),
        celebrity.get("max_images", 18),
    )
    if not images:
        result["status"] = "no_images"
        return result

    # Step 1: Create draft
    person_id = step1_create_draft(base_url, token, name)
    if not person_id:
        result["status"] = "draft_failed"
        return result
    result["person_id"] = person_id

    # Step 2: Upload images
    upload_ids = step2_upload_images(base_url, token, images)

    if upload_ids:
        # Step 3: Update draft
        updated = step3_update_draft(
            base_url, token, person_id, upload_ids,
            aliases=celebrity.get("aliases"),
        )

        if updated:
            # Step 4: Finalize
            if step4_finalize(base_url, token, person_id):
                result["status"] = "enrolled"
            else:
                result["status"] = "finalize_failed"
        else:
            # Fallback to legacy
            if try_legacy_enroll(base_url, token, name, upload_ids, celebrity.get("aliases")):
                result["status"] = "enrolled_legacy"
            else:
                result["status"] = "update_failed"
    else:
        # No upload IDs — try finalizing anyway
        print("  No upload_ids obtained. Trying finalize with draft only...")
        step4_finalize(base_url, token, person_id)
        result["status"] = "draft_only"

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enroll persons into Coactive Celebrity Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll from config file:
  python3 celebrity_enrollment.py --config config.json

  # List enrolled persons:
  python3 celebrity_enrollment.py --config config.json --list-only

  # Check face detections after backfill:
  python3 celebrity_enrollment.py --config config.json --check-faces
        """,
    )
    parser.add_argument("--config", "-c", help="Path to config JSON file")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list enrolled persons, don't enroll")
    parser.add_argument("--check-faces", action="store_true",
                        help="Check face detection counts for enrolled persons")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print(f"Celebrity Detection Enrollment — {config['client_name']}")
    print("=" * 60)
    print(f"  Dataset:      {config.get('dataset_id', 'N/A')}")
    print(f"  Seed images:  {config['seed_images_dir']}")
    print(f"  Celebrities:  {len(config.get('celebrities', []))}")
    print("=" * 60)

    # Auth
    token = get_jwt_token(config)

    # List-only mode
    if args.list_only:
        print("\n[Enrolled Persons]")
        list_persons(config["base_url"], token)
        return

    # Check-faces mode
    if args.check_faces:
        print("\n[Face Detection Counts]")
        persons = list_persons(config["base_url"], token)
        dataset_id = config.get("dataset_id")
        for p in persons:
            name = p.get("person_name") or p.get("name", "unknown")
            faces = get_faces_for_person(config["base_url"], token, name, dataset_id)
            print(f"    {name}: {len(faces)} faces in dataset")
        return

    # Enrollment mode
    celebrities = config.get("celebrities", [])
    if not celebrities:
        print("\nNo celebrities configured. Add them to config.json or use --list-only.")
        return

    print(f"\n[Enrolling {len(celebrities)} persons]")
    print(f"  Existing persons:")
    list_persons(config["base_url"], token)

    results = []
    for i, celebrity in enumerate(celebrities, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(celebrities)}] {celebrity['name']}")
        print("=" * 60)
        result = enroll_person(config, token, celebrity)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("ENROLLMENT SUMMARY")
    print("=" * 60)
    for r in results:
        status_icon = {
            "enrolled": "OK", "enrolled_legacy": "OK (legacy)",
            "no_images": "SKIP (no images)", "draft_failed": "FAIL (draft)",
            "update_failed": "FAIL (update)", "finalize_failed": "FAIL (finalize)",
            "draft_only": "PARTIAL (no uploads)", "failed": "FAIL",
        }.get(r["status"], r["status"])
        print(f"  {r['name']}: {status_icon} (person_id={r['person_id']})")

    print(f"\n[Final State]")
    list_persons(config["base_url"], token)
    print("=" * 60)


if __name__ == "__main__":
    main()
