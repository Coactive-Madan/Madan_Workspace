#!/usr/bin/env python3
"""
Celebrity Detection + Dynamic Tags Join Workflow
=================================================
Fetches celebrity face detections and Dynamic Tag scores, joins them
on image_id, and generates Coactive UI SQL queries for the top results.

Config-driven and reusable across any client or use case.

Flow:
  1. Authenticate via JWT
  2. Fetch celebrity face detections from enrolled persons
  3. (Optional) Create DT group + tags if they don't exist yet
  4. Fetch DT scores (scoring-preview or asset-check)
  5. Join celeb faces with DT scores on image_id
  6. Generate Coactive UI SQL queries for top results

Usage:
    # Full workflow (faces + DT scores + join):
    python3 celeb_dt_workflow.py --config config.json

    # Fetch faces only:
    python3 celeb_dt_workflow.py --config config.json --faces-only

    # Setup DT group + tags (create + prompt + publish):
    python3 celeb_dt_workflow.py --config config.json --dt-setup

    # Fetch DT scores only:
    python3 celeb_dt_workflow.py --config config.json --dt-scores

    # Export joined data as JSON:
    python3 celeb_dt_workflow.py --config config.json --output results.json
"""

import os
import sys
import json
import argparse
import requests
import urllib3
from collections import defaultdict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_config(config_path: str = None) -> dict:
    """
    Load configuration from JSON file, falling back to environment variables.

    Config file should contain:
    {
        "client_name": "Client Name",
        "coactive_api_key": "...",
        "base_url": "https://api.coactive.ai",
        "dataset_id": "...",
        "celebrities": [
            {"name": "Person Name", "person_id": "uuid-if-already-enrolled"}
        ],
        "dt_groups": [
            {
                "name": "Group Name",
                "tags": {"Tag Name": "Description prompt for this tag"}
            }
        ]
    }
    """
    config = {
        "client_name": "Default",
        "coactive_api_key": os.environ.get("COACTIVE_API_KEY", ""),
        "base_url": os.environ.get("COACTIVE_BASE_URL", "https://api.coactive.ai"),
        "dataset_id": os.environ.get("COACTIVE_DATASET_ID", ""),
        "celebrities": [],
        "dt_groups": [],
    }

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            file_config = json.load(f)
        for key in file_config:
            if file_config[key]:
                config[key] = file_config[key]
        print(f"  Config loaded from: {config_path}")
    else:
        print(f"  Config: using environment variables")

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
    headers = {
        "Authorization": f"Bearer {config['coactive_api_key']}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{config['base_url']}/api/v0/login",
        headers=headers,
        json={"grant_type": "refresh_token"},
        verify=False, timeout=60,
    )
    if resp.status_code == 200:
        token = (resp.json().get("access_token")
                 or resp.json().get("token")
                 or resp.json().get("jwt"))
        if token:
            print(f"  JWT obtained: {token[:40]}...")
            return token

    raise ValueError(f"Auth failed (HTTP {resp.status_code}): {resp.text[:300]}")


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
# PHASE 1: CELEBRITY FACE DATA
# =============================================================================

def fetch_celebrity_faces(config: dict, token: str) -> list:
    """
    Fetch all celebrity face detections from the API.

    Returns list of dicts:
        {face_id, image_id, dataset_id, confidence, bbox, celebrity_name, person_id}
    """
    base_url = config["base_url"]
    dataset_id = config.get("dataset_id")
    celeb_base = f"{base_url}/api/v0/celebrity-detection"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    celebrities = config.get("celebrities", [])
    if not celebrities:
        print("  No celebrities configured in config file.")
        return []

    all_faces = []

    for celeb in celebrities:
        celeb_name = celeb["name"]
        person_id = celeb.get("person_id", "")

        resp = requests.get(
            f"{celeb_base}/faces-with-person",
            headers=headers,
            params={"person_name": celeb_name},
            verify=False, timeout=120,
        )
        if resp.status_code != 200:
            print(f"    {celeb_name}: HTTP {resp.status_code}")
            continue

        faces = resp.json().get("faces", [])

        # Filter to our dataset if specified
        if dataset_id:
            dataset_faces = [f for f in faces if f.get("dataset_id") == dataset_id]
        else:
            dataset_faces = faces

        for face in dataset_faces:
            all_faces.append({
                "face_id": face["id"],
                "image_id": face["image_id"],
                "dataset_id": face.get("dataset_id", ""),
                "confidence": face["confidence"],
                "bbox": face.get("bbox", {}),
                "celebrity_name": celeb_name,
                "person_id": person_id or face.get("person_id", ""),
            })

        print(f"    {celeb_name}: {len(dataset_faces)} faces in dataset (of {len(faces)} total)")

    return all_faces


# =============================================================================
# PHASE 2: DYNAMIC TAGS SETUP
# =============================================================================

def get_dt_groups(base_url: str, token: str) -> list:
    """List existing Dynamic Tag groups."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.get(
        f"{base_url}/api/v0/dynamic-tags/groups",
        headers=headers,
        params={"page": 1, "per_page": 50},
        verify=False, timeout=30,
    )
    if resp.status_code == 200:
        return resp.json().get("data", [])
    print(f"    Failed to get groups: {resp.status_code}")
    return []


def get_group_version(base_url: str, token: str, group_id: str):
    """Get the latest version of a group (contains tag IDs)."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.get(
        f"{base_url}/api/v0/dynamic-tags/groups/{group_id}/versions/latest",
        headers=headers,
        params={"published": "false"},
        verify=False, timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()
    print(f"    Get version failed: {resp.status_code}")
    return None


def create_dt_group(config: dict, token: str, group_name: str, tag_names: list) -> dict:
    """Create a Dynamic Tag group and add tags."""
    base_url = config["base_url"]
    dataset_id = config.get("dataset_id", "")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Step 1: Create group (without tags)
    resp = requests.post(
        f"{base_url}/api/v0/dynamic-tags/groups",
        headers=headers,
        json={"name": group_name, "shareable": False, "tags": [], "dataset_id": dataset_id},
        verify=False, timeout=30,
    )
    if resp.status_code not in (200, 201):
        # Retry with training_dataset_ids format
        resp = requests.post(
            f"{base_url}/api/v0/dynamic-tags/groups",
            headers=headers,
            json={"name": group_name, "shareable": False, "tags": [],
                  "training_dataset_ids": [dataset_id]},
            verify=False, timeout=30,
        )
    if resp.status_code not in (200, 201):
        print(f"    Create group failed: {resp.status_code} - {resp.text[:300]}")
        return None

    data = resp.json()
    group_id = data.get("id") or data.get("group_id")
    print(f"    Group created: {group_id}")

    # Step 2: Get version ID
    version = get_group_version(base_url, token, group_id)
    if not version:
        print(f"    Could not get group version")
        return data

    group_version_id = version.get("id") or version.get("group_version_id")

    # Step 3: Add tags
    print(f"    Adding {len(tag_names)} tags to group...")
    resp = requests.post(
        f"{base_url}/api/v0/dynamic-tags/groups/{group_id}/versions/{group_version_id}/tags",
        headers=headers,
        json={"names": tag_names},
        verify=False, timeout=30,
    )
    if resp.status_code in (200, 201):
        print(f"    Tags created successfully")
    else:
        print(f"    Add tags failed: {resp.status_code} - {resp.text[:300]}")

    data["version_id"] = group_version_id
    data["group_id"] = group_id
    return data


def update_tag_prompts(base_url: str, token: str, group_id: str,
                       tag_id: str, tag_version_id: str, description: str) -> bool:
    """Set text prompts for a tag."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.patch(
        f"{base_url}/api/v0/dynamic-tags/groups/{group_id}/tags/{tag_id}/versions/{tag_version_id}",
        headers=headers,
        json={
            "prompts": {
                "text": [{
                    "content": description,
                    "label": "positive",
                    "modalities": ["visual"],
                }]
            }
        },
        verify=False, timeout=30,
    )
    return resp.status_code == 200


def associate_group_with_dataset(base_url: str, token: str,
                                  dataset_id: str, group_id: str,
                                  group_version_id: str) -> bool:
    """Associate a DT group with a dataset for scoring."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.put(
        f"{base_url}/api/v0/dynamic-tags/datasets/{dataset_id}/groups",
        headers=headers,
        json={"group_and_group_version_id_pairs": [[group_id, group_version_id]]},
        verify=False, timeout=30,
    )
    if resp.status_code in (200, 201, 204):
        print(f"    Group associated with dataset")
        return True
    print(f"    Association failed: {resp.status_code} - {resp.text[:300]}")
    return False


def publish_group_version(base_url: str, token: str,
                           group_id: str, group_version_id: str) -> bool:
    """Publish a group version (triggers scoring)."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Try v3 endpoint first (preferred)
    resp = requests.post(
        f"{base_url}/api/v3/dynamic-tags/groups/{group_id}/versions/{group_version_id}/publish",
        headers=headers,
        json={"shareable": False},
        verify=False, timeout=30,
    )
    if resp.status_code in (200, 201, 202):
        print(f"    Group version published via v3 API (scoring triggered)")
        return True

    # Fall back to v0 endpoint
    resp = requests.post(
        f"{base_url}/api/v0/dynamic-tags/groups/{group_id}/versions/{group_version_id}/publish",
        headers=headers,
        json={"shareable": False},
        verify=False, timeout=30,
    )
    if resp.status_code in (200, 201, 202):
        print(f"    Group version published via v0 API (scoring triggered)")
        return True

    print(f"    Publish failed: {resp.status_code} - {resp.text[:300]}")
    return False


def setup_dynamic_tags(config: dict, token: str) -> str:
    """
    Create DT group(s), set prompts, associate with dataset, publish.
    Returns group_id of the first group (for scoring).
    """
    base_url = config["base_url"]
    dataset_id = config.get("dataset_id", "")
    dt_groups_config = config.get("dt_groups", [])

    if not dt_groups_config:
        print("  No dt_groups configured in config file.")
        return None

    # Check existing groups
    print("  Checking existing DT groups...")
    existing_groups = get_dt_groups(base_url, token)
    existing_names = {g.get("name"): g for g in existing_groups}

    first_group_id = None

    for dt_group_cfg in dt_groups_config:
        group_name = dt_group_cfg["name"]
        tag_defs = dt_group_cfg.get("tags", {})

        if group_name in existing_names:
            group_id = existing_names[group_name].get("id") or existing_names[group_name].get("group_id")
            print(f"  Group '{group_name}' already exists: {group_id}")
            if not first_group_id:
                first_group_id = group_id
            continue

        # Create group with tags
        tag_names = list(tag_defs.keys())
        print(f"\n  Creating DT group '{group_name}' with {len(tag_names)} tags...")
        result = create_dt_group(config, token, group_name, tag_names)
        if not result:
            continue

        group_id = result.get("id") or result.get("group_id")
        if not first_group_id:
            first_group_id = group_id

        # Get version to find tag IDs
        version = get_group_version(base_url, token, group_id)
        if not version:
            continue

        group_version_id = version.get("id") or version.get("group_version_id")

        # Set prompts for each tag
        tags = version.get("tags", [])
        print(f"  Setting prompts for {len(tags)} tags...")
        for tag in tags:
            tag_name = tag.get("name", "")
            tag_id = tag.get("id") or tag.get("tag_id")

            # Extract tag_version_id from nested structure
            tag_version_id = tag.get("version_id") or tag.get("tag_version_id")
            if not tag_version_id:
                # v3 API nests versions: tag -> version -> {id, name}
                tag_version = tag.get("version", {})
                if tag_version:
                    tag_version_id = tag_version.get("id")
                # Fallback: versions list
                if not tag_version_id:
                    tag_versions = tag.get("versions", [])
                    if tag_versions:
                        tag_version_id = tag_versions[0].get("id")

            if tag_name in tag_defs and tag_id and tag_version_id:
                ok = update_tag_prompts(base_url, token, group_id,
                                        tag_id, tag_version_id, tag_defs[tag_name])
                print(f"    {tag_name}: {'ok' if ok else 'FAILED'}")

        # Associate with dataset
        print("\n  Associating group with dataset...")
        associate_group_with_dataset(base_url, token, dataset_id, group_id, group_version_id)

        # Publish
        print("  Publishing group version...")
        publish_group_version(base_url, token, group_id, group_version_id)

    return first_group_id


# =============================================================================
# PHASE 3: FETCH DT SCORES
# =============================================================================

def fetch_dt_scores(config: dict, token: str, group_id: str) -> dict:
    """
    Fetch Dynamic Tag scores for the dataset.

    Uses scoring-preview endpoint which returns actual 0-1 DT scores
    (limited to ~100 keyframes per tag).

    Returns: {image_id: {tag_name: score}}
    """
    base_url = config["base_url"]
    dataset_id = config.get("dataset_id", "")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    scores_by_image = defaultdict(dict)

    # Get group version with tag IDs
    version = get_group_version(base_url, token, group_id)
    if not version:
        print("    Could not get group version for scoring")
        return {}

    tags = version.get("tags", [])
    group_version_id = version.get("id") or version.get("group_version_id")
    print(f"    Found {len(tags)} tags in group")

    for tag in tags:
        tag_name = tag.get("name", "")
        tag_id = tag.get("id") or tag.get("tag_id")

        # Extract tag_version_id
        tag_version_id = tag.get("version_id") or tag.get("tag_version_id")
        if not tag_version_id:
            tag_version = tag.get("version", {})
            if tag_version:
                tag_version_id = tag_version.get("id")
            if not tag_version_id:
                tag_versions = tag.get("versions", [])
                if tag_versions:
                    tag_version_id = tag_versions[0].get("id")

        if not (tag_id and tag_version_id):
            print(f"    {tag_name}: missing tag_id or tag_version_id, skipping")
            continue

        # scoring-preview returns actual DT scores (0-1 range)
        resp = requests.get(
            f"{base_url}/api/v0/dynamic-tags/groups/{group_id}/tags/{tag_id}"
            f"/versions/{tag_version_id}/scoring-preview/image-and-keyframe",
            headers=headers,
            params={"prompt_types": ["text", "visual"]},
            verify=False, timeout=60,
        )
        if resp.status_code == 200:
            preview = resp.json()
            items = preview.get("data", preview.get("items", []))
            for item in items:
                img_id = item.get("image_id") or item.get("asset_id")
                score = item.get("score", 0)
                if img_id:
                    scores_by_image[img_id][tag_name] = score
            print(f"    {tag_name}: {len(items)} scored items")
        else:
            print(f"    {tag_name}: scoring-preview HTTP {resp.status_code}")

            # Fallback: asset-check (returns raw cosine similarities, not 0-1 scores)
            resp2 = requests.get(
                f"{base_url}/api/v0/dynamic-tags/groups/{group_id}/tags/{tag_id}"
                f"/versions/{tag_version_id}/asset-check",
                headers=headers,
                params={"dataset_id": dataset_id},
                verify=False, timeout=60,
            )
            if resp2.status_code == 200:
                check_data = resp2.json()
                items = check_data.get("data", check_data.get("items", []))
                for item in items:
                    img_id = item.get("image_id") or item.get("asset_id")
                    # asset-check uses 'similarity' not 'score'
                    score = item.get("score", item.get("similarity", 0))
                    if img_id:
                        scores_by_image[img_id][tag_name] = score
                print(f"    {tag_name}: {len(items)} items (via asset-check fallback)")

    print(f"    Total images with scores: {len(scores_by_image)}")
    return dict(scores_by_image)


# =============================================================================
# PHASE 4: JOIN CELEBRITIES + DT SCORES
# =============================================================================

def join_celeb_and_dt(faces: list, dt_scores: dict) -> list:
    """
    Join celebrity face data with Dynamic Tag scores on image_id.

    Returns list of joined rows:
        {image_id, celebrity_name, confidence, person_id, tag_name, tag_score}
    """
    # Deduplicate faces by (image_id, celebrity_name), keeping highest confidence
    best_faces = {}
    for face in faces:
        key = (face["image_id"], face["celebrity_name"])
        if key not in best_faces or face["confidence"] > best_faces[key]["confidence"]:
            best_faces[key] = face

    print(f"    Unique (image, celebrity) pairs: {len(best_faces)}")

    # Join
    joined = []
    matched_images = set()
    for (image_id, celeb_name), face in best_faces.items():
        if image_id in dt_scores:
            matched_images.add(image_id)
            for tag_name, score in dt_scores[image_id].items():
                joined.append({
                    "image_id": image_id,
                    "celebrity_name": celeb_name,
                    "confidence": face["confidence"],
                    "person_id": face["person_id"],
                    "tag_name": tag_name,
                    "tag_score": score,
                })

    print(f"    Images with DT scores: {len(matched_images)}")
    print(f"    Joined rows (celeb x tag): {len(joined)}")
    return joined


# =============================================================================
# OUTPUT: SQL QUERIES + STATS
# =============================================================================

def generate_queries(joined: list, faces: list, celebrities: list):
    """Generate Coactive UI SQL queries for top results."""
    print("\n" + "=" * 60)
    print("COACTIVE UI SQL QUERIES")
    print("=" * 60)

    celeb_names = [c["name"] for c in celebrities]

    # 1. Top images per celebrity (by confidence)
    for celeb_name in celeb_names:
        celeb_faces = [f for f in faces if f["celebrity_name"] == celeb_name]
        celeb_faces.sort(key=lambda x: x["confidence"], reverse=True)

        seen = set()
        unique = []
        for f in celeb_faces:
            if f["image_id"] not in seen:
                seen.add(f["image_id"])
                unique.append(f)
            if len(unique) >= 10:
                break

        if unique:
            ids_str = ", ".join([f"'{f['image_id']}'" for f in unique])
            print(f"\n-- {celeb_name} top 10 (highest confidence):")
            print(f"SELECT * FROM coactive_table WHERE coactive_image_id IN ({ids_str})")

    # 2. Top images per celebrity + tag combo
    if joined:
        from itertools import groupby
        joined_sorted = sorted(joined, key=lambda x: (x["celebrity_name"], x["tag_name"]))

        for (celeb, tag), group in groupby(joined_sorted,
                                            key=lambda x: (x["celebrity_name"], x["tag_name"])):
            items = sorted(list(group), key=lambda x: (-x["tag_score"], -x["confidence"]))
            items = [i for i in items if i["confidence"] > 0.7]

            seen = set()
            unique = []
            for item in items:
                if item["image_id"] not in seen:
                    seen.add(item["image_id"])
                    unique.append(item)
                if len(unique) >= 10:
                    break

            if unique:
                ids_str = ", ".join([f"'{i['image_id']}'" for i in unique])
                top_score = unique[0]["tag_score"]
                print(f"\n-- {celeb} + {tag} (top score: {top_score:.4f}):")
                print(f"SELECT * FROM coactive_table WHERE coactive_image_id IN ({ids_str})")


def print_stats(faces: list, joined: list, celebrities: list):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    celeb_names = [c["name"] for c in celebrities]

    # Per celebrity
    celeb_counts = defaultdict(int)
    celeb_images = defaultdict(set)
    for f in faces:
        celeb_counts[f["celebrity_name"]] += 1
        celeb_images[f["celebrity_name"]].add(f["image_id"])

    print("\n  Celebrity Face Detections:")
    for celeb in celeb_names:
        count = celeb_counts.get(celeb, 0)
        images = len(celeb_images.get(celeb, set()))
        print(f"    {celeb}: {count} faces, {images} unique images")

    if joined:
        # Cross-tab: celebrity x tag
        print("\n  Celebrity x Tag Matches:")
        combo_counts = defaultdict(int)
        for j in joined:
            combo_counts[(j["celebrity_name"], j["tag_name"])] += 1

        # Group by celebrity
        for celeb in celeb_names:
            celeb_combos = {k: v for k, v in combo_counts.items() if k[0] == celeb}
            if celeb_combos:
                print(f"\n    {celeb}:")
                for (_, tag), count in sorted(celeb_combos.items(), key=lambda x: -x[1]):
                    print(f"      {tag}: {count}")

        # Avg metrics
        print("\n  Avg Metrics per Celebrity:")
        celeb_metrics = defaultdict(lambda: {"conf_sum": 0, "score_sum": 0, "n": 0})
        for j in joined:
            m = celeb_metrics[j["celebrity_name"]]
            m["conf_sum"] += j["confidence"]
            m["score_sum"] += j["tag_score"]
            m["n"] += 1

        for celeb in celeb_names:
            m = celeb_metrics.get(celeb)
            if m and m["n"] > 0:
                print(f"    {celeb}: avg_confidence={m['conf_sum']/m['n']:.3f}, "
                      f"avg_tag_score={m['score_sum']/m['n']:.3f}")


def export_results(faces: list, joined: list, output_path: str):
    """Export joined data as JSON."""
    output = {
        "faces_count": len(faces),
        "joined_count": len(joined),
        "faces": faces,
        "joined": joined,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results exported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Celebrity Detection + Dynamic Tags Join Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow:
  python3 celeb_dt_workflow.py --config config.json

  # Faces only (check detections after backfill):
  python3 celeb_dt_workflow.py --config config.json --faces-only

  # Setup DT group and tags:
  python3 celeb_dt_workflow.py --config config.json --dt-setup

  # Fetch DT scores (after scoring completes):
  python3 celeb_dt_workflow.py --config config.json --dt-scores

  # Export joined data to JSON:
  python3 celeb_dt_workflow.py --config config.json --output results.json
        """,
    )
    parser.add_argument("--config", "-c", help="Path to config JSON file")
    parser.add_argument("--faces-only", action="store_true",
                        help="Only fetch celebrity faces, skip DT workflow")
    parser.add_argument("--dt-setup", action="store_true",
                        help="Only create DT groups + tags (skip faces/scores)")
    parser.add_argument("--dt-scores", action="store_true",
                        help="Only fetch DT scores (skip enrollment/faces)")
    parser.add_argument("--dt-group-id", help="Use this DT group ID instead of creating one")
    parser.add_argument("--output", "-o", help="Export joined results to JSON file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print(f"Celebrity + Dynamic Tags Workflow — {config['client_name']}")
    print("=" * 60)
    print(f"  Dataset:      {config.get('dataset_id', 'N/A')}")
    print(f"  Celebrities:  {len(config.get('celebrities', []))}")
    print(f"  DT Groups:    {len(config.get('dt_groups', []))}")
    print("=" * 60)

    # Auth
    print("\n[Auth] Getting JWT token...")
    token = get_jwt_token(config)

    celebrities = config.get("celebrities", [])

    # ---- DT Setup Only ----
    if args.dt_setup:
        print("\n[DT Setup] Creating Dynamic Tags...")
        group_id = setup_dynamic_tags(config, token)
        if group_id:
            print(f"\n  DT group ready: {group_id}")
            print("  Wait for scoring to complete (~30 min), then run with --dt-scores")
        return

    # ---- Faces ----
    print("\n[Phase 1] Fetching celebrity face detections...")
    faces = fetch_celebrity_faces(config, token)
    print(f"  Total faces in dataset: {len(faces)}")

    if not faces:
        print("\n  No faces found. Enrollment backfill may still be in progress.")
        print("  Enroll persons first with celebrity_enrollment.py, then re-run.")
        return

    if args.faces_only:
        print_stats(faces, [], celebrities)
        generate_queries([], faces, celebrities)
        return

    # ---- DT Group ----
    group_id = args.dt_group_id

    if not group_id:
        # Check if DT groups are configured
        if config.get("dt_groups"):
            print("\n[Phase 2] Setting up Dynamic Tags...")
            group_id = setup_dynamic_tags(config, token)
        else:
            # Try to find existing group
            print("\n[Phase 2] Checking for existing DT groups...")
            existing = get_dt_groups(config["base_url"], token)
            if existing:
                group_id = existing[0].get("id") or existing[0].get("group_id")
                print(f"  Using existing group: {existing[0].get('name')} ({group_id})")

    if not group_id:
        print("\n  No DT group available. Run with --dt-setup first or provide --dt-group-id.")
        print("  Generating celeb-only queries...")
        print_stats(faces, [], celebrities)
        generate_queries([], faces, celebrities)
        return

    # ---- DT Scores ----
    print(f"\n[Phase 3] Fetching Dynamic Tag scores (group: {group_id})...")
    dt_scores = fetch_dt_scores(config, token, group_id)

    if not dt_scores:
        print("\n  No DT scores available yet.")
        print("  DT scoring may still be in progress after publishing (~30 min).")
        print("  Generating celeb-only queries for now...")
        print_stats(faces, [], celebrities)
        generate_queries([], faces, celebrities)
        return

    # ---- Join ----
    print(f"\n[Phase 4] Joining celebrities + Dynamic Tags...")
    joined = join_celeb_and_dt(faces, dt_scores)

    # Output
    print_stats(faces, joined, celebrities)
    generate_queries(joined, faces, celebrities)

    if args.output:
        export_results(faces, joined, args.output)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
