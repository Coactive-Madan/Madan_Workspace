#!/usr/bin/env python3
"""
Coactive S3 Bucket Ingestion Script (Reusable)
Ingests assets from an S3 bucket into a Coactive dataset.

Usage:
  python3 coactive_ingest.py --dataset-id <ID> --s3-path <S3_PATH>
  python3 coactive_ingest.py --dataset-id <ID> --s3-path <S3_PATH> --max-depth 5
  python3 coactive_ingest.py --dataset-id <ID> --s3-path <S3_PATH> --token <TOKEN>

Environment variable:
  COACTIVE_API_TOKEN  - Personal API token (avoids passing on CLI)
"""

import argparse
import json
import os
import sys

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE_URL = "https://api.coactive.ai"
LOGIN_URL = f"{API_BASE_URL}/api/v0/login"
INGEST_URL = f"{API_BASE_URL}/api/v1/ingestion/buckets"


def exchange_token_for_jwt(personal_token: str) -> str:
    """Exchange personal token for JWT access token."""
    print(f"Exchanging personal token for JWT...")

    headers = {
        "Authorization": f"Bearer {personal_token}",
        "Content-Type": "application/json",
    }
    payload = {"grant_type": "refresh_token"}

    response = requests.post(LOGIN_URL, json=payload, headers=headers, verify=False, timeout=60)

    if response.status_code == 200:
        token_data = response.json()
        jwt_token = token_data.get("access_token") or token_data.get("token") or token_data.get("jwt")
        if jwt_token:
            print(f"  ✅ Got JWT token: {jwt_token[:40]}...")
            return jwt_token
        raise ValueError(f"Could not find JWT in response: {token_data}")

    raise ValueError(f"Failed to exchange token (HTTP {response.status_code}): {response.text[:300]}")


def create_ingest_job(jwt_token: str, dataset_id: str, s3_path: str, max_depth: int) -> dict:
    """Create an ingest job from S3 bucket."""
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "dataset_id": dataset_id,
        "path": s3_path,
        "max_depth": max_depth,
    }

    print(f"\nCreating ingest job...")
    print(f"  Dataset ID:  {dataset_id}")
    print(f"  S3 Path:     {s3_path}")
    print(f"  Max Depth:   {max_depth}")

    response = requests.post(INGEST_URL, json=payload, headers=headers, verify=False)

    if response.status_code == 202:
        result = response.json()
        print(f"\n  ✅ Ingest job created!")
        print(f"  Job ID:  {result.get('id')}")
        print(f"  Status:  {result.get('status')}")
        return result

    print(f"\n  ❌ Failed (HTTP {response.status_code})")
    print(f"  Response: {response.text}")
    response.raise_for_status()


def main():
    parser = argparse.ArgumentParser(description="Ingest S3 assets into a Coactive dataset")
    parser.add_argument("--dataset-id", required=True, help="Coactive dataset ID (UUID)")
    parser.add_argument("--s3-path", required=True, help="S3 path (e.g., s3://bucket/prefix/)")
    parser.add_argument("--max-depth", type=int, default=10, help="Max directory depth to crawl (default: 10)")
    parser.add_argument("--token", default=None, help="Coactive API token (or set COACTIVE_API_TOKEN env var)")

    args = parser.parse_args()

    token = args.token or os.environ.get("COACTIVE_API_TOKEN")
    if not token:
        print("Error: Provide --token or set COACTIVE_API_TOKEN environment variable")
        sys.exit(1)

    print("=" * 60)
    print("  Coactive S3 Bucket Ingestion")
    print("=" * 60)

    jwt_token = exchange_token_for_jwt(token)
    result = create_ingest_job(jwt_token, args.dataset_id, args.s3_path, args.max_depth)

    print("\n" + "=" * 60)
    print("Ingest Job Details:")
    print(json.dumps(result, indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    main()
