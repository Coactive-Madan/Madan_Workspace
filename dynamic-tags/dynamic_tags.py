#!/usr/bin/env python3
"""
Coactive Dynamic Tags — Prompt Generator & Pusher
===================================================
End-to-end tool that generates optimized text prompts for Dynamic Tags
and pushes them to the Coactive v3 API.

Modes:
  generate  — Generate prompts from a tags JSON file (offline, no Coactive auth needed)
  push      — Push a previously generated prompts file to Coactive API
  run       — Generate AND push in one shot (end-to-end)

Requirements:
  pip install requests openai pydantic

Usage:
  # End-to-end: generate prompts + push to Coactive
  python3 dynamic_tags.py run \\
      --token <COACTIVE_TOKEN> \\
      --group-url "https://app.coactive.ai/dynamic-tags/groups/<gid>/versions/<vid>" \\
      --publish

  # Generate only (offline)
  python3 dynamic_tags.py generate \\
      --input tags.json \\
      --output prompts.json \\
      --modality visual

  # Push only (from previously generated file)
  python3 dynamic_tags.py push \\
      --token <COACTIVE_TOKEN> \\
      --group-url "..." \\
      --prompts prompts.json \\
      --publish

  # Dry-run (preview without changes)
  python3 dynamic_tags.py run \\
      --token <COACTIVE_TOKEN> \\
      --group-url "..." \\
      --dry-run

Environment variables:
  COACTIVE_API_TOKEN  — Coactive personal token (alternative to --token)
  OPENAI_API_KEY      — OpenAI key for prompt generation
  ANTHROPIC_API_KEY   — Anthropic key (if using --provider anthropic)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    print("Error: 'requests' package required.  pip install requests")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
API_BASE = "https://api.coactive.ai"
LOGIN_URL = f"{API_BASE}/api/v0/login"
DT_PREFIX = f"{API_BASE}/api/v3/dynamic-tags"


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════
class Modality(str, Enum):
    VISUAL = "visual"
    TRANSCRIPT = "transcript"


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES  (tuned for Coactive's encoders)
# ═══════════════════════════════════════════════════════════════════════════════
VISUAL_SYSTEM_PROMPT = """You are an expert at writing optimized text prompts for CLIP-based image classification.

Your task is to generate 4-6 short, caption-like phrases that will match images containing the described concept.

Rules:
1. Each query must be ≤8 words
2. Write as if captioning an image (noun phrases, not instructions)
3. Describe what can be SEEN, not heard or read
4. Avoid filler words like "about", "related to", "showing", "depicting"

Include a mix of:
- Direct label (exact noun/action phrase)
- Synonym/alias (common alternatives)
- Close-up/detail (parts, materials, textures)
- Context/setting (typical environments)

Return ONLY a JSON array of strings. No explanations, no markdown."""

TRANSCRIPT_SYSTEM_PROMPT = """You are an expert at writing optimized text queries for semantic text retrieval using Qwen embeddings.

Your task is to generate 1-3 queries that will match transcript segments containing the described concept.

If the input is already a complete query → return it verbatim as a single query.
If the input is a fragment/keyword → generate 3 variants:
  1. Grounded Anchor: short phrase grounding concept in a situation
  2. User Question: natural search question
  3. Compact Definition: tight defining statement

Rules:
1. Each query must be ≤35 words
2. No filler: avoid "about", "related to"
3. Write as humans would speak or search

Return ONLY a JSON array of strings. No explanations, no markdown."""

USER_PROMPT_TEMPLATE = """Generate optimized search prompts for the following tag:

TAG DESCRIPTION: {tag_description}

Remember: Return ONLY a JSON array of strings."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROMPT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
class PromptGenerator:
    """Generates tag prompts using OpenAI or Anthropic."""

    def __init__(self, provider: str = "openai", model: str | None = None):
        self.provider = provider.lower()
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                print("Error: 'openai' package required.  pip install openai")
                sys.exit(1)
            self.model = model or "gpt-4o"
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                print("Error: 'anthropic' package required.  pip install anthropic")
                sys.exit(1)
            self.model = model or "claude-sonnet-4-20250514"
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(self, tag_name: str, tag_description: str, modality: Modality) -> list[str]:
        system = VISUAL_SYSTEM_PROMPT if modality == Modality.VISUAL else TRANSCRIPT_SYSTEM_PROMPT
        user = USER_PROMPT_TEMPLATE.format(tag_description=tag_description)

        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            content = resp.choices[0].message.content.strip()
        else:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            content = resp.content[0].text.strip()

        return self._parse_json_array(content)

    @staticmethod
    def _parse_json_array(content: str) -> list[str]:
        if "```" in content:
            lines, in_block = [], False
            for line in content.split("\n"):
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or line.strip().startswith("["):
                    lines.append(line)
            content = "\n".join(lines)
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON array from LLM")
        return [str(item) for item in parsed]


# ═══════════════════════════════════════════════════════════════════════════════
# COACTIVE API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════
class CoactiveClient:
    """Thin wrapper around the Coactive Dynamic Tags v3 API."""

    def __init__(self, personal_token: str):
        self.jwt: str | None = None
        self._auth(personal_token)

    def _auth(self, token: str):
        resp = requests.post(
            LOGIN_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"grant_type": "refresh_token"},
            verify=False,
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Auth failed (HTTP {resp.status_code}): {resp.text[:300]}")
        self.jwt = resp.json().get("access_token")
        if not self.jwt:
            raise RuntimeError("No access_token in login response")

    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.jwt}", "Content-Type": "application/json"}

    def get_latest_version(self, group_id: str, published: bool = False) -> dict:
        resp = requests.get(
            f"{DT_PREFIX}/groups/{group_id}/versions/latest",
            headers=self._h(),
            params={"published": str(published).lower()},
            verify=False, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def list_tags(self, group_id: str, version_id: str) -> list[dict]:
        all_tags, page = [], 1
        while True:
            resp = requests.get(
                f"{DT_PREFIX}/groups/{group_id}/versions/{version_id}/tags",
                headers=self._h(),
                params={"page": page, "per_page": 100},
                verify=False, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            all_tags.extend(data.get("data", []))
            if not data.get("meta", {}).get("has_next_page", False):
                break
            page += 1
        return all_tags

    def patch_tag(self, group_id: str, tag_id: str, tag_ver_id: str,
                  prompts: list[str], modality: str = "visual", label: str = "positive") -> dict:
        payload = {
            "prompts": {
                "text": [
                    {"content": p, "label": label, "modalities": [modality]}
                    for p in prompts
                ]
            }
        }
        resp = requests.patch(
            f"{DT_PREFIX}/groups/{group_id}/tags/{tag_id}/versions/{tag_ver_id}",
            headers=self._h(), json=payload, verify=False, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def publish(self, group_id: str, version_id: str, shareable: bool = True) -> dict:
        resp = requests.post(
            f"{DT_PREFIX}/groups/{group_id}/versions/{version_id}/publish",
            headers=self._h(),
            json={"shareable": shareable},
            verify=False, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
# URL PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def parse_url(url: str) -> tuple[str | None, str | None]:
    gm = re.search(r"/groups/([a-f0-9-]+)", url)
    vm = re.search(r"/versions/([a-f0-9-]+)", url)
    return (gm.group(1) if gm else None, vm.group(1) if vm else None)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT FILE I/O
# ═══════════════════════════════════════════════════════════════════════════════
def load_prompts_file(path: str) -> dict[str, list[str]]:
    """Load a prompts file into {tag_name: [prompt, ...]}."""
    with open(path) as f:
        data = json.load(f)

    mapping: dict[str, list[str]] = {}

    # Format A: prompt-writer output  {"results": [...]}
    if "results" in data and isinstance(data["results"], list):
        for r in data["results"]:
            name = r.get("tag_name", "")
            prompts = r.get("prompts", r.get("positive_text_prompts", []))
            if name and prompts:
                mapping[name] = prompts
        return mapping

    # Format B: single-tag  {"tag_name": ..., "positive_text_prompts": [...]}
    if "tag_name" in data:
        return {data["tag_name"]: data.get("positive_text_prompts", data.get("prompts", []))}

    # Format C: flat map  {"Tag Name": ["p1", "p2"]}
    if all(isinstance(v, list) for v in data.values()):
        return data

    raise ValueError(f"Unrecognized format in {path}")


def load_tags_file(path: str) -> list[dict]:
    """Load a tags input file (for generation)."""
    with open(path) as f:
        data = json.load(f)
    if "tags" in data:
        return data["tags"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected 'tags' key or array in {path}")


def save_prompts(results: list[dict], output_path: str, group_name: str = ""):
    """Save generated prompts to a JSON file."""
    output = {
        "group_name": group_name,
        "total_tags": len(results),
        "successful": sum(1 for r in results if r.get("prompts")),
        "failed": sum(1 for r in results if r.get("error")),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOWS
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_generate(args):
    """Generate prompts from a tags JSON file."""
    print("=" * 60)
    print("  Generate Prompts")
    print("=" * 60)

    tags = load_tags_file(args.input)
    modality = Modality(args.modality)
    print(f"  Tags: {len(tags)}  |  Modality: {modality.value}  |  Provider: {args.provider}")
    print("-" * 60)

    gen = PromptGenerator(provider=args.provider, model=args.model)
    results = []

    for i, tag in enumerate(tags, 1):
        name = tag.get("name", f"tag_{i}")
        desc = tag.get("description", name)
        print(f"[{i}/{len(tags)}] {name}...", end=" ")
        try:
            prompts = gen.generate(name, desc, modality)
            print(f"✅ {len(prompts)} prompts")
            results.append({
                "tag_name": name,
                "tag_description": desc,
                "modality": modality.value,
                "prompts": prompts,
                "error": None,
            })
        except Exception as e:
            print(f"❌ {e}")
            results.append({
                "tag_name": name,
                "tag_description": desc,
                "modality": modality.value,
                "prompts": [],
                "error": str(e),
            })

    save_prompts(results, args.output)
    ok = sum(1 for r in results if r["prompts"])
    print(f"\nDone: {ok}/{len(results)} tags generated successfully")


def cmd_push(args):
    """Push prompts to Coactive API."""
    print("=" * 60)
    print("  Push Prompts → Coactive API")
    print("=" * 60)

    # Resolve IDs
    if args.group_url:
        gid, vid = parse_url(args.group_url)
    else:
        gid, vid = args.group_id, args.version_id
    if not gid:
        print("Error: could not determine group ID"); sys.exit(1)

    modality = args.modality

    # Auth
    print("\n  Authenticating...", end=" ")
    client = CoactiveClient(args.token)
    print("✅")

    # Resolve working version
    latest = client.get_latest_version(gid, published=False)
    work_ver = latest["group_version_id"]
    print(f"  Group: {latest.get('name', gid)} | Version: {latest.get('version_name')} ({latest['status']})")

    # List tags
    tags = client.list_tags(gid, work_ver)
    tag_lookup = {}
    for t in tags:
        n = t["version"]["name"]
        tag_lookup[n.lower()] = (n, t["tag"]["id"], t["version"]["id"])
    print(f"  Found {len(tags)} tags in group")

    # Load prompts
    prompt_map = load_prompts_file(args.prompts)
    print(f"  Loaded prompts for {len(prompt_map)} tags")
    print("-" * 60)

    matched, skipped = 0, []
    for pname, prompts in prompt_map.items():
        key = pname.lower()
        if key not in tag_lookup:
            skipped.append(pname)
            continue

        real_name, tid, tvid = tag_lookup[key]
        if args.dry_run:
            print(f"  [DRY RUN] {real_name}: {len(prompts)} prompts")
            matched += 1
            continue

        print(f"  {real_name}: pushing {len(prompts)} prompts...", end=" ")
        try:
            client.patch_tag(gid, tid, tvid, prompts, modality)
            print("✅")
            matched += 1
        except Exception as e:
            print(f"❌ {e}")

    if skipped:
        print(f"\n  ⚠️  Skipped (no match): {', '.join(skipped)}")

    # Publish
    if args.publish and not args.dry_run and matched > 0:
        print("\n  Publishing...", end=" ")
        latest = client.get_latest_version(gid, published=False)
        pv = latest["group_version_id"]
        if latest["status"] in ("unpublished_changes", "draft"):
            r = client.publish(gid, pv)
            print(f"✅ {r.get('version_name')} → {r.get('status')}")
        else:
            print(f"ℹ️  Already {latest['status']}")

    print(f"\n  Result: {matched}/{len(prompt_map)} tags updated" +
          (" (DRY RUN)" if args.dry_run else ""))


def cmd_run(args):
    """End-to-end: fetch tags from API, generate prompts, push back."""
    print("=" * 60)
    print("  End-to-End: Generate + Push")
    print("=" * 60)

    # Resolve IDs
    if args.group_url:
        gid, vid = parse_url(args.group_url)
    else:
        gid, vid = args.group_id, args.version_id
    if not gid:
        print("Error: could not determine group ID"); sys.exit(1)

    modality = Modality(args.modality)

    # 1. Auth
    print("\n[1/4] Authenticating...", end=" ")
    client = CoactiveClient(args.token)
    print("✅")

    # 2. Fetch tags from group
    print("[2/4] Fetching tags from group...")
    latest = client.get_latest_version(gid, published=False)
    work_ver = latest["group_version_id"]
    group_name = latest.get("name", gid)
    print(f"  Group: {group_name} | Version: {latest.get('version_name')} ({latest['status']})")

    api_tags = client.list_tags(gid, work_ver)
    print(f"  Found {len(api_tags)} tags:")
    for t in api_tags:
        print(f"    • {t['version']['name']}")

    if not api_tags:
        print("  No tags found!"); sys.exit(1)

    # 3. Generate prompts
    print(f"\n[3/4] Generating prompts (provider={args.provider}, modality={modality.value})...")
    gen = PromptGenerator(provider=args.provider, model=args.model)

    results = []
    for i, t in enumerate(api_tags, 1):
        name = t["version"]["name"]
        tid = t["tag"]["id"]
        tvid = t["version"]["id"]
        print(f"  [{i}/{len(api_tags)}] {name}...", end=" ")

        try:
            prompts = gen.generate(name, name, modality)  # use tag name as description
            print(f"✅ {len(prompts)} prompts")
            results.append({"name": name, "tag_id": tid, "tag_ver_id": tvid, "prompts": prompts})
        except Exception as e:
            print(f"❌ {e}")
            results.append({"name": name, "tag_id": tid, "tag_ver_id": tvid, "prompts": [], "error": str(e)})

    # 4. Push prompts
    print(f"\n[4/4] Pushing prompts to API...")
    pushed = 0
    for r in results:
        if not r["prompts"]:
            continue
        if args.dry_run:
            print(f"  [DRY RUN] {r['name']}: {len(r['prompts'])} prompts")
            pushed += 1
            continue

        print(f"  {r['name']}: {len(r['prompts'])} prompts...", end=" ")
        try:
            client.patch_tag(gid, r["tag_id"], r["tag_ver_id"], r["prompts"], modality.value)
            print("✅")
            pushed += 1
        except Exception as e:
            print(f"❌ {e}")

    # Save prompts to file for reference
    output_path = args.output or f"{group_name.replace(' ', '_').lower()}_prompts.json"
    save_prompts(
        [{"tag_name": r["name"], "modality": modality.value,
          "prompts": r["prompts"], "error": r.get("error")} for r in results],
        output_path, group_name,
    )

    # Publish
    if args.publish and not args.dry_run and pushed > 0:
        print("\n  Publishing...", end=" ")
        latest = client.get_latest_version(gid, published=False)
        pv = latest["group_version_id"]
        if latest["status"] in ("unpublished_changes", "draft"):
            pub = client.publish(gid, pv)
            print(f"✅ {pub.get('version_name')} → {pub.get('status')}")
        else:
            print(f"ℹ️  Already {latest['status']}")

    # Summary
    print("\n" + "=" * 60)
    ok = sum(1 for r in results if r["prompts"])
    print(f"Done: {ok}/{len(results)} tags generated, {pushed} pushed" +
          (" (DRY RUN)" if args.dry_run else ""))
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Coactive Dynamic Tags — Generate & push optimized prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── generate ──────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Generate prompts from a tags JSON file")
    p_gen.add_argument("--input", "-i", required=True, help="Tags JSON input file")
    p_gen.add_argument("--output", "-o", default="prompts.json", help="Output file (default: prompts.json)")
    p_gen.add_argument("--modality", "-m", choices=["visual", "transcript"], default="visual")
    p_gen.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p_gen.add_argument("--model", help="Override LLM model name")

    # ── push ──────────────────────────────────────────────────────────────
    p_push = sub.add_parser("push", help="Push prompts file to Coactive API")
    p_push.add_argument("--token", "-t", default=os.environ.get("COACTIVE_API_TOKEN"))
    g1 = p_push.add_mutually_exclusive_group(required=True)
    g1.add_argument("--group-url", "-u", help="Coactive UI URL")
    g1.add_argument("--group-id", "-g", help="Group UUID")
    p_push.add_argument("--version-id", "-v", help="Version UUID (with --group-id)")
    p_push.add_argument("--prompts", "-p", required=True, help="Prompts JSON file")
    p_push.add_argument("--modality", "-m", choices=["visual", "transcript"], default="visual")
    p_push.add_argument("--publish", action="store_true", help="Auto-publish after push")
    p_push.add_argument("--dry-run", action="store_true")

    # ── run (end-to-end) ─────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Generate prompts from API tags + push back")
    p_run.add_argument("--token", "-t", default=os.environ.get("COACTIVE_API_TOKEN"))
    g2 = p_run.add_mutually_exclusive_group(required=True)
    g2.add_argument("--group-url", "-u", help="Coactive UI URL")
    g2.add_argument("--group-id", "-g", help="Group UUID")
    p_run.add_argument("--version-id", "-v", help="Version UUID (with --group-id)")
    p_run.add_argument("--modality", "-m", choices=["visual", "transcript"], default="visual")
    p_run.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p_run.add_argument("--model", help="Override LLM model name")
    p_run.add_argument("--output", "-o", help="Save prompts to file (default: auto-named)")
    p_run.add_argument("--publish", action="store_true", help="Auto-publish after push")
    p_run.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Validate token for commands that need it
    if args.command in ("push", "run") and not args.token:
        print("Error: --token required (or set COACTIVE_API_TOKEN)")
        sys.exit(1)

    {"generate": cmd_generate, "push": cmd_push, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
