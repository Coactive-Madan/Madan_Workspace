#!/usr/bin/env python3
"""
Coactive Dynamic Tags & Concepts Analyzer

A reusable tool for analyzing Coactive Dynamic Tags and Concepts performance.
Generates comprehensive statistics, score distributions, and markdown reports.

Usage:
    python coactive_analyzer.py --token YOUR_API_TOKEN --type concepts
    python coactive_analyzer.py --token YOUR_API_TOKEN --type dynamic-tags --group-id GROUP_ID
    python coactive_analyzer.py --token YOUR_API_TOKEN --url "https://app.coactive.ai/concepts?page=1"

Author: Coactive Team
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CoactiveConfig:
    """Configuration for Coactive API access."""
    api_base: str = "https://api.coactive.ai"
    app_base: str = "https://app.coactive.ai"
    login_base: str = "https://public.app.coactive.ai"
    api_version: str = "v0"
    query_api_version: str = "v1"
    poll_interval: int = 2
    max_poll_attempts: int = 90
    query_timeout: int = 180


# =============================================================================
# API CLIENT
# =============================================================================

class CoactiveClient:
    """Client for interacting with Coactive API."""

    def __init__(self, api_token: str, config: Optional[CoactiveConfig] = None):
        self.api_token = api_token
        self.config = config or CoactiveConfig()
        self.jwt_token: Optional[str] = None
        self.session = requests.Session()

    def authenticate(self) -> bool:
        """Exchange API token for JWT access token."""
        print("🔐 Authenticating...")

        try:
            response = self.session.post(
                f"{self.config.login_base}/api/{self.config.api_version}/login",
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json"
                },
                json={"grant_type": "refresh_token"},
                timeout=30
            )

            if response.status_code == 200:
                self.jwt_token = response.json().get("access_token")
                print("✅ Authentication successful")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False

        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        token = self.jwt_token or self.api_token
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def list_concepts(self) -> List[Dict]:
        """List all concepts."""
        response = self.session.get(
            f"{self.config.api_base}/api/{self.config.api_version}/concepts",
            headers=self._get_headers(),
            params={"page": 1, "size": 100},
            timeout=30
        )

        if response.status_code == 200:
            return response.json().get("data", [])
        return []

    def list_dynamic_tag_groups(self) -> List[Dict]:
        """List all dynamic tag groups."""
        response = self.session.get(
            f"{self.config.api_base}/api/{self.config.api_version}/dynamic-tags/groups",
            headers=self._get_headers(),
            params={"page": 1, "size": 100},
            timeout=30
        )

        if response.status_code == 200:
            return response.json().get("data", [])
        return []

    def get_dynamic_tag_group(self, group_id: str) -> Optional[Dict]:
        """Get a specific dynamic tag group."""
        response = self.session.get(
            f"{self.config.api_base}/api/{self.config.api_version}/dynamic-tags/groups/{group_id}/tags",
            headers=self._get_headers(),
            params={"page": 1, "size": 100},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        return None

    def execute_query(self, query: str, dataset_id: str) -> Optional[List[Dict]]:
        """Execute a SQL query and return results."""
        # Create query
        response = self.session.post(
            f"{self.config.app_base}/api/{self.config.query_api_version}/queries",
            headers=self._get_headers(),
            json={"query": query, "dataset_id": dataset_id},
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ Query creation failed: {response.text[:200]}")
            return None

        query_id = response.json().get("queryId") or response.json().get("query_id")

        # Poll for completion
        for _ in range(self.config.max_poll_attempts):
            status_response = self.session.get(
                f"{self.config.app_base}/api/{self.config.query_api_version}/queries/{query_id}",
                headers=self._get_headers(),
                timeout=30
            )

            status = status_response.json().get("status")

            if status in ["Complete", "Completed", "Succeeded", "Success"]:
                break
            elif status in ["Failed", "Error"]:
                error = status_response.json().get("result", {}).get("error", "Unknown error")
                print(f"❌ Query failed: {error[:200]}")
                return None

            time.sleep(self.config.poll_interval)
        else:
            print("❌ Query timed out")
            return None

        # Get results
        results_response = self.session.get(
            f"{self.config.app_base}/api/{self.config.query_api_version}/queries/{query_id}/csv/url",
            headers=self._get_headers(),
            timeout=30
        )

        download_url = results_response.json().get("downloadUrl") or results_response.json().get("download_url")
        csv_response = self.session.get(download_url, timeout=60)

        reader = csv.DictReader(io.StringIO(csv_response.text))
        return list(reader)


# =============================================================================
# ANALYZERS
# =============================================================================

class ConceptAnalyzer:
    """Analyzer for Coactive Concepts."""

    def __init__(self, client: CoactiveClient):
        self.client = client

    def discover_tables(self, dataset_id: str) -> List[str]:
        """Discover concept tables in the dataset."""
        print("📊 Discovering concept tables...")

        results = self.client.execute_query("SHOW TABLES", dataset_id)
        if not results:
            return []

        concept_tables = []
        for r in results:
            table = r.get("tableName", "")
            if table.startswith("concept_"):
                concept_tables.append(table)

        print(f"   Found {len(concept_tables)} concept tables")
        return sorted(concept_tables)

    def analyze(self, dataset_id: str, tables: List[str]) -> Dict:
        """Run full analysis on concept tables."""
        print(f"\n🔍 Analyzing {len(tables)} concepts...")

        # Build union query for statistics
        union_parts = []
        for table in tables:
            concept_name = table.replace("concept_", "").replace("_", " ").title()
            union_parts.append(f"""
            SELECT
                '{concept_name}' as concept_name,
                '{table}' as table_name,
                COUNT(*) as total_images,
                ROUND(AVG(score), 4) as avg_score,
                ROUND(MIN(score), 4) as min_score,
                ROUND(MAX(score), 4) as max_score,
                ROUND(PERCENTILE(score, 0.5), 4) as median_score,
                ROUND(STDDEV(score), 4) as std_dev,
                SUM(CASE WHEN score >= 0.9 THEN 1 ELSE 0 END) as above_90,
                SUM(CASE WHEN score >= 0.8 AND score < 0.9 THEN 1 ELSE 0 END) as range_80_90,
                SUM(CASE WHEN score >= 0.7 AND score < 0.8 THEN 1 ELSE 0 END) as range_70_80,
                SUM(CASE WHEN score >= 0.5 AND score < 0.7 THEN 1 ELSE 0 END) as range_50_70,
                SUM(CASE WHEN score < 0.5 THEN 1 ELSE 0 END) as below_50
            FROM {table}
            """)

        query = " UNION ALL ".join(union_parts) + " ORDER BY avg_score DESC"
        stats = self.client.execute_query(query, dataset_id)

        # Get top images for each concept
        top_images = {}
        for table in tables:
            concept_name = table.replace("concept_", "").replace("_", " ").title()
            query = f"SELECT coactive_image_id, score FROM {table} ORDER BY score DESC LIMIT 3"
            results = self.client.execute_query(query, dataset_id)
            if results:
                top_images[concept_name] = results

        return {
            "type": "concepts",
            "statistics": stats,
            "top_images": top_images,
            "dataset_id": dataset_id,
            "table_count": len(tables)
        }


class DynamicTagAnalyzer:
    """Analyzer for Coactive Dynamic Tags."""

    def __init__(self, client: CoactiveClient):
        self.client = client

    def discover_table(self, dataset_id: str, group_name: str) -> Optional[str]:
        """Discover the dynamic tag table for a group."""
        print(f"📊 Discovering table for group: {group_name}...")

        results = self.client.execute_query("SHOW TABLES", dataset_id)
        if not results:
            return None

        # Look for table matching group name
        normalized_name = group_name.lower().replace(" ", "_").replace("-", "_")

        for r in results:
            table = r.get("tableName", "")
            if "group_" in table and normalized_name in table.lower():
                print(f"   Found table: {table}")
                return table

        # Fallback: look for any group_ table
        for r in results:
            table = r.get("tableName", "")
            if table.startswith("group_"):
                print(f"   Found table: {table}")
                return table

        return None

    def analyze(self, dataset_id: str, table_name: str) -> Dict:
        """Run full analysis on dynamic tag table."""
        print(f"\n🔍 Analyzing dynamic tags from: {table_name}...")

        # Get statistics per tag
        query = f"""
        SELECT
            tag_name,
            COUNT(DISTINCT coactive_image_id) as total_images,
            ROUND(AVG(score), 4) as avg_score,
            ROUND(MIN(score), 4) as min_score,
            ROUND(MAX(score), 4) as max_score,
            ROUND(PERCENTILE(score, 0.5), 4) as median_score,
            ROUND(STDDEV(score), 4) as std_dev,
            SUM(CASE WHEN score >= 0.9 THEN 1 ELSE 0 END) as above_90,
            SUM(CASE WHEN score >= 0.8 AND score < 0.9 THEN 1 ELSE 0 END) as range_80_90,
            SUM(CASE WHEN score >= 0.7 AND score < 0.8 THEN 1 ELSE 0 END) as range_70_80,
            SUM(CASE WHEN score >= 0.5 AND score < 0.7 THEN 1 ELSE 0 END) as range_50_70,
            SUM(CASE WHEN score < 0.5 THEN 1 ELSE 0 END) as below_50
        FROM {table_name}
        GROUP BY tag_name
        ORDER BY avg_score DESC
        """

        stats = self.client.execute_query(query, dataset_id)

        # Get top images per tag
        top_images = {}
        if stats:
            for row in stats:
                tag_name = row.get("tag_name")
                query = f"""
                SELECT coactive_image_id, score
                FROM {table_name}
                WHERE tag_name = '{tag_name}'
                ORDER BY score DESC
                LIMIT 3
                """
                results = self.client.execute_query(query, dataset_id)
                if results:
                    top_images[tag_name] = results

        return {
            "type": "dynamic_tags",
            "statistics": stats,
            "top_images": top_images,
            "dataset_id": dataset_id,
            "table_name": table_name
        }


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generates markdown reports from analysis results."""

    @staticmethod
    def generate(results: Dict, title: str = "Coactive Analysis Report") -> str:
        """Generate a markdown report from analysis results."""
        lines = []

        # Header
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Dataset ID:** `{results.get('dataset_id', 'N/A')}`")

        if results["type"] == "concepts":
            lines.append(f"**Total Concepts:** {results.get('table_count', 0)}")
        else:
            lines.append(f"**Table:** `{results.get('table_name', 'N/A')}`")

        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        stats = results.get("statistics", [])

        if not stats:
            lines.append("No statistics available.")
            return "\n".join(lines)

        # Determine name column
        name_col = "concept_name" if results["type"] == "concepts" else "tag_name"

        # Statistics table
        lines.append("## Statistics (Sorted by Avg Score)")
        lines.append("")
        lines.append("| Name | Count | Avg Score | Min | Max | Median | Std Dev |")
        lines.append("|------|------:|----------:|----:|----:|-------:|--------:|")

        for r in stats:
            name = r.get(name_col, "Unknown")
            lines.append(
                f"| {name} | {r.get('total_images', 'N/A')} | "
                f"{r.get('avg_score', 'N/A')} | {r.get('min_score', 'N/A')} | "
                f"{r.get('max_score', 'N/A')} | {r.get('median_score', 'N/A')} | "
                f"{r.get('std_dev', 'N/A')} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

        # Score distribution
        lines.append("## Score Distribution by Threshold")
        lines.append("")
        lines.append("| Name | >0.9 | 0.8-0.9 | 0.7-0.8 | 0.5-0.7 | <0.5 | Total |")
        lines.append("|------|-----:|--------:|--------:|--------:|-----:|------:|")

        for r in stats:
            name = r.get(name_col, "Unknown")
            total = (
                int(r.get("above_90", 0)) +
                int(r.get("range_80_90", 0)) +
                int(r.get("range_70_80", 0)) +
                int(r.get("range_50_70", 0)) +
                int(r.get("below_50", 0))
            )
            lines.append(
                f"| {name} | {r.get('above_90', 0)} | {r.get('range_80_90', 0)} | "
                f"{r.get('range_70_80', 0)} | {r.get('range_50_70', 0)} | "
                f"{r.get('below_50', 0)} | {total} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

        # High confidence summary
        lines.append("## High Confidence Summary (>= 0.8 threshold)")
        lines.append("")
        lines.append("| Name | Count (>=0.8) | % of Total |")
        lines.append("|------|-------------:|-----------:|")

        sorted_stats = sorted(
            stats,
            key=lambda x: int(x.get("above_90", 0)) + int(x.get("range_80_90", 0)),
            reverse=True
        )

        for r in sorted_stats:
            name = r.get(name_col, "Unknown")
            high_conf = int(r.get("above_90", 0)) + int(r.get("range_80_90", 0))
            total = int(r.get("total_images", 1))
            pct = (high_conf / total * 100) if total > 0 else 0
            lines.append(f"| {name} | {high_conf} | {pct:.1f}% |")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Top images
        top_images = results.get("top_images", {})
        if top_images:
            lines.append("## Top 3 Scoring Images")
            lines.append("")

            for name, images in sorted(top_images.items()):
                lines.append(f"### {name}")
                lines.append("")
                lines.append("| Rank | Score | Image ID |")
                lines.append("|-----:|------:|----------|")

                for i, img in enumerate(images, 1):
                    score = float(img.get("score", 0))
                    img_id = img.get("coactive_image_id", "N/A")
                    lines.append(f"| {i} | {score:.4f} | `{img_id}` |")

                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*Generated by Coactive Analyzer*")

        return "\n".join(lines)


# =============================================================================
# URL PARSER
# =============================================================================

def parse_coactive_url(url: str) -> Dict[str, Optional[str]]:
    """Parse a Coactive URL to extract resource information."""
    result = {
        "type": None,
        "group_id": None,
        "dataset_id": None
    }

    if "/concepts" in url:
        result["type"] = "concepts"
    elif "/dynamic-tag-groups/" in url:
        result["type"] = "dynamic-tags"
        match = re.search(r"/dynamic-tag-groups/([a-f0-9-]+)", url)
        if match:
            result["group_id"] = match.group(1)
    elif "/datasets/" in url:
        match = re.search(r"/datasets/([a-f0-9-]+)", url)
        if match:
            result["dataset_id"] = match.group(1)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coactive Dynamic Tags & Concepts Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all concepts
  python coactive_analyzer.py --token YOUR_TOKEN --type concepts

  # Analyze a specific dynamic tag group
  python coactive_analyzer.py --token YOUR_TOKEN --type dynamic-tags --group-id GROUP_ID

  # Analyze from URL
  python coactive_analyzer.py --token YOUR_TOKEN --url "https://app.coactive.ai/concepts"

  # Save report to file
  python coactive_analyzer.py --token YOUR_TOKEN --type concepts --output report.md
        """
    )

    parser.add_argument(
        "--token", "-t",
        required=True,
        help="Coactive API token"
    )

    parser.add_argument(
        "--type",
        choices=["concepts", "dynamic-tags"],
        help="Type of resource to analyze"
    )

    parser.add_argument(
        "--url", "-u",
        help="Coactive URL to analyze (auto-detects type)"
    )

    parser.add_argument(
        "--group-id", "-g",
        help="Dynamic tag group ID (required for dynamic-tags type)"
    )

    parser.add_argument(
        "--dataset-id", "-d",
        help="Dataset ID (optional, auto-detected if not provided)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path for markdown report"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of markdown"
    )

    args = parser.parse_args()

    # Parse URL if provided
    if args.url:
        parsed = parse_coactive_url(args.url)
        if parsed["type"]:
            args.type = parsed["type"]
        if parsed["group_id"]:
            args.group_id = parsed["group_id"]
        if parsed["dataset_id"]:
            args.dataset_id = parsed["dataset_id"]

    # Validate arguments
    if not args.type:
        print("❌ Error: Must specify --type or provide a --url")
        sys.exit(1)

    if args.type == "dynamic-tags" and not args.group_id:
        print("❌ Error: --group-id is required for dynamic-tags analysis")
        sys.exit(1)

    # Initialize client
    client = CoactiveClient(args.token)

    if not client.authenticate():
        sys.exit(1)

    # Run analysis
    results = None
    title = "Coactive Analysis Report"

    if args.type == "concepts":
        print("\n📋 Fetching concepts...")
        concepts = client.list_concepts()

        if not concepts:
            print("❌ No concepts found")
            sys.exit(1)

        print(f"   Found {len(concepts)} concepts")

        # Get dataset ID from first concept
        dataset_id = args.dataset_id or concepts[0].get("dataset_id")
        if not dataset_id:
            print("❌ Could not determine dataset ID")
            sys.exit(1)

        analyzer = ConceptAnalyzer(client)
        tables = analyzer.discover_tables(dataset_id)

        if not tables:
            print("❌ No concept tables found")
            sys.exit(1)

        results = analyzer.analyze(dataset_id, tables)
        title = "IAB Interest Concepts - Analysis Report"

    elif args.type == "dynamic-tags":
        print(f"\n📋 Fetching dynamic tag group: {args.group_id}...")

        # Get all groups to find dataset ID
        groups = client.list_dynamic_tag_groups()

        target_group = None
        for g in groups:
            if g.get("group_id") == args.group_id:
                target_group = g
                break

        if not target_group:
            print(f"❌ Group {args.group_id} not found")
            sys.exit(1)

        dataset_id = args.dataset_id or target_group.get("dataset_id")
        group_name = target_group.get("name", "Unknown")

        print(f"   Group: {group_name}")
        print(f"   Dataset: {dataset_id}")

        analyzer = DynamicTagAnalyzer(client)
        table_name = analyzer.discover_table(dataset_id, group_name)

        if not table_name:
            print("❌ Could not find dynamic tag table")
            sys.exit(1)

        results = analyzer.analyze(dataset_id, table_name)
        title = f"{group_name} - Dynamic Tags Analysis Report"

    if not results:
        print("❌ Analysis failed")
        sys.exit(1)

    # Generate output
    if args.json:
        output = json.dumps(results, indent=2)
    else:
        output = ReportGenerator.generate(results, title)

    # Write or print output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\n✅ Report saved to: {args.output}")
    else:
        print("\n" + "=" * 80)
        print(output)


if __name__ == "__main__":
    main()
