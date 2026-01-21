#!/usr/bin/env python3
"""
RLM Video Narrative Metadata Enrichment
========================================
Uses Recursive Language Model (RLM) patterns to intelligently enrich large video datasets.

This approach optimizes API usage by:
1. Wave-based processing: Quick summaries first -> Clustering -> Targeted enrichment
2. Batch intelligence: RLM samples corpus to decide what enrichments to run where
3. Cross-video context sharing: Detected entities inform subsequent video processing
4. Skip unnecessary work: Not every video needs every enrichment type

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│ WAVE 1: Foundation (runs on all videos)                                     │
│   - Quick summary (cheap, fast, required for clustering)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ WAVE 2: RLM Analysis (samples corpus)                                       │
│   - Sample 10-20% of videos                                                 │
│   - Cluster by content type (movies, TV, news, etc.)                        │
│   - Determine which enrichments are worth running per cluster               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ WAVE 3: Targeted Enrichment (runs only what's needed)                       │
│   - Process by cluster with shared context                                  │
│   - Skip enrichments that don't apply (e.g., no entities on news)           │
│   - Share detected entities across related videos                           │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    # Full RLM-optimized enrichment
    python3 rlm_video_enrichment.py --config config.json

    # Analyze corpus only (no enrichment)
    python3 rlm_video_enrichment.py --config config.json --analyze-only

    # Force specific enrichments (skip RLM planning)
    python3 rlm_video_enrichment.py --config config.json --force-enrichments summary,genre,entities

License: MIT
"""

import json
import time
import argparse
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for the enrichment pipeline."""
    # API settings
    api_base_url: str = ""
    api_key: str = ""

    # LLM settings for RLM analysis (optional)
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # Processing settings
    batch_size: int = 500
    rate_limit_delay: float = 0.3
    sample_size_for_analysis: int = 20

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            api_base_url=os.environ.get("VIDEO_API_BASE_URL", ""),
            api_key=os.environ.get("VIDEO_API_KEY", ""),
            llm_api_key=os.environ.get("OPENAI_API_KEY", ""),
            llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        )


class EnrichmentType(Enum):
    """Available enrichment types."""
    SUMMARY = "summary"
    DESCRIPTION = "description"
    GENRE = "genre"
    MOOD = "mood"
    SUBJECT = "subject"
    FORMAT = "format"
    SEGMENTS = "segments"
    ENTITIES = "entities"
    KEYFRAMES = "keyframes"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VideoAsset:
    """Represents a video with its metadata and enrichment state."""
    video_id: str
    path: str
    title: str
    duration_sec: Optional[float] = None
    existing_metadata: Dict[str, Any] = field(default_factory=dict)

    # Wave 1: Foundation
    caption_summary: str = ""

    # Wave 2: Clustering
    cluster: str = ""
    content_type: str = ""

    # Wave 3: Enrichments
    enrichments: Dict[str, Any] = field(default_factory=dict)
    enrichments_to_run: List[EnrichmentType] = field(default_factory=list)


@dataclass
class ClusterPlan:
    """RLM-generated plan for a cluster of videos."""
    cluster_name: str
    video_ids: List[str]
    enrichments_to_run: List[EnrichmentType]
    reasoning: str
    shared_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusProfile:
    """RLM's understanding of the dataset."""
    total_videos: int
    content_types: List[str]
    likely_has_brands: bool
    likely_has_celebrities: bool
    genre_diversity: str  # "low", "medium", "high"
    recommended_strategy: str
    clusters: Dict[str, ClusterPlan] = field(default_factory=dict)


# =============================================================================
# Abstract API Client (implement for your video platform)
# =============================================================================

class VideoAPIClient(ABC):
    """Abstract base class for video platform API clients.

    Implement this class for your specific video platform (e.g., Coactive,
    Twelve Labs, Google Video AI, etc.)
    """

    @abstractmethod
    def get_videos(self, dataset_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetch videos from the dataset."""
        pass

    @abstractmethod
    def get_summary(self, video_id: str, intent: str = "") -> Optional[str]:
        """Get a summary/description of the video."""
        pass

    @abstractmethod
    def get_classification(self, video_id: str, classification_type: str,
                          values: List[str] = None) -> Optional[List[str]]:
        """Get classification (genre, mood, subject, format) for a video."""
        pass

    @abstractmethod
    def get_entities(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Extract entities (people, objects, brands) from video."""
        pass

    @abstractmethod
    def get_segments(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get video segments/chapters."""
        pass

    @abstractmethod
    def save_metadata(self, video_id: str, metadata: Dict[str, Any]) -> bool:
        """Save enriched metadata back to the platform."""
        pass


class MockVideoAPIClient(VideoAPIClient):
    """Mock implementation for testing/demo purposes."""

    def __init__(self, config: Config):
        self.config = config
        logger.info("Using MockVideoAPIClient - implement VideoAPIClient for your platform")

    def get_videos(self, dataset_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        logger.warning("Mock: get_videos called")
        return []

    def get_summary(self, video_id: str, intent: str = "") -> Optional[str]:
        logger.warning(f"Mock: get_summary called for {video_id}")
        return "Mock summary"

    def get_classification(self, video_id: str, classification_type: str,
                          values: List[str] = None) -> Optional[List[str]]:
        logger.warning(f"Mock: get_classification called for {video_id}")
        return ["Mock Classification"]

    def get_entities(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        logger.warning(f"Mock: get_entities called for {video_id}")
        return []

    def get_segments(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        logger.warning(f"Mock: get_segments called for {video_id}")
        return []

    def save_metadata(self, video_id: str, metadata: Dict[str, Any]) -> bool:
        logger.warning(f"Mock: save_metadata called for {video_id}")
        return True


# =============================================================================
# LLM Helper for RLM Analysis
# =============================================================================

class LLMAnalyzer:
    """Helper for RLM corpus analysis using an LLM."""

    def __init__(self, config: Config):
        self.config = config
        self.enabled = bool(config.llm_api_key)

    def analyze_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make LLM call that returns JSON for corpus analysis."""
        if not self.enabled:
            return None

        try:
            import openai
            client = openai.OpenAI(api_key=self.config.llm_api_key)

            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            return json.loads(response.choices[0].message.content)
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None


# =============================================================================
# Wave 1: Foundation - Quick Summaries
# =============================================================================

class Wave1Foundation:
    """Wave 1: Get quick summaries for all videos for clustering."""

    def __init__(self, api_client: VideoAPIClient, config: Config):
        self.api_client = api_client
        self.config = config

    def process_batch(self, assets: List[VideoAsset]) -> List[VideoAsset]:
        """Process all videos in Wave 1."""
        logger.info(f"WAVE 1: Foundation - Getting quick summaries for {len(assets)} videos")

        for i, asset in enumerate(assets, 1):
            summary = self.api_client.get_summary(
                asset.video_id,
                intent="In 2-3 sentences, describe what type of content this is."
            )
            asset.caption_summary = summary or ""

            status = "OK" if asset.caption_summary else "WARN"
            logger.info(f"  [{i}/{len(assets)}] {status} - {asset.title[:40]}")

            time.sleep(self.config.rate_limit_delay)

        return assets


# =============================================================================
# Wave 2: RLM Corpus Analysis & Clustering
# =============================================================================

class Wave2Analysis:
    """Wave 2: RLM analyzes corpus to plan enrichments."""

    def __init__(self, llm_analyzer: LLMAnalyzer, config: Config):
        self.llm = llm_analyzer
        self.config = config

    def analyze_corpus(self, assets: List[VideoAsset]) -> CorpusProfile:
        """RLM samples dataset to understand composition."""
        logger.info("WAVE 2: Corpus Analysis")

        sample_size = min(self.config.sample_size_for_analysis, len(assets))
        logger.info(f"  Sampling {sample_size} of {len(assets)} videos...")

        sample = self._get_diverse_sample(assets, sample_size)
        sample_data = [
            {"video_id": a.video_id, "title": a.title, "summary": a.caption_summary[:200]}
            for a in sample
        ]

        # Try RLM analysis
        analysis = self._rlm_analyze_corpus(sample_data, len(assets))

        if analysis:
            profile = self._parse_rlm_analysis(analysis, assets)
        else:
            logger.info("  Using heuristic clustering (set LLM API key for RLM analysis)")
            profile = self._heuristic_clustering(assets)

        return profile

    def _get_diverse_sample(self, assets: List[VideoAsset], n: int) -> List[VideoAsset]:
        """Get diverse sample across the dataset."""
        if len(assets) <= n:
            return assets
        step = max(1, len(assets) // n)
        return [assets[i] for i in range(0, len(assets), step)][:n]

    def _rlm_analyze_corpus(self, sample_data: List[Dict], total_count: int) -> Optional[Dict]:
        """Use RLM to analyze corpus."""
        prompt = f"""Analyze this video dataset and create an enrichment plan.

SAMPLE ({len(sample_data)} of {total_count} videos):
{json.dumps(sample_data, indent=2)}

AVAILABLE ENRICHMENTS:
- summary: Full narrative summary (always useful)
- description: Detailed description (always useful)
- genre: Genre classification (useful for mixed libraries)
- mood: Emotional tone (useful for entertainment)
- subject: Topic classification (always useful)
- format: Content format (useful for varied formats)
- entities: People/celebrity extraction (useful if people appear)
- segments: Scene detection (useful for long-form)

OUTPUT JSON:
{{
  "content_types": ["list of content types found"],
  "likely_has_brands": true/false,
  "likely_has_celebrities": true/false,
  "genre_diversity": "low/medium/high",
  "clusters": {{
    "cluster_name": {{
      "description": "what this cluster contains",
      "video_patterns": ["title/content patterns to match"],
      "enrichments": ["summary", "genre", etc],
      "reasoning": "why these enrichments"
    }}
  }},
  "recommended_strategy": "overall approach description"
}}"""

        return self.llm.analyze_json(prompt)

    def _parse_rlm_analysis(self, analysis: Dict, assets: List[VideoAsset]) -> CorpusProfile:
        """Parse RLM analysis into CorpusProfile."""
        profile = CorpusProfile(
            total_videos=len(assets),
            content_types=analysis.get("content_types", []),
            likely_has_brands=analysis.get("likely_has_brands", False),
            likely_has_celebrities=analysis.get("likely_has_celebrities", False),
            genre_diversity=analysis.get("genre_diversity", "medium"),
            recommended_strategy=analysis.get("recommended_strategy", "")
        )

        clusters_data = analysis.get("clusters", {})

        for cluster_name, cluster_info in clusters_data.items():
            patterns = cluster_info.get("video_patterns", [])
            try:
                enrichments = [EnrichmentType(e) for e in cluster_info.get("enrichments", ["summary"])]
            except ValueError:
                enrichments = [EnrichmentType.SUMMARY, EnrichmentType.DESCRIPTION]

            matching_ids = self._match_videos_to_cluster(assets, patterns)

            profile.clusters[cluster_name] = ClusterPlan(
                cluster_name=cluster_name,
                video_ids=matching_ids,
                enrichments_to_run=enrichments,
                reasoning=cluster_info.get("reasoning", "")
            )

        # Assign unmatched videos to "other" cluster
        assigned = set()
        for plan in profile.clusters.values():
            assigned.update(plan.video_ids)

        unassigned = [a.video_id for a in assets if a.video_id not in assigned]
        if unassigned:
            profile.clusters["other"] = ClusterPlan(
                cluster_name="other",
                video_ids=unassigned,
                enrichments_to_run=[EnrichmentType.SUMMARY, EnrichmentType.DESCRIPTION, EnrichmentType.GENRE],
                reasoning="Default enrichments for unmatched videos"
            )

        self._log_analysis(profile)
        return profile

    def _match_videos_to_cluster(self, assets: List[VideoAsset], patterns: List[str]) -> List[str]:
        """Match videos to cluster by patterns."""
        matching = []
        patterns_lower = [p.lower() for p in patterns]

        for asset in assets:
            text = f"{asset.title} {asset.caption_summary}".lower()
            if any(pattern in text for pattern in patterns_lower):
                matching.append(asset.video_id)

        return matching

    def _heuristic_clustering(self, assets: List[VideoAsset]) -> CorpusProfile:
        """Fallback heuristic clustering when no LLM available."""
        profile = CorpusProfile(
            total_videos=len(assets),
            content_types=["mixed"],
            likely_has_brands=True,
            likely_has_celebrities=True,
            genre_diversity="medium",
            recommended_strategy="Standard enrichments with keyword-based clustering"
        )

        # Simple keyword-based clustering
        cluster_definitions = {
            "news": (
                ["news", "report", "breaking", "headline"],
                [EnrichmentType.SUMMARY, EnrichmentType.DESCRIPTION, EnrichmentType.SUBJECT]
            ),
            "entertainment": (
                ["movie", "film", "show", "episode", "series"],
                [EnrichmentType.SUMMARY, EnrichmentType.DESCRIPTION, EnrichmentType.GENRE,
                 EnrichmentType.MOOD, EnrichmentType.ENTITIES]
            ),
            "music": (
                ["music", "song", "performance", "concert", "live"],
                [EnrichmentType.SUMMARY, EnrichmentType.GENRE, EnrichmentType.ENTITIES]
            ),
            "sports": (
                ["game", "match", "tournament", "championship", "sports"],
                [EnrichmentType.SUMMARY, EnrichmentType.DESCRIPTION, EnrichmentType.ENTITIES]
            ),
        }

        assigned = set()
        for cluster_name, (keywords, enrichments) in cluster_definitions.items():
            matching_ids = []
            for asset in assets:
                if asset.video_id in assigned:
                    continue
                text = f"{asset.title} {asset.caption_summary}".lower()
                if any(kw in text for kw in keywords):
                    matching_ids.append(asset.video_id)
                    assigned.add(asset.video_id)

            if matching_ids:
                profile.clusters[cluster_name] = ClusterPlan(
                    cluster_name=cluster_name,
                    video_ids=matching_ids,
                    enrichments_to_run=enrichments,
                    reasoning=f"Matched keywords: {keywords}"
                )

        # Default cluster for unassigned
        unassigned = [a.video_id for a in assets if a.video_id not in assigned]
        if unassigned:
            profile.clusters["other"] = ClusterPlan(
                cluster_name="other",
                video_ids=unassigned,
                enrichments_to_run=[EnrichmentType.SUMMARY, EnrichmentType.DESCRIPTION,
                                   EnrichmentType.GENRE, EnrichmentType.MOOD],
                reasoning="Default enrichments"
            )

        self._log_analysis(profile)
        return profile

    def _log_analysis(self, profile: CorpusProfile):
        """Log analysis summary."""
        logger.info(f"  Analysis Results:")
        logger.info(f"    Content types: {', '.join(profile.content_types)}")
        logger.info(f"    Genre diversity: {profile.genre_diversity}")
        logger.info(f"    Has celebrities: {profile.likely_has_celebrities}")
        logger.info(f"  Clusters:")
        for name, plan in profile.clusters.items():
            enrichment_names = [e.value for e in plan.enrichments_to_run]
            logger.info(f"    - {name}: {len(plan.video_ids)} videos -> {enrichment_names}")


# =============================================================================
# Wave 3: Targeted Enrichment
# =============================================================================

class Wave3Enrichment:
    """Wave 3: Run targeted enrichments based on cluster analysis."""

    def __init__(self, api_client: VideoAPIClient, config: Config):
        self.api_client = api_client
        self.config = config

    def process_cluster(self, cluster: ClusterPlan, assets_by_id: Dict[str, VideoAsset]) -> Dict[str, Dict]:
        """Process all videos in a cluster with shared context."""
        logger.info(f"  Cluster: {cluster.cluster_name}")
        logger.info(f"    Videos: {len(cluster.video_ids)}")
        logger.info(f"    Enrichments: {[e.value for e in cluster.enrichments_to_run]}")

        shared_context = {"detected_entities": set()}
        results = {}

        for i, video_id in enumerate(cluster.video_ids, 1):
            asset = assets_by_id.get(video_id)
            if not asset:
                continue

            logger.info(f"    [{i}/{len(cluster.video_ids)}] {asset.title[:35]}...")

            enrichments = self._enrich_video(asset, cluster.enrichments_to_run, shared_context)

            # Update shared context with detected entities
            if 'entities_raw' in enrichments:
                for entity in enrichments.get('entities_raw', []):
                    if isinstance(entity, dict) and 'name' in entity:
                        shared_context["detected_entities"].add(entity['name'])

            results[video_id] = enrichments
            time.sleep(self.config.rate_limit_delay)

        return results

    def _enrich_video(self, asset: VideoAsset, enrichments: List[EnrichmentType],
                     shared_context: Dict) -> Dict[str, Any]:
        """Run specified enrichments on a video."""
        results = {}

        for enrichment in enrichments:
            result = self._run_enrichment(asset, enrichment, shared_context)
            if result:
                results.update(result)

        if results:
            results['enrichment_source'] = 'RLM Pipeline'
            results['enrichment_timestamp'] = datetime.now(timezone.utc).isoformat()
            results['enrichment_cluster'] = asset.cluster or 'default'

        return results

    def _run_enrichment(self, asset: VideoAsset, enrichment: EnrichmentType,
                       shared_context: Dict) -> Optional[Dict[str, Any]]:
        """Run a single enrichment type."""

        if enrichment == EnrichmentType.SUMMARY:
            summary = self.api_client.get_summary(
                asset.video_id,
                intent="Comprehensive summary of content, plot, key scenes, and themes"
            )
            if summary:
                return {'narrative_summary': summary}

        elif enrichment == EnrichmentType.DESCRIPTION:
            # Use summary API with description intent
            desc = self.api_client.get_summary(
                asset.video_id,
                intent="Detailed description of the video content"
            )
            if desc:
                return {'narrative_description': desc}

        elif enrichment == EnrichmentType.GENRE:
            genres = self.api_client.get_classification(asset.video_id, 'genre')
            if genres:
                return {'narrative_genre': ', '.join(genres)}

        elif enrichment == EnrichmentType.MOOD:
            moods = self.api_client.get_classification(asset.video_id, 'mood')
            if moods:
                return {'narrative_mood': ', '.join(moods)}

        elif enrichment == EnrichmentType.SUBJECT:
            subjects = self.api_client.get_classification(asset.video_id, 'subject')
            if subjects:
                return {'narrative_subject': ', '.join(subjects)}

        elif enrichment == EnrichmentType.FORMAT:
            formats = self.api_client.get_classification(asset.video_id, 'format')
            if formats:
                return {'narrative_format': ', '.join(formats)}

        elif enrichment == EnrichmentType.ENTITIES:
            entities = self.api_client.get_entities(asset.video_id)
            if entities:
                people = [e.get('name', '') for e in entities if e.get('type') == 'person']
                return {
                    'entities': json.dumps(entities),
                    'people': ', '.join(people),
                    'entity_count': str(len(entities)),
                    'entities_raw': entities
                }

        elif enrichment == EnrichmentType.SEGMENTS:
            segments = self.api_client.get_segments(asset.video_id)
            if segments:
                chapters = []
                for seg in segments:
                    start = seg.get('start_time', 0)
                    end = seg.get('end_time', 0)
                    desc = seg.get('description', '')[:100]
                    chapters.append(f"[{int(start//60)}:{int(start%60):02d}] {desc}")

                return {
                    'segments': json.dumps(segments),
                    'chapters': '\n'.join(chapters),
                    'segment_count': str(len(segments))
                }

        elif enrichment == EnrichmentType.KEYFRAMES:
            # Platform-specific keyframe extraction
            return {'keyframes_requested': 'true'}

        return None


# =============================================================================
# Main Pipeline
# =============================================================================

class RLMEnrichmentPipeline:
    """Main RLM-optimized enrichment pipeline."""

    def __init__(self, api_client: VideoAPIClient, config: Config):
        self.api_client = api_client
        self.config = config

        self.llm = LLMAnalyzer(config)
        self.wave1 = Wave1Foundation(api_client, config)
        self.wave2 = Wave2Analysis(self.llm, config)
        self.wave3 = Wave3Enrichment(api_client, config)

        self.assets: Dict[str, VideoAsset] = {}
        self.profile: Optional[CorpusProfile] = None

    def load_dataset(self, dataset_id: str, limit: int = 500) -> List[VideoAsset]:
        """Load videos from dataset."""
        logger.info(f"Loading videos from dataset {dataset_id}...")

        videos = self.api_client.get_videos(dataset_id, limit)

        assets = []
        for video in videos:
            video_id = video.get('id', video.get('video_id', ''))
            path = video.get('path', '')
            title = video.get('title', path.split('/')[-1] if path else video_id)

            asset = VideoAsset(
                video_id=video_id,
                path=path,
                title=title[:60],
                existing_metadata=video.get('metadata', {})
            )
            assets.append(asset)
            self.assets[video_id] = asset

        logger.info(f"  Loaded {len(assets)} videos")
        return assets

    def run(self,
            dataset_id: str,
            limit: int = 500,
            skip_wave1: bool = False,
            skip_wave2: bool = False,
            analyze_only: bool = False,
            force_enrichments: List[EnrichmentType] = None,
            save_results: bool = True) -> Dict[str, Any]:
        """Run the full RLM pipeline."""

        # Load dataset
        assets = self.load_dataset(dataset_id, limit)
        if not assets:
            logger.error("No videos found in dataset")
            return {"error": "No videos found"}

        start_time = time.time()

        # Wave 1: Foundation
        if not skip_wave1:
            assets = self.wave1.process_batch(assets)

        # Wave 2: Analysis
        if not skip_wave2 and not force_enrichments:
            self.profile = self.wave2.analyze_corpus(assets)
        elif force_enrichments:
            # Create forced profile
            self.profile = CorpusProfile(
                total_videos=len(assets),
                content_types=["forced"],
                likely_has_brands=True,
                likely_has_celebrities=True,
                genre_diversity="unknown",
                recommended_strategy="Forced enrichments mode"
            )
            self.profile.clusters["all"] = ClusterPlan(
                cluster_name="all",
                video_ids=[a.video_id for a in assets],
                enrichments_to_run=force_enrichments,
                reasoning="Forced by user"
            )

        if analyze_only:
            logger.info("Analysis complete (--analyze-only)")
            return {
                "profile": self.profile,
                "elapsed_seconds": time.time() - start_time
            }

        # Wave 3: Targeted Enrichment
        logger.info("WAVE 3: Targeted Enrichment")

        all_results = {}
        total_success = 0
        total_failed = 0

        for cluster_name, cluster in self.profile.clusters.items():
            results = self.wave3.process_cluster(cluster, self.assets)

            if save_results:
                for video_id, enrichments in results.items():
                    # Remove internal fields before saving
                    save_data = {k: v for k, v in enrichments.items() if not k.endswith('_raw')}
                    if save_data:
                        if self.api_client.save_metadata(video_id, save_data):
                            total_success += 1
                        else:
                            total_failed += 1

            all_results[cluster_name] = results

        elapsed = time.time() - start_time

        # Summary
        logger.info("=" * 60)
        logger.info("RLM ENRICHMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"Saved: {total_success}")
        logger.info(f"Failed: {total_failed}")
        logger.info(f"By cluster:")
        for name, results in all_results.items():
            logger.info(f"  - {name}: {len(results)} videos")

        return {
            "profile": self.profile,
            "results": all_results,
            "success_count": total_success,
            "failed_count": total_failed,
            "elapsed_seconds": elapsed
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RLM-Optimized Video Narrative Metadata Enrichment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python rlm_video_enrichment.py --config config.json --dataset-id abc123

  # Analyze only (no enrichment)
  python rlm_video_enrichment.py --config config.json --dataset-id abc123 --analyze-only

  # Force specific enrichments
  python rlm_video_enrichment.py --config config.json --dataset-id abc123 --force-enrichments summary,genre,entities

  # Use environment variables
  export VIDEO_API_BASE_URL="https://api.example.com"
  export VIDEO_API_KEY="your-key"
  export OPENAI_API_KEY="your-openai-key"  # Optional, for RLM analysis
  python rlm_video_enrichment.py --dataset-id abc123
        """
    )

    parser.add_argument('--dataset-id', '-d', required=True, help='Dataset ID to process')
    parser.add_argument('--config', '-c', help='Path to config JSON file')
    parser.add_argument('--limit', '-l', type=int, default=500, help='Max videos to process (default: 500)')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze corpus, no enrichment')
    parser.add_argument('--skip-wave1', action='store_true', help='Skip Wave 1 (summaries)')
    parser.add_argument('--skip-wave2', action='store_true', help='Skip Wave 2 (analysis)')
    parser.add_argument('--force-enrichments', help='Comma-separated enrichments: summary,genre,entities')
    parser.add_argument('--no-save', action='store_true', help='Dry run - do not save results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config.from_env()

    logger.info("=" * 60)
    logger.info("RLM VIDEO NARRATIVE ENRICHMENT")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse force enrichments
    force_enrichments = None
    if args.force_enrichments:
        force_enrichments = []
        for e in args.force_enrichments.split(','):
            try:
                force_enrichments.append(EnrichmentType(e.strip()))
            except ValueError:
                logger.warning(f"Unknown enrichment type: {e}")

    # Create API client - replace with your implementation
    api_client = MockVideoAPIClient(config)

    # Run pipeline
    pipeline = RLMEnrichmentPipeline(api_client, config)

    result = pipeline.run(
        dataset_id=args.dataset_id,
        limit=args.limit,
        skip_wave1=args.skip_wave1,
        skip_wave2=args.skip_wave2 or bool(force_enrichments),
        analyze_only=args.analyze_only,
        force_enrichments=force_enrichments,
        save_results=not args.no_save
    )

    return result


if __name__ == "__main__":
    main()
