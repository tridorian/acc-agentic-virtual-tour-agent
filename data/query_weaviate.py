#!/usr/bin/env python3
"""Query Weaviate for website and YouTube timestamp retrieval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_EMBED_MODEL = "gemini-embedding-2-preview"
DEFAULT_CAPTION_MODEL = "gemini-2.5-flash"

# Keywords that suggest the user wants to see the virtual tour video.
TOUR_HINT_KEYWORDS = frozenset({"tour", "demo", "show", "watch", "video", "playback", "footage"})

# Boost applied to YouTube frame scores when the query has tour intent.
_TOUR_INTENT_SCORE_BOOST = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Star Learners Weaviate index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-type", choices=["youtube", "website"], default=None)
    parser.add_argument(
        "--collection",
        default=os.getenv("WEAVIATE_COLLECTION", "StarLearnersKB"),
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def to_youtube_deeplink(video_id: Optional[str], timestamp_sec: Optional[int]) -> Optional[str]:
    if not video_id or timestamp_sec is None:
        return None
    return f"https://www.youtube.com/watch?v={video_id}&t={timestamp_sec}s"


class GeminiQuery:
    def __init__(self) -> None:
        gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        if not gcp_project:
            raise RuntimeError("Missing GCP_PROJECT / GOOGLE_CLOUD_PROJECT env var")
        self.client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
        self.embed_model = os.getenv("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.caption_model = os.getenv("GEMINI_CAPTION_MODEL", DEFAULT_CAPTION_MODEL)

    @staticmethod
    def _first_vector(response: Any) -> List[float]:
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise ValueError("No embeddings in response")
        first = embeddings[0]
        values = getattr(first, "values", None)
        if not values:
            raise ValueError("Embedding vector missing values")
        return list(values)

    def embed_query(self, query: str) -> List[float]:
        # gemini-embedding-2-preview returns one embedding per call.
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=[query],
        )
        return self._first_vector(response)

    def build_visual_bridge_query(self, query: str) -> str:
        """Rewrite a text query into a visual scene description for better frame matching."""
        response = self.client.models.generate_content(
            model=self.caption_model,
            contents=[
                (
                    "Rewrite this user query into a short visual scene description suitable "
                    "for matching video frames: "
                    f"{query}"
                )
            ],
        )
        if getattr(response, "text", None):
            return response.text.strip()
        return query

    def embed_visual_query(self, query: str) -> Optional[List[float]]:
        """Build a visual bridge query and embed it for frame search."""
        try:
            bridge = self.build_visual_bridge_query(query)
            response = self.client.models.embed_content(
                model=self.embed_model,
                contents=[bridge],
            )
            return self._first_vector(response)
        except Exception:
            return None


def weaviate_client():
    import weaviate
    from weaviate.connect import ConnectionParams

    endpoint = require_env("WEAVIATE_ENDPOINT")
    api_key = os.getenv("WEAVIATE_API_KEY")
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    parsed = urlparse(endpoint)
    http_host = parsed.hostname or "localhost"
    http_port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    secure = parsed.scheme == "https"

    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None

    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host=http_host,
            http_port=http_port,
            http_secure=secure,
            grpc_host=http_host,
            grpc_port=grpc_port,
            grpc_secure=secure,
        ),
        auth_client_secret=auth,
    )
    client.connect()
    return client


def search_collection(
    client,
    collection_name: str,
    vector: List[float],
    limit: int,
    source_type_filter: Optional[str] = None,
) -> List[Any]:
    from weaviate.classes.query import Filter, MetadataQuery

    collection = client.collections.get(collection_name)
    filters = None
    if source_type_filter:
        filters = Filter.by_property("source_type").equal(source_type_filter)
    result = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        filters=filters,
        return_metadata=MetadataQuery(distance=True),
    )
    return result.objects


def has_tour_intent(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in TOUR_HINT_KEYWORDS)


def format_result(obj: Any, source: str) -> Dict[str, Any]:
    props = obj.properties
    content = props.get("content") or ""
    video_id = props.get("video_id")
    timestamp_sec = props.get("timestamp_sec")
    distance = obj.metadata.distance
    score = (1.0 - distance) if distance is not None else 0.0

    return {
        "score": float(score),
        "source_type": source,
        "content_preview": content[:240],
        "url": props.get("source_url"),
        "video_id": video_id,
        "timestamp_sec": timestamp_sec,
        "timestamp_hms": props.get("timestamp_hms"),
        "youtube_deeplink": to_youtube_deeplink(video_id, timestamp_sec),
    }


def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")
    args = parse_args()

    gemini = GeminiQuery()
    client = weaviate_client()
    results: List[Dict[str, Any]] = []

    try:
        if args.source_type in (None, "website"):
            text_vec = gemini.embed_query(args.query)
            text_hits = search_collection(
                client=client,
                collection_name=args.collection,
                vector=text_vec,
                limit=args.top_k,
                source_type_filter="website",
            )
            results.extend(format_result(hit, "website") for hit in text_hits)

        if args.source_type in (None, "youtube"):
            visual_vec = gemini.embed_visual_query(args.query)
            if visual_vec is not None:
                frame_hits = search_collection(
                    client=client,
                    collection_name=args.collection,
                    vector=visual_vec,
                    limit=args.top_k,
                    source_type_filter="youtube_frame",
                )
                results.extend(format_result(hit, "youtube_frame") for hit in frame_hits)
    finally:
        client.close()

    # Prioritize timestamped YouTube hits when intent suggests demo/tour browsing.
    if has_tour_intent(args.query):
        for row in results:
            if row["source_type"] == "youtube_frame":
                row["score"] += _TOUR_INTENT_SCORE_BOOST

    results.sort(key=lambda x: x["score"], reverse=True)
    output = {
        "query": args.query,
        "results": results[: args.top_k],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
