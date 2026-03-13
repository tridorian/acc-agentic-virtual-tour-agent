"""Weaviate retrieval tool for the Star Learners agent.

Provides both a structured search function (used by the API endpoint)
and a string-returning function tool (used by the ADK agent).

All embeddings use gemini-embedding-2-preview via Vertex AI (GCP_PROJECT env var required).
Searches the unified StarLearnersKB collection and splits results by source_type.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy singletons to avoid re-initializing on every call
# ---------------------------------------------------------------------------
_genai_client = None
_weaviate_client = None

_genai_lock = threading.Lock()
_weaviate_lock = threading.Lock()

# Named constants — avoids magic numbers in result truncation
_TEXT_CONTENT_MAX_CHARS = 600   # Max chars returned per text result
_VIDEO_CONTENT_MAX_CHARS = 300  # Max chars returned per video frame caption

_EMBED_LOCATION = "us-central1"
_EMBED_MODEL = "gemini-embedding-2-preview"

# Ensure environment variables are available regardless of process cwd.
_APP_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_APP_ENV_PATH)


def _get_genai_client():
    """Return a cached Vertex AI genai client for all embeddings."""
    global _genai_client
    if _genai_client is None:
        with _genai_lock:
            if _genai_client is None:
                from google import genai
                gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
                if not gcp_project:
                    raise RuntimeError("Missing GCP_PROJECT / GOOGLE_CLOUD_PROJECT env var")
                _genai_client = genai.Client(
                    vertexai=True,
                    project=gcp_project,
                    location=_EMBED_LOCATION,
                )
    return _genai_client


def _get_weaviate_client():
    """Return a lazy singleton Weaviate v4 client."""
    global _weaviate_client
    if _weaviate_client is None:
        with _weaviate_lock:
            if _weaviate_client is None:
                import weaviate
                from weaviate.connect import ConnectionParams

                endpoint = os.getenv("WEAVIATE_ENDPOINT", "http://localhost:8080")
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
                logger.info("Weaviate client connected to %s", endpoint)
                _weaviate_client = client
    return _weaviate_client


def close_weaviate_client() -> None:
    """Close the Weaviate singleton client. Call from FastAPI lifespan on shutdown."""
    global _weaviate_client
    with _weaviate_lock:
        if _weaviate_client is not None:
            _weaviate_client.close()
            _weaviate_client = None
            logger.info("Weaviate client closed")


def search_weaviate(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search Weaviate for both website text and YouTube video frames.

    Embeds the query with gemini-embedding-2-preview, then runs two filtered
    near_vector queries against the unified StarLearnersKB collection — one for
    website chunks and one for youtube_frame objects.

    Returns a structured dict suitable for both the REST endpoint and the agent tool.
    """
    from google.genai import types
    from weaviate.classes.query import Filter, MetadataQuery

    collection_name = os.getenv("WEAVIATE_COLLECTION", "StarLearnersKB")
    embed_model = os.getenv("GEMINI_EMBED_MODEL", _EMBED_MODEL)
    logger.info("Weaviate search | query=%r", query[:80])

    genai_client = _get_genai_client()
    weaviate_client = _get_weaviate_client()

    # Embed query once — gemini-embedding-2-preview handles both text and image search
    embed_response = genai_client.models.embed_content(
        model=embed_model,
        contents=[query],
    )
    query_vec = list(embed_response.embeddings[0].values)

    collection = weaviate_client.collections.get(collection_name)

    # --- Website text results ---
    text_results: List[Dict[str, Any]] = []
    try:
        result = collection.query.near_vector(
            near_vector=query_vec,
            limit=top_k,
            filters=Filter.by_property("source_type").equal("website"),
            return_metadata=MetadataQuery(distance=True),
        )
        for obj in result.objects:
            props = obj.properties
            distance = obj.metadata.distance
            score = (1.0 - distance) if distance is not None else 0.0
            text_results.append({
                "score": float(score),
                "content": str(props.get("content", ""))[:_TEXT_CONTENT_MAX_CHARS],
                "source_url": str(props.get("source_url", "")),
            })
        logger.info("Text search returned %d hits", len(text_results))
    except Exception as exc:
        logger.error("Text search failed: %s", exc, exc_info=True)

    # --- Video frame results ---
    video_results: List[Dict[str, Any]] = []
    try:
        result = collection.query.near_vector(
            near_vector=query_vec,
            limit=top_k,
            filters=Filter.by_property("source_type").equal("youtube_frame"),
            return_metadata=MetadataQuery(distance=True),
        )
        for obj in result.objects:
            props = obj.properties
            distance = obj.metadata.distance
            score = (1.0 - distance) if distance is not None else 0.0
            video_id = props.get("video_id")
            timestamp_sec = props.get("timestamp_sec")
            deeplink: Optional[str] = None
            if video_id and timestamp_sec is not None:
                deeplink = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp_sec)}s"
            video_results.append({
                "score": float(score),
                "content": str(props.get("content", ""))[:_VIDEO_CONTENT_MAX_CHARS],
                "video_id": video_id,
                "timestamp_sec": timestamp_sec,
                "timestamp_hms": str(props.get("timestamp_hms", "")),
                "youtube_deeplink": deeplink,
            })
        logger.info("Video search returned %d hits", len(video_results))
    except Exception as exc:
        logger.error("Video search failed: %s", exc, exc_info=True)

    return {"text_results": text_results, "video_results": video_results}


def search_knowledge_base(query: str) -> str:
    """Search the Star Learners knowledge base for content relevant to the user's query.

    Searches both website text content and video frame content from the virtual tour.
    When a relevant video frame is found, the result includes a YouTube timestamp URL
    so the frontend can navigate to that exact moment in the tour video.

    Use this tool when the user asks about Star Learners facilities, classrooms,
    programs, fees, enrollment, or anything else that may be shown in the virtual tour.

    Args:
        query: The user's question or search query about Star Learners.

    Returns:
        Formatted text with relevant knowledge base content and video timestamps.
    """
    try:
        results = search_weaviate(query, top_k=3)
        parts = []

        if results["text_results"]:
            parts.append("Knowledge Base:")
            for i, r in enumerate(results["text_results"], 1):
                parts.append(f"\n[{i}] {r['content']}")
                if r.get("source_url"):
                    parts.append(f"    Source: {r['source_url']}")

        if results["video_results"]:
            parts.append("\nVirtual Tour Video References:")
            for r in results["video_results"]:
                if r.get("youtube_deeplink"):
                    ts = r.get("timestamp_hms") or f"{r.get('timestamp_sec', 0)}s"
                    parts.append(f"\n- At {ts}: {r['content']}")
                    parts.append(f"  Link: {r['youtube_deeplink']}")

        if not parts:
            return "No relevant information found in the knowledge base."

        return "\n".join(parts)

    except Exception as exc:
        logger.error("Knowledge base search failed: %s", exc, exc_info=True)
        return f"Knowledge base search failed: {exc}"
