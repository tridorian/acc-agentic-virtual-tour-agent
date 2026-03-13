"""Weaviate retrieval tool for the Star Learners agent.

Provides both a structured search function (used by the API endpoint)
and a string-returning function tool (used by the ADK agent).

All embeddings use Vertex AI credentials (GCP_PROJECT env var required).
Text embeddings use gemini-embedding-001 in us-central1.
Multimodal embeddings use multimodalembedding@001 in us-central1.
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
_mm_model = None
_weaviate_client = None

_genai_lock = threading.Lock()
_mm_lock = threading.Lock()
_weaviate_lock = threading.Lock()

# Named constants — avoids magic numbers in result truncation
_TEXT_CONTENT_MAX_CHARS = 600   # Max chars returned per text result
_VIDEO_CONTENT_MAX_CHARS = 300  # Max chars returned per video frame caption

# Region where gemini-embedding-001 is reliably available
_EMBED_LOCATION = "us-central1"

# Ensure environment variables are available regardless of process cwd.
_APP_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_APP_ENV_PATH)


def _get_genai_client():
    """Return a Vertex AI genai client for text embeddings."""
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


def _get_mm_model():
    """Return the cached Vertex AI multimodal embedding model.

    NOTE: multimodalembedding@001 requires us-central1. This function
    re-initializes the global vertexai state to us-central1 on first call.
    This is safe because it only runs once (singleton) and the process-wide
    vertexai location is us-central1 for all embedding paths anyway.
    """
    global _mm_model
    if _mm_model is None:
        with _mm_lock:
            if _mm_model is None:
                import vertexai
                from vertexai.vision_models import MultiModalEmbeddingModel
                gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
                if not gcp_project:
                    raise RuntimeError("Missing GCP_PROJECT / GOOGLE_CLOUD_PROJECT env var")
                vertexai.init(project=gcp_project, location="us-central1")
                _mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    return _mm_model


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

    Returns a structured dict suitable for both the REST endpoint and the agent tool.
    """
    from google.genai import types
    from weaviate.classes.query import MetadataQuery

    website_collection = os.getenv("WEAVIATE_COLLECTION_WEBSITE", "StarLearnersWebsite")
    frame_collection = os.getenv("WEAVIATE_COLLECTION_FRAME", "StarLearnersFrame")
    logger.info("Weaviate search | query=%r", query[:80])

    genai_client = _get_genai_client()
    weaviate_client = _get_weaviate_client()

    # --- Text / website search ---
    text_results: List[Dict[str, Any]] = []
    try:
        text_response = genai_client.models.embed_content(
            model=os.getenv("GEMINI_TEXT_EMBED_MODEL", "gemini-embedding-001"),
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        text_vec = list(text_response.embeddings[0].values)

        collection = weaviate_client.collections.get(website_collection)
        result = collection.query.near_vector(
            near_vector=text_vec,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        for obj in result.objects:
            props = obj.properties
            # Convert distance to similarity score (lower distance = more similar)
            distance = obj.metadata.distance
            score = (1.0 - distance) if distance is not None else 0.0
            text_results.append({
                "score": float(score),
                "content": str(props.get("content", ""))[:_TEXT_CONTENT_MAX_CHARS],
                "source_url": str(props.get("source_url", "")),
            })
        logger.info("Text search returned %d hits", len(text_results))
    except Exception as exc:
        logger.error("Text embedding/search failed: %s", exc, exc_info=True)

    # --- Image / video frame search ---
    video_results: List[Dict[str, Any]] = []
    try:
        mm_model = _get_mm_model()
        image_embed = mm_model.get_embeddings(contextual_text=query)
        image_vec = list(image_embed.text_embedding)

        collection = weaviate_client.collections.get(frame_collection)
        result = collection.query.near_vector(
            near_vector=image_vec,
            limit=top_k,
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
        logger.info("Image search returned %d hits", len(video_results))
    except Exception as exc:
        logger.error("Image embedding/search failed: %s", exc, exc_info=True)

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
