"""Qdrant retrieval tool for the Star Learners agent.

Provides both a structured search function (used by the API endpoint)
and a string-returning function tool (used by the ADK agent).

All embeddings use Vertex AI credentials (GCP_PROJECT env var required).
Text embeddings use gemini-embedding-001 in us-central1.
Multimodal embeddings use multimodalembedding@001 in us-central1.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy singletons to avoid re-initializing on every call
# ---------------------------------------------------------------------------
_genai_client = None
_mm_model = None
_qdrant_client = None

# Region where gemini-embedding-001 is reliably available
_EMBED_LOCATION = "us-central1"

# Ensure environment variables are available regardless of process cwd.
_APP_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_APP_ENV_PATH)
load_dotenv()


def _get_genai_client():
    """Return a Vertex AI genai client for text embeddings."""
    global _genai_client
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

    multimodalembedding@001 is only available in us-central1, so we must
    override the global Vertex AI location for this call.
    """
    global _mm_model
    if _mm_model is None:
        import vertexai
        from vertexai.vision_models import MultiModalEmbeddingModel
        gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        # Re-init to us-central1 where multimodalembedding@001 is available
        vertexai.init(project=gcp_project, location="us-central1")
        _mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    return _mm_model


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return _qdrant_client


def search_qdrant(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search Qdrant for both website text and YouTube video frames.

    Returns a structured dict suitable for both the REST endpoint and the agent tool.
    """
    from google.genai import types

    collection = os.getenv("QDRANT_COLLECTION", "star_learners_kb")
    logger.info("Qdrant search | collection=%s query=%r", collection, query[:80])

    genai_client = _get_genai_client()
    qdrant = _get_qdrant_client()

    # --- Text / website search ---
    text_results: List[Dict[str, Any]] = []
    try:
        text_response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        text_vec = list(text_response.embeddings[0].values)

        text_hits = qdrant.query_points(
            collection_name=collection,
            query=text_vec,
            using="text_vector",
            limit=top_k,
            with_payload=True,
        ).points

        for hit in text_hits:
            payload = dict(hit.payload or {})
            text_results.append({
                "score": float(hit.score),
                "content": payload.get("content", "")[:600],
                "source_url": payload.get("source_url", ""),
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

        image_hits = qdrant.query_points(
            collection_name=collection,
            query=image_vec,
            using="image_vector",
            limit=top_k,
            with_payload=True,
        ).points

        for hit in image_hits:
            payload = dict(hit.payload or {})
            video_id = payload.get("video_id")
            timestamp_sec = payload.get("timestamp_sec")
            deeplink = None
            if video_id and timestamp_sec is not None:
                deeplink = f"https://www.youtube.com/watch?v={video_id}&t={timestamp_sec}s"
            video_results.append({
                "score": float(hit.score),
                "content": payload.get("content", "")[:300],
                "video_id": video_id,
                "timestamp_sec": timestamp_sec,
                "timestamp_hms": payload.get("timestamp_hms", ""),
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
        results = search_qdrant(query, top_k=3)
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
        return f"Knowledge base search failed: {exc}"
