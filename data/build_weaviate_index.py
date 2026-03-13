#!/usr/bin/env python3
"""Build a Weaviate index from Star Learners website pages and a YouTube demo video.

This script ingests:
- website text chunks -> Gemini text embeddings -> Weaviate StarLearnersWebsite collection
- YouTube frames (every N seconds) -> Gemini frame captions + image embeddings -> Weaviate StarLearnersFrame collection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import requests
import vertexai
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types
from vertexai.vision_models import Image as VertexImage
from vertexai.vision_models import MultiModalEmbeddingModel
from yt_dlp import YoutubeDL

LOGGER = logging.getLogger("build_weaviate_index")
USER_AGENT = "Mozilla/5.0 (StarLearners-Weaviate-Ingestor)"

DEFAULT_WEBSITE_COLLECTION = "StarLearnersWebsite"
DEFAULT_FRAME_COLLECTION = "StarLearnersFrame"
DEFAULT_FRAME_INTERVAL_SEC = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEXT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_IMAGE_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_CAPTION_MODEL = "gemini-2.0-flash"
DEFAULT_FRAMES_DIR = "data/frames"
DEFAULT_FRAMES_INDEX = "data/frames/frames_index.jsonl"


@dataclass
class ScriptConfig:
    mode: str
    sources_path: Path
    frame_interval_sec: int
    website_collection: str
    frame_collection: str
    recreate_collection: bool
    batch_size: int
    frames_dir: Path
    frames_index: Path


def parse_args() -> ScriptConfig:
    parser = argparse.ArgumentParser(description="Build Weaviate index from websites and YouTube frames")
    parser.add_argument(
        "--mode",
        choices=["all", "websites", "youtube", "extract-frames", "upload-frames"],
        default="all",
        help=(
            "all: websites+youtube end-to-end | "
            "websites: ingest website text only | "
            "youtube: download+embed+upload video frames | "
            "extract-frames: extract frames+captions+embeddings to local JSONL | "
            "upload-frames: bulk-upload pre-extracted frames JSONL to Weaviate"
        ),
    )
    parser.add_argument("--sources", default="data/sources.yaml")
    parser.add_argument("--frame-interval-sec", type=int, default=DEFAULT_FRAME_INTERVAL_SEC)
    parser.add_argument("--website-collection", default=os.getenv("WEAVIATE_COLLECTION_WEBSITE", DEFAULT_WEBSITE_COLLECTION))
    parser.add_argument("--frame-collection", default=os.getenv("WEAVIATE_COLLECTION_FRAME", DEFAULT_FRAME_COLLECTION))
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR, help="Directory to save extracted frames")
    parser.add_argument("--frames-index", default=DEFAULT_FRAMES_INDEX, help="JSONL file for frame metadata+embeddings")
    args = parser.parse_args()
    return ScriptConfig(
        mode=args.mode,
        sources_path=Path(args.sources),
        frame_interval_sec=max(1, args.frame_interval_sec),
        website_collection=args.website_collection,
        frame_collection=args.frame_collection,
        recreate_collection=args.recreate_collection,
        batch_size=max(1, args.batch_size),
        frames_dir=Path(args.frames_dir),
        frames_index=Path(args.frames_index),
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(parts: Iterable[str]) -> str:
    raw = "||".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16]))


def extract_video_id_from_url(url: str) -> Optional[str]:
    """Extract YouTube video ID from watch, short-link, or Shorts URLs."""
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return None


def to_hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def load_sources(path: Path) -> Tuple[List[str], str, Optional[Path]]:
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    websites = payload.get("websites") or []
    youtube = payload.get("youtube") or {}
    youtube_url = youtube.get("url")
    local_video_str = youtube.get("local_video")
    if not isinstance(websites, list):
        raise ValueError("sources.yaml: `websites` must be a list")
    if not youtube_url:
        raise ValueError("sources.yaml: missing youtube.url")
    local_video = Path(local_video_str) if local_video_str else None
    return websites, youtube_url, local_video


def extract_readable_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    body = soup.body if soup.body else soup
    text = body.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return title, text


_http_session: Optional[requests.Session] = None


def _get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        _http_session.headers.update({"User-Agent": USER_AGENT})
    return _http_session


def fetch_url(url: str, attempts: int = 3, timeout: int = 20) -> str:
    session = _get_http_session()
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep_s = min(2 * attempt, 6)
            LOGGER.warning("Fetch failed (%s/%s) for %s: %s", attempt, attempts, url, exc)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def with_retry(fn, attempts: int = 3, base_sleep: float = 1.0):
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            delay = base_sleep * (2 ** (attempt - 1))
            LOGGER.warning("Operation failed (%s/%s): %s", attempt, attempts, exc)
            time.sleep(delay)
    raise RuntimeError(f"Operation failed after {attempts} attempts: {last_error}")


class GeminiEmbedder:
    def __init__(self) -> None:
        gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        if not gcp_project:
            raise RuntimeError("Missing GCP_PROJECT / GOOGLE_CLOUD_PROJECT env var")
        vertexai.init(project=gcp_project, location=gcp_location)
        self.client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
        self.text_embed_model = os.getenv("GEMINI_TEXT_EMBED_MODEL", DEFAULT_TEXT_EMBED_MODEL)
        self.image_embed_model = os.getenv("GEMINI_IMAGE_EMBED_MODEL", DEFAULT_IMAGE_EMBED_MODEL)
        self.caption_model = os.getenv("GEMINI_CAPTION_MODEL", DEFAULT_CAPTION_MODEL)

        self._vertex_image_model: Optional[MultiModalEmbeddingModel] = None
        if "multimodalembedding" in self.image_embed_model:
            self._vertex_image_model = MultiModalEmbeddingModel.from_pretrained(self.image_embed_model)
            LOGGER.info("Vertex AI multimodal embedding model loaded: %s", self.image_embed_model)

    @staticmethod
    def _extract_vectors(response: Any) -> List[List[float]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if not embeddings:
            raise ValueError("Embedding response has no embeddings")

        vectors: List[List[float]] = []
        for item in embeddings:
            values = getattr(item, "values", None)
            if values is None and isinstance(item, dict):
                values = item.get("values")
            if not values:
                raise ValueError("Embedding item has no values")
            vectors.append(list(values))
        return vectors

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        def _call() -> Any:
            return self.client.models.embed_content(
                model=self.text_embed_model,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )

        response = with_retry(_call)
        return self._extract_vectors(response)

    def embed_single_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> List[float]:
        if self._vertex_image_model is not None:
            vertex_image = VertexImage(image_bytes=image_bytes)

            def _vertex_call() -> Any:
                return self._vertex_image_model.get_embeddings(image=vertex_image)

            result = with_retry(_vertex_call)
            return list(result.image_embedding)

        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        def _call() -> Any:
            return self.client.models.embed_content(
                model=self.image_embed_model,
                contents=[part],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )

        response = with_retry(_call)
        return self._extract_vectors(response)[0]

    def caption_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        prompt = (
            "Describe this frame in 1-2 short sentences focused on what a parent can see "
            "during a preschool tour (classroom setup, activity, people, signage)."
        )
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        def _call() -> Any:
            return self.client.models.generate_content(
                model=self.caption_model,
                contents=[prompt, part],
            )

        response = with_retry(_call)
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    return t.strip()
        return "Frame from Star Learners tour video."


class WeaviateStore:
    def __init__(self, website_collection: str, frame_collection: str, recreate_collection: bool) -> None:
        import weaviate
        from weaviate.classes.config import Configure, DataType, Property, VectorDistances
        
        from weaviate.connect import ConnectionParams

        endpoint = require_env("WEAVIATE_ENDPOINT")
        api_key = os.getenv("WEAVIATE_API_KEY")
        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

        parsed = urlparse(endpoint)
        http_host = parsed.hostname or "localhost"
        http_port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        secure = parsed.scheme == "https"

        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None

        self.client = weaviate.WeaviateClient(
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
        self.client.connect()

        self.website_collection = website_collection
        self.frame_collection = frame_collection

        website_props = [
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="source_type", data_type=DataType.TEXT),
            Property(name="source_url", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="created_at", data_type=DataType.TEXT),
        ]
        frame_props = [
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="source_type", data_type=DataType.TEXT),
            Property(name="source_url", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="video_id", data_type=DataType.TEXT),
            Property(name="timestamp_sec", data_type=DataType.INT),
            Property(name="timestamp_hms", data_type=DataType.TEXT),
            Property(name="created_at", data_type=DataType.TEXT),
        ]

        vector_index = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)

        self._ensure_collection(
            website_collection, website_props, vector_index, recreate_collection, Configure
        )
        self._ensure_collection(
            frame_collection, frame_props, vector_index, recreate_collection, Configure
        )

    def _ensure_collection(self, name: str, properties, vector_index, recreate: bool, Configure) -> None:
        exists = self.client.collections.exists(name)
        if exists and recreate:
            LOGGER.info("Recreating collection: %s", name)
            self.client.collections.delete(name)
            exists = False

        if not exists:
            LOGGER.info("Creating collection: %s", name)
            self.client.collections.create(
                name=name,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=vector_index,
                properties=properties,
            )

    def upsert_website_objects(self, objects: List[Dict[str, Any]]) -> None:
        if not objects:
            return
        collection = self.client.collections.get(self.website_collection)
        with collection.batch.fixed_size(batch_size=100) as batch:
            for obj in objects:
                batch.add_object(
                    properties=obj["properties"],
                    uuid=obj["doc_id"],
                    vector=obj["vector"],
                )
        LOGGER.info("Upserted %d website objects", len(objects))

    def upsert_frame_objects(self, objects: List[Dict[str, Any]]) -> None:
        if not objects:
            return
        collection = self.client.collections.get(self.frame_collection)
        with collection.batch.fixed_size(batch_size=100) as batch:
            for obj in objects:
                batch.add_object(
                    properties=obj["properties"],
                    uuid=obj["doc_id"],
                    vector=obj["vector"],
                )
        LOGGER.info("Upserted %d frame objects", len(objects))

    def close(self) -> None:
        self.client.close()


def iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_website_objects(
    websites: List[str],
    embedder: GeminiEmbedder,
    failures: List[Dict[str, str]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for url in websites:
        try:
            html = fetch_url(url)
            title, text = extract_readable_text(html)
            chunks = chunk_text(text)
            if not chunks:
                raise ValueError("No readable content extracted")

            for idx, chunk in enumerate(chunks):
                doc_id = stable_hash(["website", url, str(idx), chunk])
                records.append({
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "chunk_index": idx,
                    "content": chunk,
                })
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Website ingestion failed for %s", url)
            failures.append({"source": url, "error": str(exc)})

    objects: List[Dict[str, Any]] = []
    for batch in iter_batches(records, batch_size):
        texts = [r["content"] for r in batch]
        vectors = embedder.embed_texts(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: sent {len(texts)} texts, "
                f"received {len(vectors)} embeddings"
            )

        for item, vector in zip(batch, vectors):
            objects.append({
                "doc_id": item["doc_id"],
                "vector": vector,
                "properties": {
                    "doc_id": item["doc_id"],
                    "source_type": "website_chunk",
                    "source_url": item["url"],
                    "title": item["title"],
                    "content": item["content"],
                    "chunk_index": item["chunk_index"],
                    "created_at": now_iso(),
                },
            })

    return objects


def download_youtube_video(youtube_url: str, out_dir: Path) -> Tuple[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "format": "mp4/best[ext=mp4]/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id")
        if not video_id:
            raise RuntimeError("Failed to determine YouTube video id")
        ext = info.get("ext") or "mp4"
        local_path = out_dir / f"{video_id}.{ext}"
        if not local_path.exists():
            candidates = sorted(out_dir.glob(f"{video_id}.*"))
            if not candidates:
                raise RuntimeError("Downloaded video file not found")
            local_path = candidates[0]
        return video_id, local_path


def extract_frames(video_path: Path, frame_interval_sec: int, out_dir: Path) -> List[Tuple[int, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frames: List[Tuple[int, Path]] = []
    next_ts = 0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            current_sec = int(frame_index / fps)
            if current_sec >= next_ts:
                frame_path = out_dir / f"frame_{current_sec:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append((current_sec, frame_path))
                next_ts += frame_interval_sec
            frame_index += 1
    finally:
        capture.release()

    return frames


def build_youtube_objects(
    youtube_url: str,
    frame_interval_sec: int,
    embedder: GeminiEmbedder,
    failures: List[Dict[str, str]],
    local_video: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="yt_ingest_") as tmp:
        tmp_path = Path(tmp)
        if local_video and local_video.exists():
            LOGGER.info("Using local video file: %s", local_video)
            video_id = extract_video_id_from_url(youtube_url) or local_video.stem[:64]
            video_path = local_video
        else:
            video_id, video_path = download_youtube_video(youtube_url, tmp_path / "video")
        frame_items = extract_frames(video_path, frame_interval_sec, tmp_path / "frames")

        objects: List[Dict[str, Any]] = []
        for timestamp_sec, frame_path in frame_items:
            try:
                image_bytes = frame_path.read_bytes()
                caption = embedder.caption_image(image_bytes, mime_type="image/jpeg")
                image_vector = embedder.embed_image(image_bytes, mime_type="image/jpeg")

                doc_id = stable_hash(["youtube", video_id, str(timestamp_sec)])
                objects.append({
                    "doc_id": doc_id,
                    "vector": image_vector,
                    "properties": {
                        "doc_id": doc_id,
                        "source_type": "youtube_frame",
                        "source_url": youtube_url,
                        "title": f"YouTube frame at {to_hms(timestamp_sec)}",
                        "content": caption,
                        "video_id": video_id,
                        "timestamp_sec": timestamp_sec,
                        "timestamp_hms": to_hms(timestamp_sec),
                        "created_at": now_iso(),
                    },
                })
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed processing frame %s", frame_path)
                failures.append({"source": str(frame_path), "error": str(exc)})

        return objects


def extract_and_save_frames(
    youtube_url: str,
    frame_interval_sec: int,
    embedder: GeminiEmbedder,
    frames_dir: Path,
    frames_index: Path,
    failures: List[Dict[str, str]],
    local_video: Optional[Path] = None,
) -> int:
    """Extract frames from video, caption + embed each, save to local JSONL. Returns count saved."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames_index.parent.mkdir(parents=True, exist_ok=True)

    _tmp_ctx = None
    try:
        if local_video and local_video.exists():
            LOGGER.info("Using local video file: %s", local_video)
            video_id = extract_video_id_from_url(youtube_url) or local_video.stem[:64]
            video_path = local_video
        else:
            _tmp_ctx = tempfile.TemporaryDirectory(prefix="yt_ingest_")
            tmp_path = Path(_tmp_ctx.name)
            video_id, video_path = download_youtube_video(youtube_url, tmp_path / "video")
        frame_items = extract_frames(video_path, frame_interval_sec, frames_dir)
    except Exception:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()
        raise

    if frames_index.exists():
        LOGGER.warning("Frames index already exists: %s — appending to existing file.", frames_index)

    saved = 0
    open_mode = "a" if frames_index.exists() else "w"
    with frames_index.open(open_mode, encoding="utf-8") as fh:
        for timestamp_sec, frame_path in frame_items:
            try:
                image_bytes = frame_path.read_bytes()
                caption = embedder.caption_image(image_bytes, mime_type="image/jpeg")
                image_vector = embedder.embed_image(image_bytes, mime_type="image/jpeg")

                doc_id = stable_hash(["youtube", video_id, str(timestamp_sec)])
                record = {
                    "doc_id": doc_id,
                    "source_type": "youtube_frame",
                    "source_url": youtube_url,
                    "title": f"YouTube frame at {to_hms(timestamp_sec)}",
                    "content": caption,
                    "video_id": video_id,
                    "timestamp_sec": timestamp_sec,
                    "timestamp_hms": to_hms(timestamp_sec),
                    "frame_path": str(frame_path),
                    "image_vector": image_vector,
                    "created_at": now_iso(),
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                saved += 1
                LOGGER.info("Saved frame %s/%s — %s", saved, len(frame_items), to_hms(timestamp_sec))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed processing frame %s", frame_path)
                failures.append({"source": str(frame_path), "error": str(exc)})

    if _tmp_ctx is not None:
        _tmp_ctx.cleanup()
    LOGGER.info("Frames extracted and saved to %s (%s records)", frames_index, saved)
    return saved


def upload_frames_from_index(
    frames_index: Path,
    store: WeaviateStore,
    batch_size: int,
    failures: List[Dict[str, str]],
) -> int:
    """Read pre-extracted frames JSONL and bulk-upsert to Weaviate. Returns count upserted."""
    if not frames_index.exists():
        raise FileNotFoundError(f"Frames index not found: {frames_index}")

    records = []
    with frames_index.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    LOGGER.info("Loaded %s frame records from %s", len(records), frames_index)

    objects: List[Dict[str, Any]] = []
    for r in records:
        try:
            properties = {k: v for k, v in r.items() if k not in ("image_vector", "frame_path")}
            objects.append({
                "doc_id": r["doc_id"],
                "vector": r["image_vector"],
                "properties": properties,
            })
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed building object for %s", r.get("frame_path"))
            failures.append({"source": r.get("frame_path", ""), "error": str(exc)})

    store.upsert_frame_objects(objects)
    return len(objects)


def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")
    configure_logging()
    cfg = parse_args()

    websites, youtube_url, local_video = load_sources(cfg.sources_path)
    failures: List[Dict[str, str]] = []

    # extract-frames: only needs embedder, no Weaviate
    if cfg.mode == "extract-frames":
        embedder = GeminiEmbedder()
        saved = extract_and_save_frames(
            youtube_url=youtube_url,
            frame_interval_sec=cfg.frame_interval_sec,
            embedder=embedder,
            frames_dir=cfg.frames_dir,
            frames_index=cfg.frames_index,
            failures=failures,
            local_video=local_video,
        )
        print(json.dumps({
            "mode": cfg.mode,
            "frames_saved": saved,
            "frames_index": str(cfg.frames_index),
            "failures_count": len(failures),
            "failures": failures,
        }, indent=2))
        return

    embedder = GeminiEmbedder()
    store = WeaviateStore(
        website_collection=cfg.website_collection,
        frame_collection=cfg.frame_collection,
        recreate_collection=cfg.recreate_collection,
    )

    try:
        # upload-frames: read pre-extracted JSONL and bulk upsert
        if cfg.mode == "upload-frames":
            upserted = upload_frames_from_index(
                frames_index=cfg.frames_index,
                store=store,
                batch_size=cfg.batch_size,
                failures=failures,
            )
            print(json.dumps({
                "mode": cfg.mode,
                "frames_upserted": upserted,
                "failures_count": len(failures),
                "failures": failures,
            }, indent=2))
            return

        website_objects: List[Dict[str, Any]] = []
        youtube_objects: List[Dict[str, Any]] = []

        if cfg.mode in {"all", "websites"}:
            LOGGER.info("Ingesting website sources (%s URLs)", len(websites))
            website_objects = build_website_objects(
                websites=websites,
                embedder=embedder,
                failures=failures,
                batch_size=cfg.batch_size,
            )
            store.upsert_website_objects(website_objects)
            LOGGER.info("Upserted website objects: %s", len(website_objects))

        if cfg.mode in {"all", "youtube"}:
            LOGGER.info("Ingesting YouTube source: %s", youtube_url)
            youtube_objects = build_youtube_objects(
                youtube_url=youtube_url,
                frame_interval_sec=cfg.frame_interval_sec,
                embedder=embedder,
                failures=failures,
                local_video=local_video,
            )
            store.upsert_frame_objects(youtube_objects)
            LOGGER.info("Upserted YouTube frame objects: %s", len(youtube_objects))

        summary = {
            "mode": cfg.mode,
            "website_collection": cfg.website_collection,
            "frame_collection": cfg.frame_collection,
            "website_objects_upserted": len(website_objects),
            "youtube_objects_upserted": len(youtube_objects),
            "failures_count": len(failures),
            "failures": failures,
        }
        print(json.dumps(summary, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
