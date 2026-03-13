#!/usr/bin/env python3
"""Build a Weaviate index from Star Learners website pages and a YouTube demo video.

Pipeline modes:
  extract-frames  Download video → extract JPEG frames → write per-video index.jsonl (no embedding)
  embed           Embed website text + frame captions → write JSONL object files
  upload          Bulk upload JSONL object files to Weaviate StarLearnersKB collection
  all             Run extract-frames → embed → upload sequentially
  websites        Extract + embed + upload website text only (no video)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types
from yt_dlp import YoutubeDL

LOGGER = logging.getLogger("build_weaviate_index")
USER_AGENT = "Mozilla/5.0 (StarLearners-Weaviate-Ingestor)"

DEFAULT_COLLECTION = "StarLearnersKB"
DEFAULT_FRAME_INTERVAL_SEC = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBED_MODEL = "gemini-embedding-2-preview"
DEFAULT_CAPTION_MODEL = "gemini-2.5-flash"
DEFAULT_FRAMES_BASE_DIR = "videos/frames"
DEFAULT_WEBSITE_OBJECTS = "videos/website_objects.jsonl"
DEFAULT_FRAME_OBJECTS = "videos/frame_objects.jsonl"


@dataclass
class ScriptConfig:
    mode: str
    sources_path: Path
    frame_interval_sec: int
    collection: str
    recreate_collection: bool
    batch_size: int
    frames_base_dir: Path
    website_objects_path: Path
    frame_objects_path: Path


def parse_args() -> ScriptConfig:
    parser = argparse.ArgumentParser(description="Build Weaviate index from websites and YouTube frames")
    parser.add_argument(
        "--mode",
        choices=["all", "websites", "extract-frames", "embed", "upload"],
        default="all",
        help=(
            "all: extract-frames → embed → upload end-to-end | "
            "websites: website text only (embed + upload inline) | "
            "extract-frames: download video and extract JPEG frames to disk | "
            "embed: embed website text + frame images → write JSONL object files | "
            "upload: bulk-upload pre-embedded JSONL files to Weaviate"
        ),
    )
    parser.add_argument("--sources", default="data/sources.yaml")
    parser.add_argument("--frame-interval-sec", type=int, default=DEFAULT_FRAME_INTERVAL_SEC)
    parser.add_argument("--collection", default=os.getenv("WEAVIATE_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--frames-base-dir", default=DEFAULT_FRAMES_BASE_DIR)
    parser.add_argument("--website-objects", default=DEFAULT_WEBSITE_OBJECTS)
    parser.add_argument("--frame-objects", default=DEFAULT_FRAME_OBJECTS)
    args = parser.parse_args()
    return ScriptConfig(
        mode=args.mode,
        sources_path=Path(args.sources),
        frame_interval_sec=max(1, args.frame_interval_sec),
        collection=args.collection,
        recreate_collection=args.recreate_collection,
        batch_size=max(1, args.batch_size),
        frames_base_dir=Path(args.frames_base_dir),
        website_objects_path=Path(args.website_objects),
        frame_objects_path=Path(args.frame_objects),
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


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 120) -> List[str]:
    """Split text into chunks at sentence boundaries to avoid cutting words mid-word."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            # Overlap: keep trailing sentences that fit within overlap_chars
            overlap: List[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 <= overlap_chars:
                    overlap.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            current, current_len = overlap[:], overlap_len
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
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
    text = body.get_text(separator=" ", strip=True)
    text = re.sub(r" {2,}", " ", text)
    return title, text


def fetch_url(url: str, attempts: int = 3, timeout: int = 20) -> str:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except requests.RequestException as exc:
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
        except Exception as exc:
            last_error = exc
            delay = base_sleep * (2 ** (attempt - 1))
            LOGGER.warning("Operation failed (%s/%s): %s", attempt, attempts, exc)
            time.sleep(delay)
    raise RuntimeError(f"Operation failed after {attempts} attempts: {last_error}")


class GeminiEmbedder:
    """Handles all text and image embeddings via gemini-embedding-2-preview."""

    def __init__(self) -> None:
        gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        if not gcp_project:
            raise RuntimeError("Missing GCP_PROJECT / GOOGLE_CLOUD_PROJECT env var")
        self.client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
        self.embed_model = os.getenv("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.caption_model = os.getenv("GEMINI_CAPTION_MODEL", DEFAULT_CAPTION_MODEL)

    @staticmethod
    def _extract_vectors(response: Any) -> List[List[float]]:
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise ValueError("Embedding response has no embeddings")
        vectors: List[List[float]] = []
        for item in embeddings:
            values = getattr(item, "values", None)
            if not values:
                raise ValueError("Embedding item has no values")
            vectors.append(list(values))
        return vectors

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # gemini-embedding-2-preview returns one embedding per call — call once per text.
        vectors: List[List[float]] = []
        for text in texts:
            def _call(t: str = text) -> Any:
                return self.client.models.embed_content(
                    model=self.embed_model,
                    contents=[t],
                )
            response = with_retry(_call)
            vectors.append(self._extract_vectors(response)[0])
        return vectors

    def embed_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> List[float]:
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        def _call() -> Any:
            return self.client.models.embed_content(
                model=self.embed_model,
                contents=[part],
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
    """Single unified StarLearnersKB collection storing both website and frame objects."""

    def __init__(self, collection: str, recreate_collection: bool) -> None:
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
        self.collection_name = collection

        # Unified schema: source_type distinguishes website vs youtube_frame objects
        properties = [
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="source_type", data_type=DataType.TEXT),
            Property(name="source_url", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="video_id", data_type=DataType.TEXT),
            Property(name="timestamp_sec", data_type=DataType.INT),
            Property(name="timestamp_hms", data_type=DataType.TEXT),
            Property(name="created_at", data_type=DataType.TEXT),
        ]
        vector_index = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)

        exists = self.client.collections.exists(collection)
        if exists and recreate_collection:
            LOGGER.info("Recreating collection: %s", collection)
            self.client.collections.delete(collection)
            exists = False
        if not exists:
            LOGGER.info("Creating collection: %s", collection)
            self.client.collections.create(
                name=collection,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=vector_index,
                properties=properties,
            )

    def upsert_objects(self, objects: List[Dict[str, Any]], batch_size: int) -> None:
        if not objects:
            return
        collection = self.client.collections.get(self.collection_name)
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for obj in objects:
                batch.add_object(
                    properties=obj["properties"],
                    uuid=obj["doc_id"],
                    vector=obj["vector"],
                )
        LOGGER.info("Upserted %d objects into %s", len(objects), self.collection_name)

    def close(self) -> None:
        self.client.close()


def iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------------
# Phase 1 — extract-frames
# ---------------------------------------------------------------------------

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


def extract_and_save_frames(
    youtube_url: str,
    frame_interval_sec: int,
    frames_base_dir: Path,
    failures: List[Dict[str, str]],
    local_video: Optional[Path] = None,
) -> Tuple[str, Path]:
    """Phase 1: Extract frames from video and write index.jsonl. Returns (video_id, index_path)."""
    if local_video and local_video.exists():
        LOGGER.info("Using local video file: %s", local_video)
        video_id = extract_video_id_from_url(youtube_url) or local_video.stem[:64]
        video_path = local_video
    else:
        dl_dir = frames_base_dir / "_download"
        video_id, video_path = download_youtube_video(youtube_url, dl_dir)

    frames_dir = frames_base_dir / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    index_path = frames_dir / "index.jsonl"
    saved = 0
    next_ts = 0
    frame_index = 0

    try:
        with index_path.open("w", encoding="utf-8") as idx_fh:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                current_sec = int(frame_index / fps)
                if current_sec >= next_ts:
                    frame_n = saved + 1
                    frame_path = frames_dir / f"frame_{frame_n:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    record = {
                        "frame_n": frame_n,
                        "timestamp_sec": current_sec,
                        "timestamp_hms": to_hms(current_sec),
                        "jpg_path": str(frame_path),
                    }
                    idx_fh.write(json.dumps(record) + "\n")
                    saved += 1
                    next_ts += frame_interval_sec
                frame_index += 1
    finally:
        capture.release()

    LOGGER.info("Extracted %d frames for video %s → %s", saved, video_id, index_path)
    return video_id, index_path


# ---------------------------------------------------------------------------
# Phase 2 — embed
# ---------------------------------------------------------------------------

def embed_website_objects(
    websites: List[str],
    embedder: GeminiEmbedder,
    output_path: Path,
    batch_size: int,
    failures: List[Dict[str, str]],
) -> int:
    """Embed website text chunks and write Weaviate-ready objects to JSONL. Returns count written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
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
        except Exception as exc:
            LOGGER.exception("Website ingestion failed for %s", url)
            failures.append({"source": url, "error": str(exc)})

    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for batch in iter_batches(records, batch_size):
            texts = [r["content"] for r in batch]
            vectors = embedder.embed_texts(texts)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch: sent {len(texts)}, received {len(vectors)}"
                )
            for item, vector in zip(batch, vectors):
                obj = {
                    "doc_id": item["doc_id"],
                    "vector": vector,
                    "properties": {
                        "doc_id": item["doc_id"],
                        "source_type": "website",
                        "source_url": item["url"],
                        "title": item["title"],
                        "content": item["content"],
                        "chunk_index": item["chunk_index"],
                        "created_at": now_iso(),
                    },
                }
                fh.write(json.dumps(obj) + "\n")
                written += 1

    LOGGER.info("Wrote %d website objects to %s", written, output_path)
    return written


def embed_frame_objects(
    youtube_url: str,
    video_id: str,
    index_path: Path,
    embedder: GeminiEmbedder,
    output_path: Path,
    failures: List[Dict[str, str]],
) -> int:
    """Caption + embed frames from index.jsonl and write Weaviate-ready objects. Returns count written."""
    if not index_path.exists():
        raise FileNotFoundError(f"Frame index not found: {index_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_records = []
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                frame_records.append(json.loads(line))

    LOGGER.info("Embedding %d frames from %s", len(frame_records), index_path)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for fr in frame_records:
            try:
                jpg_path = Path(fr["jpg_path"])
                image_bytes = jpg_path.read_bytes()
                caption = embedder.caption_image(image_bytes)
                image_vector = embedder.embed_image(image_bytes)
                doc_id = stable_hash(["youtube", video_id, str(fr["timestamp_sec"])])
                obj = {
                    "doc_id": doc_id,
                    "vector": image_vector,
                    "properties": {
                        "doc_id": doc_id,
                        "source_type": "youtube_frame",
                        "source_url": youtube_url,
                        "title": f"YouTube frame at {fr['timestamp_hms']}",
                        "content": caption,
                        "video_id": video_id,
                        "timestamp_sec": fr["timestamp_sec"],
                        "timestamp_hms": fr["timestamp_hms"],
                        "created_at": now_iso(),
                    },
                }
                fh.write(json.dumps(obj) + "\n")
                fh.flush()
                written += 1
                LOGGER.info(
                    "Embedded frame %d/%d at %s", written, len(frame_records), fr["timestamp_hms"]
                )
            except Exception as exc:
                LOGGER.exception("Failed embedding frame %s", fr.get("jpg_path"))
                failures.append({"source": fr.get("jpg_path", ""), "error": str(exc)})

    LOGGER.info("Wrote %d frame objects to %s", written, output_path)
    return written


# ---------------------------------------------------------------------------
# Phase 3 — upload
# ---------------------------------------------------------------------------

def upload_objects_from_jsonl(
    jsonl_path: Path,
    store: WeaviateStore,
    batch_size: int,
    failures: List[Dict[str, str]],
) -> int:
    """Read JSONL of Weaviate objects and bulk-upsert. Returns count upserted."""
    if not jsonl_path.exists():
        LOGGER.warning("JSONL file not found, skipping: %s", jsonl_path)
        return 0

    objects: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    objects.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping malformed JSONL line: %s", exc)

    LOGGER.info("Loaded %d objects from %s", len(objects), jsonl_path)
    store.upsert_objects(objects, batch_size)
    return len(objects)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")
    configure_logging()
    cfg = parse_args()

    websites, youtube_url, local_video = load_sources(cfg.sources_path)
    failures: List[Dict[str, str]] = []
    video_id: Optional[str] = None
    index_path: Optional[Path] = None

    # ---- Phase 1: extract-frames ----
    if cfg.mode in {"all", "extract-frames"}:
        LOGGER.info("Phase 1: Extracting frames from %s", youtube_url)
        video_id, index_path = extract_and_save_frames(
            youtube_url=youtube_url,
            frame_interval_sec=cfg.frame_interval_sec,
            frames_base_dir=cfg.frames_base_dir,
            failures=failures,
            local_video=local_video,
        )
        if cfg.mode == "extract-frames":
            print(json.dumps({
                "mode": cfg.mode,
                "video_id": video_id,
                "index_path": str(index_path),
                "failures_count": len(failures),
                "failures": failures,
            }, indent=2))
            return

    # ---- Phase 2: embed ----
    if cfg.mode in {"all", "embed", "websites"}:
        embedder = GeminiEmbedder()

        LOGGER.info("Phase 2: Embedding website text (%d URLs)", len(websites))
        website_written = embed_website_objects(
            websites=websites,
            embedder=embedder,
            output_path=cfg.website_objects_path,
            batch_size=cfg.batch_size,
            failures=failures,
        )

        frame_written = 0
        if cfg.mode in {"all", "embed"}:
            # In standalone embed mode, locate index from a prior extract-frames run
            if cfg.mode == "embed":
                index_files = list(cfg.frames_base_dir.glob("*/index.jsonl"))
                if not index_files:
                    raise FileNotFoundError(
                        f"No frame index files found in {cfg.frames_base_dir}. "
                        "Run --mode extract-frames first."
                    )
                if len(index_files) > 1:
                    LOGGER.warning("Multiple frame indexes found, using: %s", index_files[0])
                index_path = index_files[0]
                video_id = index_path.parent.name

            LOGGER.info("Phase 2: Embedding frames for video %s", video_id)
            frame_written = embed_frame_objects(
                youtube_url=youtube_url,
                video_id=video_id,
                index_path=index_path,
                embedder=embedder,
                output_path=cfg.frame_objects_path,
                failures=failures,
            )

        if cfg.mode == "embed":
            print(json.dumps({
                "mode": cfg.mode,
                "website_objects_written": website_written,
                "frame_objects_written": frame_written,
                "website_objects_path": str(cfg.website_objects_path),
                "frame_objects_path": str(cfg.frame_objects_path),
                "failures_count": len(failures),
                "failures": failures,
            }, indent=2))
            return

    # ---- Phase 3: upload ----
    if cfg.mode in {"all", "upload", "websites"}:
        store = WeaviateStore(
            collection=cfg.collection,
            recreate_collection=cfg.recreate_collection,
        )
        try:
            website_upserted = upload_objects_from_jsonl(
                jsonl_path=cfg.website_objects_path,
                store=store,
                batch_size=cfg.batch_size,
                failures=failures,
            )
            frame_upserted = 0
            if cfg.mode in {"all", "upload"}:
                frame_upserted = upload_objects_from_jsonl(
                    jsonl_path=cfg.frame_objects_path,
                    store=store,
                    batch_size=cfg.batch_size,
                    failures=failures,
                )
        finally:
            store.close()

        print(json.dumps({
            "mode": cfg.mode,
            "collection": cfg.collection,
            "website_objects_upserted": website_upserted,
            "frame_objects_upserted": frame_upserted,
            "failures_count": len(failures),
            "failures": failures,
        }, indent=2))


if __name__ == "__main__":
    main()
