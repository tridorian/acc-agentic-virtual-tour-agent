# Weaviate + Gemini Data Pipeline

This folder contains the data ingestion and retrieval pipeline for:
- Star Learners website ingestion (text chunks → `StarLearnersWebsite` collection)
- YouTube demo video ingestion (timestamped frames → `StarLearnersFrame` collection)
- Querying Weaviate for website answers and tour/demo timestamps

## Files

- `build_weaviate_index.py`: ingest websites + YouTube into Weaviate
- `query_weaviate.py`: search and return top results with YouTube deeplinks
- `sources.yaml`: exact URLs and YouTube source URL
- `requirements.txt`: Python dependencies for this pipeline
- `.env`: environment variables (copy from below)

## Infrastructure

Weaviate runs on GKE cluster `weaviate-cluster` in `us-central1`, inside VPC `weaviate-vpc`.

| Service | Type | Internal LB IP | Ports |
|---------|------|----------------|-------|
| `weaviate-ilb` | LoadBalancer | `10.10.0.3` | HTTP `8080`, gRPC `50051` |

The internal LB is only reachable from within the VPC. For local development, use `kubectl port-forward`.

## Local Setup

### 1. Install dependencies

```bash
# From repo root
uv pip install -r data/requirements.txt
```

### 2. GKE cluster credentials

```bash
export KUBERNETES_CLUSTER_NAME=weaviate-cluster
export GOOGLE_CLOUD_REGION=us-central1
gcloud container clusters get-credentials $KUBERNETES_CLUSTER_NAME --region $GOOGLE_CLOUD_REGION
```

Verify Weaviate services are running:

```bash
kubectl get svc -n weaviate
```

### 3. Port-forward Weaviate to localhost

Run this in a separate terminal and keep it open during ingestion/querying:

```bash
kubectl port-forward svc/weaviate-ilb 8080:8080 50051:50051 -n weaviate
```

Expected output:
```
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from 127.0.0.1:50051 -> 50051
```

### 4. Configure `.env`

Create `data/.env` with:

```env
# Weaviate (port-forwarded for local dev)
WEAVIATE_ENDPOINT=http://localhost:8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_API_KEY=<your-weaviate-api-key>
WEAVIATE_COLLECTION_WEBSITE=StarLearnersWebsite
WEAVIATE_COLLECTION_FRAME=StarLearnersFrame

# Vertex AI (used for embeddings and captioning — no GOOGLE_API_KEY needed)
GCP_PROJECT=tridorian-sg-vertex-ai
GCP_LOCATION=us-central1

# Model overrides (optional)
GEMINI_TEXT_EMBED_MODEL=gemini-embedding-001
GEMINI_IMAGE_EMBED_MODEL=multimodalembedding@001
GEMINI_CAPTION_MODEL=gemini-2.5-flash
```

> **Note:** Embeddings and captioning use Vertex AI credentials (ADC via `gcloud auth application-default login`). No `GOOGLE_API_KEY` is required.

Make sure ADC is set up:

```bash
gcloud auth application-default login
```

## Ingestion Commands

Run from the **repo root** with port-forward active:

```bash
# Full ingest (websites + YouTube frames)
python data/build_weaviate_index.py --mode all --recreate-collection

# Website text only
python data/build_weaviate_index.py --mode websites

# YouTube frames only (downloads, captions, embeds, uploads)
python data/build_weaviate_index.py --mode youtube

# Two-phase YouTube ingest:
# Phase 1 — extract frames + captions + embeddings to local JSONL (slow, resumable)
python data/build_weaviate_index.py --mode extract-frames

# Phase 2 — upload pre-extracted JSONL to Weaviate (fast)
python data/build_weaviate_index.py --mode upload-frames
```

Optional flags:

```bash
python data/build_weaviate_index.py \
  --mode all \
  --frame-interval-sec 5 \
  --batch-size 32 \
  --recreate-collection \
  --frames-dir data/frames \
  --frames-index data/frames/frames_index.jsonl
```

## Query Commands

```bash
python data/query_weaviate.py --query "show me tour demo classroom" --top-k 5
python data/query_weaviate.py --query "infant care programme" --top-k 5 --source-type website
python data/query_weaviate.py --query "show the tour video" --top-k 5 --source-type youtube
```

Output format:
- `query`
- `results[]` with fields:
  - `score` (1 - cosine_distance)
  - `source_type` (`website_chunk` or `youtube_frame`)
  - `content_preview`
  - `url`
  - `video_id`
  - `timestamp_sec`
  - `timestamp_hms`
  - `youtube_deeplink`

## Notes

- `build_weaviate_index.py` uses deterministic `doc_id` UUID hashes, so reruns upsert without creating duplicates.
- Script is resilient to per-item failures and prints a final JSON summary.
- YouTube query path uses `multimodalembedding@001` text→image-space embedding. Falls back gracefully if unavailable.
- For Cloud Run deployment, set `WEAVIATE_ENDPOINT=http://10.10.0.3:8080` (internal LB, reachable via VPC egress).
