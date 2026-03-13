# Weaviate + Gemini Data Pipeline

This folder contains the data ingestion and retrieval pipeline for Star Learners:
- Website text chunks → `StarLearnersKB` collection (`source_type=website`)
- YouTube tour video frames → `StarLearnersKB` collection (`source_type=youtube_frame`)

Both are stored in a single unified Weaviate collection using `gemini-embedding-2-preview` (3072-dim vectors) for text and image embeddings.

## Files

- `build_weaviate_index.py` — 3-phase ETL pipeline: extract frames → embed → upload to Weaviate
- `query_weaviate.py` — CLI tool to search and return top results with YouTube deeplinks
- `sources.yaml` — website URLs and YouTube source URL
- `requirements.txt` — Python dependencies
- `.env` — environment variables (see below)
- `videos/frames/` — extracted JPEG frames per video (`{video_id}/frame_NNNN.jpg` + `index.jsonl`)
- `videos/website_objects.jsonl` — embedded website objects ready for upload
- `videos/frame_objects.jsonl` — embedded frame objects ready for upload

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
WEAVIATE_COLLECTION=StarLearnersKB

# Vertex AI (used for embeddings and captioning — ADC, no API key needed)
GCP_PROJECT=tridorian-sg-vertex-ai
GCP_LOCATION=us-central1

# Model overrides (optional — these are the defaults)
GEMINI_EMBED_MODEL=gemini-embedding-2-preview
GEMINI_CAPTION_MODEL=gemini-2.5-flash
```

> Embeddings and captioning use Vertex AI ADC credentials. Run `gcloud auth application-default login` if not already authenticated.

## Pipeline Modes

Run from the **repo root** with port-forward active:

| Mode | Description |
|---|---|
| `all` | Run all three phases sequentially |
| `extract-frames` | Download video → extract JPEG frames → write `index.jsonl` (no API calls) |
| `embed` | Embed website text + frames → write `videos/website_objects.jsonl` + `videos/frame_objects.jsonl` |
| `upload` | Upload pre-embedded JSONL files to Weaviate |
| `websites` | Website text only (embed + upload inline) |

```bash
# Full ingest (all phases at once)
python data/build_weaviate_index.py --mode all --recreate-collection

# Website text only
python data/build_weaviate_index.py --mode websites

# Run phases independently:
python data/build_weaviate_index.py --mode extract-frames
python data/build_weaviate_index.py --mode embed
python data/build_weaviate_index.py --mode upload --recreate-collection
```

Optional flags:

```bash
python data/build_weaviate_index.py \
  --mode all \
  --frame-interval-sec 5 \
  --batch-size 32 \
  --recreate-collection \
  --frames-base-dir videos/frames \
  --website-objects videos/website_objects.jsonl \
  --frame-objects videos/frame_objects.jsonl
```

## Query Commands

```bash
python data/query_weaviate.py --query "show me tour demo classroom" --top-k 5
python data/query_weaviate.py --query "infant care programme" --top-k 5 --source-type website
python data/query_weaviate.py --query "show the tour video" --top-k 5 --source-type youtube
```

Output fields per result:
- `score` (1 - cosine_distance)
- `source_type` (`website` or `youtube_frame`)
- `content_preview`
- `url`
- `video_id`, `timestamp_sec`, `timestamp_hms`, `youtube_deeplink` (frames only)

## Notes

- `build_weaviate_index.py` uses deterministic `doc_id` UUID hashes — reruns upsert without duplicates.
- Splitting into phases is useful for large videos: `extract-frames` is fast (no API calls), `embed` is slow (one API call per frame), `upload` is fast and can be retried independently if Weaviate is unavailable.
- For Cloud Run deployment, set `WEAVIATE_ENDPOINT=http://10.10.0.3:8080` (internal LB, reachable via VPC egress).
