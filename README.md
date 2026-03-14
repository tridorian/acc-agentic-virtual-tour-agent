# Star Learners Bidi Agent

An AI-powered voice assistant for Star Learners childcare centre, built with Google Gemini Live (bidirectional audio), FastAPI, and Weaviate vector search. The assistant — named **Stella** — conducts a live tour experience, answering questions about programmes, facilities, fees, and enrolment while narrating relevant moments from the centre's video tour.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Reranking Pipeline](#reranking-pipeline)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Step 1 — Weaviate Setup (GKE Port-Forward)](#step-1--weaviate-setup-gke-port-forward)
6. [Step 2 — Data Pipeline (Build Knowledge Base)](#step-2--data-pipeline-build-knowledge-base)
7. [Step 3 — Application Setup](#step-3--application-setup)
8. [Running the Application](#running-the-application)
9. [Cloud Run Deployment](#cloud-run-deployment)
10. [Environment Variables Reference](#environment-variables-reference)
11. [Querying the Knowledge Base Directly](#querying-the-knowledge-base-directly)

---

## Architecture Overview

```
Browser (WebRTC audio + chat)
        │
        ▼
FastAPI Server (app/main.py)
  ├── WebSocket  ──► Google ADK Agent (Stella)
  │                       └── search_knowledge_base()
  │                                  │
  │                          search_weaviate()
  │                        ┌─────────┴──────────┐
  │                        ▼                    ▼
  │              Weaviate (website)    Weaviate (video frames)
  │                        │                    │
  │                  _rerank_results()   _rerank_results()
  │                  (per-source)        (per-source)
  │                        └─────────┬──────────┘
  │                     cross-source gate
  │                  (drop weaker source if
  │                   best score < threshold)
  │                                  │
  │                    ┌─────────────────────────┐
  │                    │  StarLearnersKB (GKE)    │
  │                    │  source_type=website     │
  │                    │  source_type=youtube_frame│
  │                    └─────────────────────────┘
  │
  └── REST API  ──► /api/search
```

**Key technologies:**

| Layer | Technology |
|---|---|
| Voice AI | Gemini Live 2.5 Flash (native audio) via Google ADK |
| Embeddings | `gemini-embedding-2-preview` (text + images, 3072-dim) |
| Reranking | Google Discovery Engine `semantic-ranker-fast-004` |
| Frame captioning | `gemini-2.5-flash` |
| Vector database | Weaviate on GKE (`weaviate-cluster`, `us-central1`) |
| Backend | FastAPI + Python |
| Frontend | Vanilla JS + WebSocket |

---

## Reranking Pipeline

After the initial vector search, results go through a two-stage relevance filter before being returned to the agent.

**Stage 1 — Per-source reranking** (`_rerank_results`):
Each source (website text, video frames) is independently scored by [Google Discovery Engine `semantic-ranker-fast-004`](https://cloud.google.com/generative-ai-app-builder/docs/ranking). Chunks scoring below `0.5` are dropped. If *all* chunks score below the threshold, the single highest-scoring chunk is kept as a fallback (so the agent always has at least one result to reason from).

**Stage 2 — Cross-source relevance gate** (in `search_weaviate`):
After per-source filtering, the best scores from each source are compared. If one source's top score is below `0.5` **and** the other source scored higher, the weaker source is dropped entirely. This prevents the frontend from jumping to an unrelated video timestamp when only the text source has a confident match (and vice versa).

```
vector search results
  ├── website chunks  ──► rerank → drop < 0.5 (keep ≥1 fallback)
  └── video frames   ──► rerank → drop < 0.5 (keep ≥1 fallback)
                                  │
                         cross-source gate
                         if video_best < 0.5 and text_best ≥ video_best → drop video
                         if text_best  < 0.5 and video_best > text_best  → drop text
```

---

## Prerequisites

- Python 3.10+
- A Google Cloud project with **Vertex AI API** enabled
- `gcloud` CLI authenticated (`gcloud auth application-default login`)
- `kubectl` configured with access to `weaviate-cluster`

---

## Project Structure

```
star_learners_bidi_agent/
├── app/                            # FastAPI application
│   ├── main.py                     # Server entry point
│   ├── .env                        # App environment variables
│   ├── requirements.txt
│   ├── Dockerfile                  # Container image for Cloud Run
│   ├── google_search_agent/
│   │   ├── agent.py                # ADK agent definition (Stella)
│   │   └── weaviate_tool.py        # Weaviate search tool
│   └── static/                     # Frontend (HTML/CSS/JS)
│
├── data/                           # Data ingestion pipeline
│   ├── build_weaviate_index.py     # 3-phase ETL: extract → embed → upload
│   ├── query_weaviate.py           # CLI tool to query Weaviate
│   ├── sources.yaml                # Website URLs and YouTube source
│   ├── videos/                     # Extracted frames + embedded JSONL files
│   └── requirements.txt
│
├── cloudbuild.yaml                 # Cloud Build → Cloud Run deployment
└── README.md
```

---

## Step 1 — Weaviate Setup (GKE Port-Forward)

Weaviate runs on GKE cluster `weaviate-cluster` in `us-central1`, inside VPC `weaviate-vpc`. The internal load balancer (`10.10.0.3`) is only reachable from within the VPC.

For local development, use `kubectl port-forward`.

### 1.1 Get GKE credentials

```bash
gcloud container clusters get-credentials weaviate-cluster --region us-central1
```

Verify the Weaviate services:

```bash
kubectl get svc -n weaviate
```

Expected output:

```
NAME                TYPE           CLUSTER-IP    EXTERNAL-IP   PORT(S)
weaviate            ClusterIP      10.52.11.99   <none>        80/TCP
weaviate-grpc       ClusterIP      10.52.2.219   <none>        50051/TCP
weaviate-ilb        LoadBalancer   10.52.4.114   10.10.0.3     8080/TCP,50051/TCP
```

### 1.2 Start port-forward (keep open during dev)

Open a **separate terminal** and run:

```bash
kubectl port-forward svc/weaviate-ilb 8080:8080 50051:50051 -n weaviate
```

Expected output:

```
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from 127.0.0.1:50051 -> 50051
```

> Keep this terminal open. All local app and data pipeline connections go through `localhost:8080`.

### 1.3 Verify connectivity

```bash
curl http://localhost:8080/v1/meta | python3 -m json.tool | head -10
```

---

## Step 2 — Data Pipeline (Build Knowledge Base)

The pipeline ingests Star Learners website content and YouTube tour video frames into a single unified Weaviate collection (`StarLearnersKB`) using `gemini-embedding-2-preview` for both text and image embeddings.

The pipeline runs in three phases, which can be run all at once or independently:

| Mode | What it does |
|---|---|
| `extract-frames` | Download video → extract JPEG frames → write `videos/frames/{video_id}/index.jsonl` |
| `embed` | Embed website text + frame images → write `videos/website_objects.jsonl` + `videos/frame_objects.jsonl` |
| `upload` | Bulk-upload pre-embedded JSONL files into Weaviate |
| `all` | Run all three phases sequentially |
| `websites` | Website text only (embed + upload inline) |

### 2.1 Install data pipeline dependencies

```bash
# From repo root (uses project .venv managed by uv)
uv pip install -r data/requirements.txt
```

### 2.2 Configure `data/.env`

```env
# Weaviate (use localhost when port-forwarding; use 10.10.0.3:8080 on Cloud Run)
WEAVIATE_ENDPOINT=http://localhost:8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_API_KEY=<your-weaviate-api-key>
WEAVIATE_COLLECTION=StarLearnersKB

# Vertex AI — used for embeddings and captioning (ADC, no API key needed)
GCP_PROJECT=tridorian-sg-vertex-ai
GCP_LOCATION=us-central1

# Model overrides (optional — these are the defaults)
GEMINI_EMBED_MODEL=gemini-embedding-2-preview
GEMINI_CAPTION_MODEL=gemini-2.5-flash
```

Authenticate ADC if not already done:

```bash
gcloud auth application-default login
```

### 2.3 Run ingestion (port-forward must be active)

```bash
# Full ingest — extract frames → embed everything → upload to Weaviate
python data/build_weaviate_index.py --mode all --recreate-collection

# Website text only
python data/build_weaviate_index.py --mode websites

# Run phases independently (recommended for large videos):
# Phase 1 — extract JPEG frames to disk (fast, no API calls)
python data/build_weaviate_index.py --mode extract-frames

# Phase 2 — caption + embed website + frames, write JSONL files
python data/build_weaviate_index.py --mode embed

# Phase 3 — upload JSONL files to Weaviate (requires port-forward)
python data/build_weaviate_index.py --mode upload --recreate-collection
```

> Ingestion is idempotent — reruns upsert using deterministic `doc_id` hashes without duplicating data.

---

## Step 3 — Application Setup

### 3.1 Install application dependencies

```bash
uv pip install -r app/requirements.txt
```

### 3.2 Configure `app/.env`

```env
# Google Cloud / Vertex AI
GOOGLE_CLOUD_PROJECT=tridorian-sg-vertex-ai
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
GCP_PROJECT=tridorian-sg-vertex-ai
GCP_LOCATION=us-central1

# Gemini Live model
DEMO_AGENT_MODEL=gemini-live-2.5-flash-native-audio

# Weaviate (use localhost when port-forwarding; use 10.10.0.3:8080 on Cloud Run)
WEAVIATE_ENDPOINT=http://localhost:8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_API_KEY=<your-weaviate-api-key>
WEAVIATE_COLLECTION=StarLearnersKB

# Embedding model — must match what was used during ingestion
GEMINI_EMBED_MODEL=gemini-embedding-2-preview
```

---

## Running the Application

Make sure the port-forward from Step 1.2 is active, then:

```bash
cd app
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` to start a voice conversation with Stella.

**Available endpoints:**

| Endpoint | Description |
|---|---|
| `GET /` | Web frontend |
| `WS /ws/{user_id}/{session_id}` | Bidirectional audio/text stream |
| `POST /api/search` | Direct Weaviate search (JSON: `{"query": "..."}`) |

---

## Cloud Run Deployment

The app is deployed to Cloud Run inside `weaviate-vpc` so it can reach the Weaviate internal LB directly (no port-forward needed).

### Prerequisites

- Artifact Registry repository in `us-central1`
- Cloud Run service account with roles: `Vertex AI User`, `Secret Manager Secret Accessor`
- Cloud Build API enabled

### Deploy

Update `WEAVIATE_ENDPOINT` in `cloudbuild.yaml` `--set-env-vars` to the internal LB IP:

```
WEAVIATE_ENDPOINT=http://10.10.0.3:8080
```

Then submit the build:

```bash
gcloud builds submit --config cloudbuild.yaml
```

This will:
1. Build the Docker image from `app/Dockerfile`
2. Push to `gcr.io/$PROJECT_ID/star-learners-bidi-agent`
3. Deploy to Cloud Run with VPC egress into `weaviate-vpc`
4. Grant IAP access to `@tridorian.com` users — only authenticated Google accounts under that domain can access the deployed app

---

## Environment Variables Reference

### Application (`app/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | Yes | — | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | No | `us-central1` | GCP region |
| `GOOGLE_GENAI_USE_VERTEXAI` | Yes | `true` | Use Vertex AI backend |
| `GCP_PROJECT` | Yes | — | GCP project ID (embedding client) |
| `GCP_LOCATION` | No | `us-central1` | GCP region (embedding client) |
| `DEMO_AGENT_MODEL` | No | `gemini-live-2.5-flash-native-audio` | Gemini Live model |
| `WEAVIATE_ENDPOINT` | No | `http://localhost:8080` | Weaviate HTTP endpoint |
| `WEAVIATE_GRPC_PORT` | No | `50051` | Weaviate gRPC port |
| `WEAVIATE_API_KEY` | No | — | Weaviate API key |
| `WEAVIATE_COLLECTION` | No | `StarLearnersKB` | Unified knowledge base collection |
| `GEMINI_EMBED_MODEL` | No | `gemini-embedding-2-preview` | Embedding model (text + images) |

### Data Pipeline (`data/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `WEAVIATE_ENDPOINT` | Yes | — | Weaviate HTTP endpoint |
| `WEAVIATE_GRPC_PORT` | No | `50051` | Weaviate gRPC port |
| `WEAVIATE_API_KEY` | No | — | Weaviate API key |
| `WEAVIATE_COLLECTION` | No | `StarLearnersKB` | Unified knowledge base collection |
| `GCP_PROJECT` | Yes | — | GCP project ID (Vertex AI embeddings) |
| `GCP_LOCATION` | No | `us-central1` | GCP region |
| `GEMINI_EMBED_MODEL` | No | `gemini-embedding-2-preview` | Embedding model (text + images) |
| `GEMINI_CAPTION_MODEL` | No | `gemini-2.5-flash` | Frame captioning model |

---

## Querying the Knowledge Base Directly

```bash
# Search all sources
python data/query_weaviate.py --query "what programmes are available?" --top-k 5

# Search website content only
python data/query_weaviate.py --query "infant care programme fees" --top-k 5 --source-type website

# Search video frames only (returns YouTube timestamps)
python data/query_weaviate.py --query "classroom tour" --top-k 5 --source-type youtube
```

Example output for a video query:

```json
{
  "query": "classroom tour",
  "results": [
    {
      "score": 0.87,
      "source_type": "youtube_frame",
      "content_preview": "Children exploring art materials at a table...",
      "video_id": "tkhpVEcBfv0",
      "timestamp_sec": 45,
      "timestamp_hms": "00:00:45",
      "youtube_deeplink": "https://www.youtube.com/watch?v=tkhpVEcBfv0&t=45s"
    }
  ]
}
```
