# Star Learners Bidi Agent

An AI-powered voice assistant for Star Learners childcare centre, built with Google Gemini Live (bidirectional audio), FastAPI, and Weaviate vector search. The assistant — named **Stella** — answers questions about programmes, facilities, fees, and enrolment, and can reference relevant moments in the centre's video tour.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Step 1 — Weaviate Setup (GKE Port-Forward)](#step-1--weaviate-setup-gke-port-forward)
5. [Step 2 — Data Pipeline (Build Knowledge Base)](#step-2--data-pipeline-build-knowledge-base)
6. [Step 3 — Application Setup](#step-3--application-setup)
7. [Running the Application](#running-the-application)
8. [Cloud Run Deployment](#cloud-run-deployment)
9. [Environment Variables Reference](#environment-variables-reference)
10. [Querying the Knowledge Base Directly](#querying-the-knowledge-base-directly)

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
  │                                  ▼
  │                       Weaviate Vector DB (GKE)
  │                    ┌──────────────────────────────┐
  │                    │  StarLearnersWebsite (text)   │
  │                    │  StarLearnersFrame  (video)   │
  │                    └──────────────────────────────┘
  │
  └── REST API  ──► /api/search
```

**Key technologies:**

| Layer | Technology |
|---|---|
| Voice AI | Gemini Live 2.5 Flash (native audio) via Google ADK |
| Embeddings (text) | `gemini-embedding-001` (Vertex AI) |
| Embeddings (video frames) | `multimodalembedding@001` (Vertex AI) |
| Frame captioning | `gemini-2.5-flash` |
| Vector database | Weaviate on GKE (`weaviate-cluster`, `us-central1`) |
| Backend | FastAPI + Python |
| Frontend | Vanilla JS + WebSocket |

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
│   ├── build_weaviate_index.py     # Ingest websites + YouTube into Weaviate
│   ├── query_weaviate.py           # CLI tool to query Weaviate
│   ├── sources.yaml                # Website URLs and YouTube source
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

This step scrapes the Star Learners website and extracts frames from the YouTube tour video, then stores everything in Weaviate with text and image embeddings.

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
WEAVIATE_COLLECTION_WEBSITE=StarLearnersWebsite
WEAVIATE_COLLECTION_FRAME=StarLearnersFrame

# Vertex AI — used for embeddings and captioning (ADC, no API key needed)
GCP_PROJECT=tridorian-sg-vertex-ai
GCP_LOCATION=us-central1

# Model overrides (optional)
GEMINI_TEXT_EMBED_MODEL=gemini-embedding-001
GEMINI_IMAGE_EMBED_MODEL=multimodalembedding@001
GEMINI_CAPTION_MODEL=gemini-2.5-flash
```

Authenticate ADC if not already done:

```bash
gcloud auth application-default login
```

### 2.3 Run ingestion (port-forward must be active)

```bash
# Full ingest — websites + YouTube frames
python data/build_weaviate_index.py --mode all --recreate-collection

# Website text only
python data/build_weaviate_index.py --mode websites

# YouTube frames only
python data/build_weaviate_index.py --mode youtube

# Two-phase YouTube ingest (recommended for large videos):
# Phase 1 — extract + caption + embed frames to local JSONL (resumable)
python data/build_weaviate_index.py --mode extract-frames
# Phase 2 — upload pre-extracted JSONL to Weaviate
python data/build_weaviate_index.py --mode upload-frames
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
WEAVIATE_COLLECTION_WEBSITE=StarLearnersWebsite
WEAVIATE_COLLECTION_FRAME=StarLearnersFrame

# Embedding models — must match what was used during ingestion
GEMINI_TEXT_EMBED_MODEL=gemini-embedding-001
GEMINI_IMAGE_EMBED_MODEL=multimodalembedding@001
```

---

## Running the Application

Make sure the port-forward from Step 1.2 is active, then:

```bash
cd app
uvicorn main:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` to start a voice conversation with Stella.

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
4. Grant IAP access to `@tridorian.com` users via:
   ```yaml
   - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
     entrypoint: gcloud
     args:
       - iap
       - web
       - add-iam-policy-binding
       - --member="domain:tridorian.com"
       - --role="roles/iap.httpsResourceAccessor"
       - --region=us-central1
       - --resource-type=cloud-run
       - --service=star-learners-bidi-agent
   ```
   This ensures only users with `@tridorian.com` Google accounts can access the deployed app. The service is deployed with `--no-allow-unauthenticated` so all unauthenticated requests are rejected at the Cloud Run level.

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
| `WEAVIATE_COLLECTION_WEBSITE` | No | `StarLearnersWebsite` | Website chunks collection |
| `WEAVIATE_COLLECTION_FRAME` | No | `StarLearnersFrame` | Video frames collection |
| `GEMINI_TEXT_EMBED_MODEL` | No | `gemini-embedding-001` | Text embedding model |
| `GEMINI_IMAGE_EMBED_MODEL` | No | `multimodalembedding@001` | Image embedding model |

### Data Pipeline (`data/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `WEAVIATE_ENDPOINT` | Yes | — | Weaviate HTTP endpoint |
| `WEAVIATE_GRPC_PORT` | No | `50051` | Weaviate gRPC port |
| `WEAVIATE_API_KEY` | No | — | Weaviate API key |
| `WEAVIATE_COLLECTION_WEBSITE` | No | `StarLearnersWebsite` | Website chunks collection |
| `WEAVIATE_COLLECTION_FRAME` | No | `StarLearnersFrame` | Video frames collection |
| `GCP_PROJECT` | Yes | — | GCP project ID (Vertex AI embeddings) |
| `GCP_LOCATION` | No | `us-central1` | GCP region |
| `GEMINI_TEXT_EMBED_MODEL` | No | `gemini-embedding-001` | Text embedding model |
| `GEMINI_IMAGE_EMBED_MODEL` | No | `multimodalembedding@001` | Frame embedding model |
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
