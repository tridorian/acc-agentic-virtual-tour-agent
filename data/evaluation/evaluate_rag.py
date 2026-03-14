#!/usr/bin/env python3
"""RAG evaluation script for Star Learners Bidi Agent.

Runs each question from questions.csv through the full RAG pipeline,
evaluates retrieval + answer quality using Gemini LLM-as-judge,
and outputs per-question scores and a summary table.

Usage:
    python data/evaluation/evaluate_rag.py
    python data/evaluation/evaluate_rag.py --top-k 5 --output-dir data/evaluation/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load app/.env before importing any google/weaviate modules
_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP_ENV = _REPO_ROOT / "app" / ".env"
from dotenv import load_dotenv
load_dotenv(_APP_ENV)

from google import genai
from weaviate.classes.query import Filter, MetadataQuery


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_TOP_K = 5
_JUDGE_MODEL = "gemini-2.5-flash"
_GENERATE_MODEL = "gemini-2.5-flash"
_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "StarLearnersKB")
_QUESTIONS_FILE = Path(__file__).parent / "questions.csv"

_RERANK_MODEL = "semantic-ranker-fast-004"
_RERANK_THRESHOLD = 0.5  # Normalized chunks scoring below this are dropped

_LLM_JUDGE_PROMPT = """\
You are evaluating the quality of a RAG (retrieval-augmented generation) system.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER: {answer}

Score each metric from 0.0 to 1.0 and return ONLY a JSON object with this exact structure:
{{
  "context_relevance": <float 0-1>,
  "context_recall": <float 0-1>,
  "answer_faithfulness": <float 0-1>,
  "answer_relevance": <float 0-1>,
  "answer_completeness": <float 0-1>,
  "reasoning": "<brief 1-2 sentence explanation>"
}}

Metric definitions:
- context_relevance: Are the retrieved chunks relevant to the question? (avg across chunks)
- context_recall: Do the chunks collectively cover all aspects of the question?
- answer_faithfulness: Are all answer claims grounded in the retrieved context? (anti-hallucination)
- answer_relevance: Does the answer directly address the question?
- answer_completeness: Does the answer fully address all aspects of the question?

Return ONLY the JSON object, no markdown fences, no extra text.
"""

_GENERATE_PROMPT = """\
You are Stella, a friendly AI assistant for Star Learners childcare.
Use the context below to answer the question. Be concise and helpful.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ---------------------------------------------------------------------------
# Client singletons
# ---------------------------------------------------------------------------
_genai_client: Optional[genai.Client] = None
_weaviate_client = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GCP_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("Missing GCP_PROJECT / GOOGLE_CLOUD_PROJECT env var")
        _genai_client = genai.Client(vertexai=True, project=project, location=location)
    return _genai_client


def _get_weaviate_client():
    global _weaviate_client
    if _weaviate_client is None:
        import weaviate
        from weaviate.connect import ConnectionParams
        from urllib.parse import urlparse

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
        _weaviate_client = client
    return _weaviate_client


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def embed_query(question: str) -> List[float]:
    client = _get_genai_client()
    response = client.models.embed_content(model=_EMBED_MODEL, contents=[question])
    return list(response.embeddings[0].values)


def search_source(
    vector: List[float],
    source_type: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    wc = _get_weaviate_client()
    collection = wc.collections.get(_COLLECTION)
    result = collection.query.near_vector(
        near_vector=vector,
        limit=top_k,
        filters=Filter.by_property("source_type").equal(source_type),
        return_metadata=MetadataQuery(distance=True),
    )
    chunks = []
    for obj in result.objects:
        props = obj.properties
        distance = obj.metadata.distance
        score = float(1.0 - distance) if distance is not None else 0.0
        chunk: Dict[str, Any] = {
            "score": score,
            "source_type": source_type,
            "content": str(props.get("content", ""))[:600],
        }
        if source_type == "youtube_frame":
            vid = props.get("video_id")
            ts = props.get("timestamp_sec")
            chunk["video_id"] = vid
            chunk["timestamp_sec"] = ts
            chunk["timestamp_hms"] = str(props.get("timestamp_hms", ""))
            if vid and ts is not None:
                chunk["youtube_deeplink"] = f"https://www.youtube.com/watch?v={vid}&t={int(ts)}s"
        else:
            chunk["source_url"] = str(props.get("source_url", ""))
        chunks.append(chunk)
    return chunks


def retrieve(question: str, top_k: int) -> Dict[str, Any]:
    vector = embed_query(question)
    website_chunks = search_source(vector, "website", top_k)
    video_chunks = search_source(vector, "youtube_frame", top_k)
    all_chunks = website_chunks + video_chunks
    return {
        "website_chunks": website_chunks,
        "video_chunks": video_chunks,
        "all_chunks": all_chunks,
    }


# ---------------------------------------------------------------------------
# LLM Reranking
# ---------------------------------------------------------------------------

def rerank_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    threshold: float = _RERANK_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Rerank chunks using Google Discovery Engine Ranking API (semantic-ranker-fast-004).

    Scores are normalized to 0–1 then filtered by threshold.
    Returns all chunks unchanged on any error.
    Always keeps at least 1 chunk to avoid empty context.
    """
    if not chunks:
        return chunks

    try:
        from google.cloud import discoveryengine_v1 as discoveryengine

        project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError("Missing GCP_PROJECT / GOOGLE_CLOUD_PROJECT env var")

        rank_client = discoveryengine.RankServiceClient()
        ranking_config = rank_client.ranking_config_path(
            project=project,
            location="global",
            ranking_config="default_ranking_config",
        )

        records = []
        for i, c in enumerate(chunks):
            src = c["source_type"]
            content = c["content"]
            if src == "youtube_frame":
                ts = c.get("timestamp_hms") or f"{c.get('timestamp_sec', '?')}s"
                title = f"Video frame at {ts}"
            else:
                title = c.get("source_url", "Website content")
            records.append(discoveryengine.RankingRecord(
                id=str(i),
                title=title,
                content=content,
            ))

        response = rank_client.rank(
            request=discoveryengine.RankRequest(
                ranking_config=ranking_config,
                model=_RERANK_MODEL,
                top_n=len(records),
                query=question,
                records=records,
            )
        )

        raw_scores = {rec.id: rec.score for rec in response.records}
        max_score = max(raw_scores.values()) if raw_scores else 1.0
        norm = max_score if max_score > 0 else 1.0

        scored = []
        for i, c in enumerate(chunks):
            rerank_score = float(raw_scores.get(str(i), 0.0)) / norm
            scored.append(dict(c, rerank_score=rerank_score))

    except Exception as exc:
        print(f"  [warn] Reranking failed, using raw retrieval: {exc}")
        return chunks

    kept = [c for c in scored if c["rerank_score"] >= threshold]

    if not kept:
        kept = [max(scored, key=lambda c: c["rerank_score"])]

    return kept


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_context_text(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        src = c["source_type"]
        content = c["content"]
        if src == "youtube_frame":
            ts = c.get("timestamp_hms") or f"{c.get('timestamp_sec', '?')}s"
            parts.append(f"[{i}] (Video at {ts}) {content}")
        else:
            url = c.get("source_url", "")
            parts.append(f"[{i}] (Website: {url}) {content}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    context = build_context_text(chunks)
    prompt = _GENERATE_PROMPT.format(context=context, question=question)
    client = _get_genai_client()
    response = client.models.generate_content(model=_GENERATE_MODEL, contents=[prompt])
    return response.text.strip() if getattr(response, "text", None) else ""


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluation
# ---------------------------------------------------------------------------

def evaluate_with_judge(
    question: str,
    chunks: List[Dict[str, Any]],
    answer: str,
) -> Dict[str, Any]:
    context = build_context_text(chunks)
    prompt = _LLM_JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer,
    )
    client = _get_genai_client()
    response = client.models.generate_content(model=_JUDGE_MODEL, contents=[prompt])
    text = response.text.strip() if getattr(response, "text", None) else "{}"

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        scores = {
            "context_relevance": 0.0,
            "context_recall": 0.0,
            "answer_faithfulness": 0.0,
            "answer_relevance": 0.0,
            "answer_completeness": 0.0,
            "reasoning": f"Judge parse error: {text[:200]}",
        }
    return scores


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------

def evaluate_question(
    idx: int,
    total: int,
    question: str,
    top_k: int,
) -> Dict[str, Any]:
    print(f"\n[{idx}/{total}] Q: {question}")

    # 1. Retrieve
    retrieval = retrieve(question, top_k)
    website_hits = len(retrieval["website_chunks"])
    video_hits = len(retrieval["video_chunks"])
    all_chunks = retrieval["all_chunks"]
    chunks_before = len(all_chunks)
    print(f"  Retrieved: {website_hits} website, {video_hits} video chunks ({chunks_before} total)")

    # 2. Compute objective retrieval score (pre-rerank, raw cosine similarity)
    scores_list = [c["score"] for c in all_chunks]
    retrieval_score_mean = sum(scores_list) / len(scores_list) if scores_list else 0.0

    # 3. LLM reranking — filter irrelevant chunks before generation
    reranked_chunks = rerank_chunks(question, all_chunks)
    chunks_after = len(reranked_chunks)
    rerank_website_kept = sum(1 for c in reranked_chunks if c["source_type"] == "website")
    rerank_video_kept = sum(1 for c in reranked_chunks if c["source_type"] == "youtube_frame")
    dropped = chunks_before - chunks_after
    print(
        f"  Reranked: kept {chunks_after}/{chunks_before} chunks "
        f"({rerank_website_kept} website, {rerank_video_kept} video, dropped {dropped})"
    )

    # 4. Generate answer with reranked (filtered) context
    answer = generate_answer(question, reranked_chunks)
    answer_preview = answer[:120].replace("\n", " ")
    print(f"  Answer: {answer_preview}{'...' if len(answer) > 120 else ''}")

    # 5. LLM-as-judge (evaluate against reranked context)
    judge_scores = evaluate_with_judge(question, reranked_chunks, answer)

    cr = judge_scores.get("context_relevance", 0.0)
    cc = judge_scores.get("context_recall", 0.0)
    af = judge_scores.get("answer_faithfulness", 0.0)
    ar = judge_scores.get("answer_relevance", 0.0)
    ac = judge_scores.get("answer_completeness", 0.0)
    reasoning = judge_scores.get("reasoning", "")

    print(
        f"  Scores: context_relevance={cr:.2f}, context_recall={cc:.2f}, "
        f"faithfulness={af:.2f}, answer_relevance={ar:.2f}, completeness={ac:.2f}, "
        f"retrieval_mean={retrieval_score_mean:.2f}"
    )

    return {
        "question": question,
        "answer": answer,
        "context_relevance": cr,
        "context_recall": cc,
        "answer_faithfulness": af,
        "answer_relevance": ar,
        "answer_completeness": ac,
        "retrieval_score_mean": retrieval_score_mean,
        "source_website_hits": website_hits,
        "source_video_hits": video_hits,
        "chunks_before_rerank": chunks_before,
        "chunks_after_rerank": chunks_after,
        "rerank_website_kept": rerank_website_kept,
        "rerank_video_kept": rerank_video_kept,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(rows: List[Dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"results_{timestamp}.csv"
    fieldnames = [
        "question", "answer",
        "context_relevance", "context_recall",
        "answer_faithfulness", "answer_relevance", "answer_completeness",
        "retrieval_score_mean", "source_website_hits", "source_video_hits",
        "chunks_before_rerank", "chunks_after_rerank",
        "rerank_website_kept", "rerank_video_kept",
        "reasoning",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def print_summary(rows: List[Dict[str, Any]]) -> None:
    llm_metrics = [
        "context_relevance",
        "context_recall",
        "answer_faithfulness",
        "answer_relevance",
        "answer_completeness",
        "retrieval_score_mean",
    ]
    count = len(rows)

    print("\n" + "=" * 50)
    print("=== RAG EVALUATION SUMMARY ===")
    print(f"Questions evaluated: {count}\n")

    header = f"{'Metric':<28} {'Mean':>6}  {'Min':>6}  {'Max':>6}"
    print(header)
    print("-" * 50)

    overall_scores = []
    for metric in llm_metrics:
        values = [r[metric] for r in rows if isinstance(r.get(metric), (int, float))]
        if not values:
            continue
        mean_v = sum(values) / len(values)
        min_v = min(values)
        max_v = max(values)
        print(f"{metric:<28} {mean_v:>6.2f}  {min_v:>6.2f}  {max_v:>6.2f}")
        if metric != "retrieval_score_mean":
            overall_scores.append(mean_v)

    website_avg = sum(r["source_website_hits"] for r in rows) / count
    video_avg = sum(r["source_video_hits"] for r in rows) / count
    before_avg = sum(r.get("chunks_before_rerank", 0) for r in rows) / count
    after_avg = sum(r.get("chunks_after_rerank", 0) for r in rows) / count
    rw_avg = sum(r.get("rerank_website_kept", 0) for r in rows) / count
    rv_avg = sum(r.get("rerank_video_kept", 0) for r in rows) / count
    print(f"\nSource Coverage (retrieved):")
    print(f"  website_hits avg:       {website_avg:.1f}")
    print(f"  video_hits avg:         {video_avg:.1f}")
    print(f"\nReranking (threshold={_RERANK_THRESHOLD}):")
    print(f"  chunks before rerank:   {before_avg:.1f}")
    print(f"  chunks after rerank:    {after_avg:.1f}  ({100*after_avg/before_avg:.0f}% kept)" if before_avg else "")
    print(f"  website kept avg:       {rw_avg:.1f}")
    print(f"  video kept avg:         {rv_avg:.1f}")

    if overall_scores:
        overall = sum(overall_scores) / len(overall_scores)
        print(f"\nOverall RAG Score:   {overall:.2f}  (mean of all 0-1 LLM metrics)")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(line)
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline for Star Learners")
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K, help="Chunks per source type")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for results CSV",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=_QUESTIONS_FILE,
        help="Path to questions CSV (one per line)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.questions.exists():
        print(f"ERROR: Questions file not found: {args.questions}", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(args.questions)
    if not questions:
        print("ERROR: No questions found in file.", file=sys.stderr)
        sys.exit(1)

    print(f"Star Learners RAG Evaluation")
    print(f"Questions: {len(questions)} | top_k={args.top_k} per source")
    print(f"Collection: {_COLLECTION} | Embed: {_EMBED_MODEL}")

    rows: List[Dict[str, Any]] = []
    try:
        for i, question in enumerate(questions, 1):
            row = evaluate_question(i, len(questions), question, args.top_k)
            rows.append(row)
    finally:
        # Always close Weaviate even if evaluation is interrupted
        global _weaviate_client
        if _weaviate_client is not None:
            _weaviate_client.close()
            _weaviate_client = None

    if not rows:
        print("No results to save.")
        return

    out_path = save_results(rows, args.output_dir)
    print(f"\nResults saved to: {out_path}")

    print_summary(rows)


if __name__ == "__main__":
    main()
