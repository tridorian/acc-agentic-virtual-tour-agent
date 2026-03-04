"""
FastAPI application for Star Learners AI.
Integrates Vertex AI RAG for knowledge-base retrieval.
"""

import asyncio
import base64
import json
import logging
import warnings
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi import Body
from pydantic import BaseModel
import requests
import tempfile
import re
from urllib.parse import urlparse

# server-side Text-to-Speech
from google.cloud import texttospeech

# ADK and GenAI Imports
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Vertex AI RAG Imports
import vertexai
from vertexai.preview import rag

# ========================================
# Phase 0: Environment & SDK Initialization
# ========================================
load_dotenv()

# Force Vertex AI Mode for the GenAI SDK
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "tridorian-sg-vertex-ai")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")  # Must match corpus region

# Initialize Vertex AI for both RAG and GenAI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Import agent AFTER environment setup
from google_search_agent.agent import agent

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

class URLImportRequest(BaseModel):
    url: str

APP_NAME = "star-learners-bidi"

# ========================================
# Phase 1: Application Setup
# ========================================
app = FastAPI()

static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

# ========================================
# Phase 2: Root Endpoint
# ========================================
@app.get("/")
async def root():
    """Serve the main landing page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# Debug endpoint to check agent configuration
@app.get("/api/agent-config")
async def get_agent_config():
    """Return agent configuration info including model and voice settings."""
    return {
        "agent_name": agent.name,
        "model": os.getenv("DEMO_AGENT_MODEL", "gemini-live-2.5-flash-native-audio"),
        "tts_service": "Google Vertex AI Text-to-Speech (via gemini-live native-audio)",
        "voice_info": "The agent uses Gemini's native audio voice, which is Google's premium neural voice synthesized via Vertex AI",
        "recommendation": "To match the agent voice exactly, use Google Cloud Text-to-Speech API with voice-id 'Kore' or the corresponding premium neural voice",
        "note": "The exact voice parameters are handled by the Gemini Live API and not directly configurable"
    }


# ---------------------------------------------------------------------------
# RAG corpus management endpoints (not part of the chat websocket)
# these allow external tooling or a separate UI to manage the knowledge base.
# ---------------------------------------------------------------------------


def _corpus_path() -> str:
    path = os.getenv("RAG_CORPUS_PATH")
    if not path:
        raise RuntimeError("RAG_CORPUS_PATH environment variable is not set")
    return path


# ---------------------------------------------------------------------------
# Text-to-Speech helper and endpoint
# ---------------------------------------------------------------------------

def _synthesize_text_to_mp3(text: str, voice_name: str = "Kore", language_code: str = "en-US") -> bytes:
    """Helper that calls Google Cloud TTS and returns MP3 bytes."""
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    return response.audio_content


@app.post("/api/tts")
async def synthesize_speech(body: dict = Body(...)):
    """Endpoint to synthesize speech.

    Expects JSON {"text": ..., "voice_name": <optional>, "language_code": <optional>}. Returns
    {"audio": "<base64 string>", "mimeType": "audio/mp3"}.
    """
    text = body.get("text", "")
    if not text:
        return {"error": "text field is required"}
    voice_name = body.get("voice_name", "Kore")
    language_code = body.get("language_code", "en-US")
    try:
        mp3_bytes = _synthesize_text_to_mp3(text, voice_name=voice_name, language_code=language_code)
        b64_audio = base64.b64encode(mp3_bytes).decode("utf-8")
        return {"audio": b64_audio, "mimeType": "audio/mp3"}
    except Exception as e:
        logger.error("TTS synthesis failed", exc_info=e)
        return {"error": str(e)}



@app.post("/upload")
async def upload_url(payload: dict):
    """Import a public URL into the configured RAG corpus.

    Payload should be JSON ``{"url": "https://…"}``.
    Fetches the URL, saves it as an HTML file, and uploads to RAG corpus.
    The response is asynchronous; indexing typically completes within a few minutes.
    """
    url = payload.get("url")
    if not url:
        return {"error": "missing url"}
    
    try:
        # 1. Fetch the content
        headers = {"User-Agent": "Mozilla/5.0 (Vertex-AI-RAG-Tool)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 2. Generate a clean filename from the URL
        parsed = urlparse(url)
        clean_name = re.sub(r"[^a-zA-Z0-9]", "_", parsed.netloc + parsed.path).strip("_")
        filename = f"{clean_name if clean_name else 'web_page'}.html"
        
        # 3. Create a temporary file and upload it
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, filename)
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            # Use upload_file for direct ingestion from a local source
            rag.upload_file(
                corpus_name=_corpus_path(),
                path=temp_file_path,
                display_name=url,
            )
        
        return {
            "status": "upload_completed",
            "url": url,
            "filename": filename,
            "message": f"Web page '{url}' scraped and uploaded successfully.",
        }
    except Exception as e:
        logger.error("RAG import error", exc_info=e)
        return {"error": str(e)}


@app.post("/kb/create")
async def create_corpus(payload: dict):
    name = payload.get("name")
    if not name:
        return {"error": "missing name"}
    corpus = rag.create_corpus(display_name=name)
    return {"corpus_name": corpus.name}


@app.get("/kb/files")
async def list_corpus_files():
    try:
        files = rag.list_files(corpus_name=_corpus_path())
        return {"files": [{"id": f.name, "source": f.display_name} for f in files]}
    except Exception as e:
        return {"error": str(e)}


@app.post("/kb/delete_file")
async def delete_file(payload: dict):
    file_id = payload.get("file_id")
    if not file_id:
        return {"error": "missing file_id"}
    try:
        rag.delete_file(name=file_id)
        return {"status": "deleted", "file_id": file_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/kb/delete_corpus")
async def delete_corpus(payload: dict):
    confirm = payload.get("confirm")
    if confirm != "DELETE":
        return {"error": "please send {'confirm':'DELETE'} to proceed"}
    try:
        rag.delete_corpus(name=_corpus_path())
        return {"status": "corpus_deleted"}
    except Exception as e:
        return {"error": str(e)}

# ========================================
# Phase 3: WebSocket Live API Endpoint
# ========================================
@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    await websocket.accept()
    logger.info(f"WebSocket session connected: {user_id}/{session_id}")

    # Configuration for Native Audio Model (optimized for Gemini Live)
    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"  # Change to "Puck", "Kore", or "Fenrir"
                )
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        session_resumption=types.SessionResumptionConfig(),
    )

    # Ensure session exists
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    live_request_queue = LiveRequestQueue()

    async def upstream_task() -> None:
        """Handle incoming Browser -> Server messages."""
        while True:
            try:
                message = await websocket.receive()
                
                # Handle microphone audio (16-bit PCM, 16kHz)
                if "bytes" in message:
                    audio_blob = types.Blob(
                        mime_type="audio/pcm;rate=16000", 
                        data=message["bytes"]
                    )
                    live_request_queue.send_realtime(audio_blob)
                
                # Handle text messages or image frames
                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == "text":
                        content = types.Content(
                            parts=[types.Part(text=data["text"])]
                        )
                        live_request_queue.send_content(content)
                    elif data.get("type") == "image":
                        image_data = base64.b64decode(data["data"])
                        image_blob = types.Blob(
                            mime_type=data.get("mimeType", "image/jpeg"), 
                            data=image_data
                        )
                        live_request_queue.send_realtime(image_blob)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Upstream Error: {e}")
                break

    async def downstream_task() -> None:
        """Handle outgoing Server -> Browser events (Audio, Transcripts, Tools)."""
        try:
            async for event in runner.run_live(
                user_id=user_id,
                session_id=session_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                # Send raw ADK event as JSON string
                await websocket.send_text(event.model_dump_json(exclude_none=True, by_alias=True))
        except Exception as e:
            logger.error(f"Downstream Error: {e}")

    # Run tasks concurrently until disconnect
    try:
        await asyncio.gather(upstream_task(), downstream_task())
    finally:
        logger.info("Closing Live API stream and queue")
        live_request_queue.close()

# ========================================
# Phase 4: Start the Server
# ========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
