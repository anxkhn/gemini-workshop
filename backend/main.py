"""FastAPI backend for the Gemini Workshop app."""

import asyncio
import json
import logging
import os
import time
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import (
    BASE_DIR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MAX_RETRIES,
    MODEL_CARD_BASE,
    MODEL_CARDS,
    RETRY_DELAY,
    SAMPLE_DOCS_DIR,
    UPLOAD_DIR,
)
from backend.rag import (
    build_rag_prompt,
    delete_collection,
    ingest_file,
    ingest_text,
    list_collections,
    query_collection,
)
from backend.tools import ALL_TOOLS, TOOL_SCHEMAS, execute_tool

# ---------------------------------------------------------------------------
# Logging (never log API keys)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("workshop")


def _redact(key: str) -> str:
    if not key:
        return ""
    return key[:4] + "..." + key[-4:] if len(key) > 8 else "****"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Gemini Workshop", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client(api_key: str):
    """Create a Gemini client. Key is never logged."""
    from google import genai
    logger.info("Creating Gemini client (key=%s)", _redact(api_key))
    return genai.Client(api_key=api_key)


async def _retry_async(coro_fn, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Retry an async function with exponential backoff."""
    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_fn()
        except Exception as exc:
            last_exc = exc
            wait = delay * (2 ** attempt)
            logger.warning("Attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, exc, wait)
            await asyncio.sleep(wait)
    raise last_exc


def _retry_sync(fn, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Retry a sync function with exponential backoff."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = delay * (2 ** attempt)
            logger.warning("Attempt %d failed: %s. Retrying in %.1fs...", attempt + 1, exc, wait)
            time.sleep(wait)
    raise last_exc


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelsResponse(BaseModel):
    models: list[dict]


@app.get("/api/models")
async def list_models(api_key: str = ""):
    """List available Gemini models."""
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required. Pass ?api_key=YOUR_KEY")
    try:
        client = _get_client(api_key)
        models = []
        for m in _retry_sync(lambda: list(client.models.list())):
            info = {
                "name": m.name if hasattr(m, "name") else str(m),
                "display_name": m.display_name if hasattr(m, "display_name") else "",
                "description": m.description if hasattr(m, "description") else "",
                "input_token_limit": m.input_token_limit if hasattr(m, "input_token_limit") else None,
                "output_token_limit": m.output_token_limit if hasattr(m, "output_token_limit") else None,
                "supported_actions": [],
            }
            if hasattr(m, "supported_generation_methods"):
                info["supported_actions"] = m.supported_generation_methods or []
            models.append(info)
        return {"models": models}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error listing models: %s", exc)
        raise HTTPException(status_code=502, detail=f"Failed to list models: {exc}")


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

@app.get("/api/model-card")
async def get_model_card(model: str = ""):
    """Return the best-effort model card link for a given model name."""
    if not model:
        raise HTTPException(status_code=400, detail="Provide ?model=models/gemini-...")

    model_lower = model.lower()
    card_link = MODEL_CARD_BASE
    family = "unknown"

    for prefix, link in MODEL_CARDS.items():
        if prefix in model_lower:
            card_link = link
            family = prefix
            break

    return {
        "model": model,
        "family": family,
        "card_link": card_link,
        "note": "Visit the link for full model card, capabilities, and limitations.",
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    api_key: str
    model: str = DEFAULT_MODEL
    system_prompt: str = ""
    messages: list[dict] = []
    user_message: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_output_tokens: int = DEFAULT_MAX_TOKENS
    stop_sequences: list[str] = []
    safety_level: str = "default"
    response_mime_type: str = ""
    stream: bool = True
    enable_tools: bool = False
    enable_web_search: bool = False


SAFETY_PRESETS = {
    "none": "BLOCK_NONE",
    "low": "BLOCK_ONLY_HIGH",
    "default": "BLOCK_MEDIUM_AND_ABOVE",
    "high": "BLOCK_LOW_AND_ABOVE",
}


def _build_safety_settings(level: str):
    from google.genai import types
    threshold = SAFETY_PRESETS.get(level, "BLOCK_MEDIUM_AND_ABOVE")
    categories = [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    ]
    return [types.SafetySetting(category=c, threshold=threshold) for c in categories]


def _build_config(req: ChatRequest):
    from google.genai import types

    config_kwargs = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "max_output_tokens": req.max_output_tokens,
        "safety_settings": _build_safety_settings(req.safety_level),
    }

    if req.stop_sequences:
        config_kwargs["stop_sequences"] = req.stop_sequences

    if req.response_mime_type:
        config_kwargs["response_mime_type"] = req.response_mime_type

    if req.system_prompt:
        config_kwargs["system_instruction"] = req.system_prompt

    if req.enable_tools:
        tools = [ALL_TOOLS]
        if req.enable_web_search:
            from google.genai import types as gtypes
            tools.append(gtypes.Tool(google_search=gtypes.GoogleSearch()))
        config_kwargs["tools"] = tools
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

    return types.GenerateContentConfig(**config_kwargs)


def _build_contents(req: ChatRequest):
    from google.genai import types

    contents = []
    for msg in req.messages:
        role = msg.get("role", "user")
        text = msg.get("text", "")
        parts = msg.get("parts", None)
        if parts:
            # Already structured parts (for tool call flow)
            content_parts = []
            for p in parts:
                if "text" in p:
                    content_parts.append(types.Part.from_text(text=p["text"]))
                elif "function_call" in p:
                    content_parts.append(types.Part(function_call=types.FunctionCall(
                        name=p["function_call"]["name"],
                        args=p["function_call"]["args"],
                    )))
                elif "function_response" in p:
                    content_parts.append(types.Part.from_function_response(
                        name=p["function_response"]["name"],
                        response=p["function_response"]["response"],
                    ))
            contents.append(types.Content(role=role, parts=content_parts))
        else:
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    if req.user_message:
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=req.user_message)],
        ))

    return contents


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat endpoint with optional streaming and tool calling."""
    if not req.api_key:
        raise HTTPException(status_code=400, detail="API key required")

    client = _get_client(req.api_key)
    config = _build_config(req)
    contents = _build_contents(req)

    # Build raw request payload for debug (redact key)
    raw_request = {
        "model": req.model,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "max_output_tokens": req.max_output_tokens,
        "stop_sequences": req.stop_sequences,
        "safety_level": req.safety_level,
        "response_mime_type": req.response_mime_type,
        "enable_tools": req.enable_tools,
        "system_prompt": req.system_prompt[:200] + "..." if len(req.system_prompt) > 200 else req.system_prompt,
        "message_count": len(contents),
    }

    if req.stream:
        return StreamingResponse(
            _stream_response(client, req.model, contents, config, raw_request, req),
            media_type="text/event-stream",
        )
    else:
        try:
            response = _retry_sync(
                lambda: client.models.generate_content(
                    model=req.model, contents=contents, config=config
                )
            )
            result = _parse_response(response)
            result["raw_request"] = raw_request
            return result
        except Exception as exc:
            logger.error("Chat error: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))


async def _stream_response(client, model, contents, config, raw_request, req):
    """Generator for SSE streaming."""
    # Send raw request as first event
    yield f"data: {json.dumps({'type': 'debug', 'raw_request': raw_request})}\n\n"

    try:
        stream = _retry_sync(
            lambda: client.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
        )

        full_text = ""
        for chunk in stream:
            # Check for tool calls
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_call_data = {
                            "type": "tool_call",
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }
                        yield f"data: {json.dumps(tool_call_data)}\n\n"

                        # Execute the tool
                        tool_result = execute_tool(fc.name, dict(fc.args) if fc.args else {}, req.api_key)
                        tool_result_data = {
                            "type": "tool_result",
                            "name": fc.name,
                            "result": tool_result,
                        }
                        yield f"data: {json.dumps(tool_result_data)}\n\n"
                        continue

                    if hasattr(part, "text") and part.text:
                        full_text += part.text
                        yield f"data: {json.dumps({'type': 'text', 'text': part.text})}\n\n"

            # Usage metadata
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage = {}
                if hasattr(chunk.usage_metadata, "prompt_token_count"):
                    usage["prompt_tokens"] = chunk.usage_metadata.prompt_token_count
                if hasattr(chunk.usage_metadata, "candidates_token_count"):
                    usage["completion_tokens"] = chunk.usage_metadata.candidates_token_count
                if hasattr(chunk.usage_metadata, "total_token_count"):
                    usage["total_tokens"] = chunk.usage_metadata.total_token_count
                if usage:
                    yield f"data: {json.dumps({'type': 'usage', 'usage': usage})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'full_text': full_text})}\n\n"

    except Exception as exc:
        logger.error("Stream error: %s\n%s", exc, traceback.format_exc())
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"


def _parse_response(response) -> dict:
    """Parse a non-streaming response into a dict."""
    result: dict = {"text": "", "tool_calls": [], "usage": {}}

    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                result["tool_calls"].append({
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args) if part.function_call.args else {},
                })
            elif hasattr(part, "text") and part.text:
                result["text"] += part.text

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        result["usage"] = {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "completion_tokens": getattr(um, "candidates_token_count", None),
            "total_tokens": getattr(um, "total_token_count", None),
        }

    # Raw response for debug
    try:
        result["raw_response"] = json.loads(response.model_dump_json()) if hasattr(response, "model_dump_json") else str(response)
    except Exception:
        result["raw_response"] = str(response)

    return result


# ---------------------------------------------------------------------------
# RAG endpoints
# ---------------------------------------------------------------------------

class RagQueryRequest(BaseModel):
    api_key: str
    query: str
    model: str = DEFAULT_MODEL
    top_k: int = 5
    grounded: bool = True
    temperature: float = 0.3
    max_output_tokens: int = DEFAULT_MAX_TOKENS


@app.post("/api/rag/ingest")
async def rag_ingest(
    api_key: str = Form(...),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    source: Optional[str] = Form("pasted_text"),
):
    """Ingest a file or pasted text into the vector DB."""
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")

    try:
        if file:
            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            result = _retry_sync(lambda: ingest_file(file_path, api_key))
        elif text:
            result = _retry_sync(lambda: ingest_text(text, source, api_key))
        else:
            raise HTTPException(status_code=400, detail="Provide a file or text to ingest")

        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingest error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}")


@app.post("/api/rag/ingest-samples")
async def rag_ingest_samples(api_key: str = Form(...)):
    """Ingest all sample docs from the sample_docs/ directory."""
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")

    results = []
    for fname in sorted(os.listdir(SAMPLE_DOCS_DIR)):
        if fname.endswith((".md", ".txt")):
            fpath = os.path.join(SAMPLE_DOCS_DIR, fname)
            try:
                r = _retry_sync(lambda fp=fpath: ingest_file(fp, api_key))
                results.append(r)
            except Exception as exc:
                results.append({"status": "error", "source": fname, "error": str(exc)})
    return {"results": results}


@app.post("/api/rag/query")
async def rag_query(req: RagQueryRequest):
    """Retrieve relevant chunks and generate an answer."""
    if not req.api_key:
        raise HTTPException(status_code=400, detail="API key required")

    try:
        # Retrieve
        chunks = _retry_sync(lambda: query_collection(req.query, req.api_key, req.top_k))

        if not chunks:
            return {
                "answer": "No documents found. Please ingest some documents first.",
                "chunks": [],
                "usage": {},
            }

        # Build prompt
        prompt = build_rag_prompt(req.query, chunks, grounded=req.grounded)

        # Generate
        from google.genai import types

        client = _get_client(req.api_key)
        config = types.GenerateContentConfig(
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
        )
        response = _retry_sync(
            lambda: client.models.generate_content(
                model=req.model, contents=prompt, config=config
            )
        )

        answer = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    answer += part.text

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", None),
                "completion_tokens": getattr(um, "candidates_token_count", None),
                "total_tokens": getattr(um, "total_token_count", None),
            }

        return {"answer": answer, "chunks": chunks, "usage": usage}

    except Exception as exc:
        logger.error("RAG query error: %s", exc)
        raise HTTPException(status_code=502, detail=f"RAG query failed: {exc}")


@app.get("/api/rag/collections")
async def rag_collections():
    """List all collections."""
    try:
        return {"collections": list_collections()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/rag/collections/{name}")
async def rag_delete_collection(name: str):
    """Delete a collection."""
    try:
        return delete_collection(name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Tool execute (for manual tool calling flow)
# ---------------------------------------------------------------------------

class ToolExecuteRequest(BaseModel):
    api_key: str = ""
    tool_name: str
    args: dict


@app.post("/api/tools/execute")
async def tools_execute(req: ToolExecuteRequest):
    """Execute a tool and return its result."""
    try:
        result = execute_tool(req.tool_name, req.args, req.api_key)
        return {"tool": req.tool_name, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/tools/schemas")
async def tools_schemas():
    """Return tool schemas for display."""
    schemas = []
    for decl in TOOL_SCHEMAS:
        schemas.append({
            "name": decl.name,
            "description": decl.description,
            "parameters": decl.parameters_json_schema,
        })
    return {"tools": schemas}
