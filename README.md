# Gemini Workshop

A hands-on workshop app for exploring the Google Gemini API. Build, experiment, and learn about LLMs in 65 minutes.

**Features:** Model explorer, parameter playground, SSE streaming, tool calling (calculator + doc search + web search), RAG with local vector DB, grounded answers with citations, and a debug panel showing raw requests/responses.

## Prerequisites

- Python 3.11+ (managed via [uv](https://docs.astral.sh/uv/))
- [uv](https://docs.astral.sh/uv/) package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier works)
- ~500 MB disk space (for ChromaDB + dependencies)

## Quick Start

### One-command run

```bash
./run.sh
```

This creates a Python 3.11 virtual environment via uv, installs dependencies, and starts the server at **http://127.0.0.1:8000**.

### Manual setup

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### One-command tests

```bash
./test.sh
```

For integration tests (hit the real API):

```bash
GEMINI_API_KEY=your_key_here ./test.sh
```

## Project Structure

```
gemini-workshop/
├── backend/
│   ├── config.py       # All configuration in one place
│   ├── main.py         # FastAPI endpoints
│   ├── rag.py          # RAG pipeline (chunk, embed, store, retrieve)
│   └── tools.py        # Tool definitions (calc, search_docs, web_search)
├── static/
│   └── index.html      # Single-page UI (oat.css, vanilla JS)
├── sample_docs/        # Pre-loaded documents for RAG
│   ├── python_basics.md
│   ├── machine_learning.md
│   └── web_development.md
├── tests/
│   └── test_app.py     # pytest tests (mock + integration)
├── requirements.txt
├── run.sh              # One-command run
├── test.sh             # One-command test
├── LICENSE             # MIT
└── CONTRIBUTING.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List available Gemini models (live from API) |
| GET | `/api/model-card` | Get model card link by model name |
| POST | `/api/chat` | Chat with streaming (SSE) or non-streaming |
| POST | `/api/rag/ingest` | Upload and ingest a document |
| POST | `/api/rag/ingest-samples` | Ingest all sample docs |
| POST | `/api/rag/query` | RAG query with citations |
| GET | `/api/rag/collections` | List vector DB collections |
| POST | `/api/tools/execute` | Execute a tool manually |
| GET | `/api/tools/schemas` | Get tool schemas |

## Workshop Exercises (65 minutes)

### Exercise 1: Parameter Sweeps (10 min)
**Goal:** Understand how generation parameters affect output.

1. Go to **Setup** tab, enter your API key, and select `gemini-2.5-flash`.
2. Go to **Chat** tab. Set system prompt: `"You are a creative storyteller."`
3. Send: `"Write a short story about a robot learning to paint."`
4. Now change **Temperature** to 0.1 and send again. Compare outputs.
5. Try Temperature=2.0. What happens?
6. Experiment with **Top-K** (1 vs 40 vs 100) and **Top-P** (0.1 vs 0.95).

**Expected:** Low temperature = deterministic, focused output. High temperature = creative, sometimes incoherent. Low top-k = very constrained vocabulary.

### Exercise 2: Structured Output & Safety (10 min)
**Goal:** Learn about response formats and safety settings.

1. Set **Response MIME Type** to `JSON`.
2. System prompt: `"Return responses as JSON with keys: answer, confidence (0-1), reasoning."`
3. Ask: `"What is the capital of France?"`
4. Observe the structured JSON response.
5. Change **Safety Level** to `None`, then `High`. Try a borderline prompt and see how filtering changes.
6. Add a **Stop Sequence**: `"END"`. System prompt: `"Always end your response with the word END."` See it get cut off.

**Expected:** JSON mime type forces structured output. Safety settings visibly block or allow content.

### Exercise 3: Tool Calling (15 min)
**Goal:** See how models use tools transparently.

1. Go to **Tools** tab. Read the tool schemas.
2. In **Manual Tool Test**, try `calc` with `{"expression": "sqrt(144) + 2**10"}`.
3. Go to **Chat** tab. Check **Enable Tools**.
4. Ask: `"What is 15% of 847.50? Use the calculator."`
5. Watch the tool call appear in the output and in the **Tool Call Log**.
6. Ask: `"Search my docs for information about neural networks."` (ingest samples first)
7. Check **Enable Web Search** and ask: `"What's the weather in London right now?"`

**Expected:** The model generates a function call, you see the schema match, the execution result, and the model's final answer incorporating the tool output.

### Exercise 4: RAG Pipeline (20 min)
**Goal:** Build and test retrieval-augmented generation.

1. Go to **RAG** tab. Click **Ingest Sample Docs**. Watch the chunk count.
2. Ask: `"What is the Transformer architecture?"` with **Grounded mode** on.
3. Observe the answer cites specific chunk IDs. Check **Retrieved Chunks** panel.
4. Ask something NOT in the docs: `"What is quantum computing?"` with grounded mode.
5. The model should say "I don't know based on the provided context."
6. Uncheck grounded mode and ask again - now it uses its own knowledge.
7. Upload your own `.txt` or `.md` file and query it.

**Expected:** Grounded mode constrains answers to retrieved context. Citations trace back to specific chunks. Out-of-scope questions get honest "I don't know" responses.

### Exercise 5: Evaluation & Comparison (10 min)
**Goal:** Compare model behavior and evaluate RAG quality.

1. Try the same RAG question with different models (e.g., `gemini-2.5-flash` vs `gemini-2.0-flash`).
2. Compare answer quality, citation accuracy, and token usage.
3. Change **Top-K** in RAG settings (1 vs 5 vs 10). How does retrieval count affect answer quality?
4. Check the **Debug** tab to see raw request/response payloads.
5. Try: Does the grounded answer actually only use information from the retrieved chunks? (Manual groundedness check)

**Expected:** More retrieved chunks generally improve answers but increase cost. Different models have different strengths. Debug view reveals the full API interaction.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your venv |
| "API key required" error | Enter your key in the Setup tab and click Save |
| Models list empty | Check your API key is valid at [AI Studio](https://aistudio.google.com/) |
| ChromaDB errors | Delete `chroma_db/` folder and re-ingest |
| Port 8000 in use | Kill the process: `lsof -ti:8000 \| xargs kill` or change PORT in `backend/config.py` |
| Streaming not working | Make sure you're using a model that supports `generateContent` |
| Import errors | Make sure you run from the project root directory |
| Rate limit errors | The app has built-in retry logic. Wait a moment and try again |

## Security

- API keys are stored **only** in your browser's `localStorage`.
- Keys are **never** logged, stored on disk, or sent to any server other than Google's API.
- The calculator tool uses safe AST parsing - no arbitrary code execution.
- File uploads are limited to the `uploads/` folder.

## References

- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Gemini Models Reference](https://ai.google.dev/gemini-api/docs/models)
- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Oat CSS Framework](https://github.com/knadh/oat)
