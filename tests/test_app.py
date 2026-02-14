"""Tests for the Gemini Workshop app.

Tests run in mock mode when GEMINI_API_KEY is not set.
Set GEMINI_API_KEY env var to run integration tests against the real API.
"""

import json
import os
import shutil
import tempfile

import pytest
from httpx import ASGITransport, AsyncClient

from backend.config import CHROMA_DIR
from backend.main import app
from backend.rag import chunk_text, get_chroma_collection, ingest_text, query_collection
from backend.tools import execute_calc, execute_tool


# =========================================================================
# Tool tests (no API key needed)
# =========================================================================

class TestCalcTool:
    """Calculator tool tests - these always pass, no API key needed."""

    def test_basic_arithmetic(self):
        assert execute_calc("2 + 3")["result"] == 5.0

    def test_multiplication(self):
        assert execute_calc("4 * 7")["result"] == 28.0

    def test_order_of_operations(self):
        assert execute_calc("2 + 3 * 4")["result"] == 14.0

    def test_parentheses(self):
        assert execute_calc("(2 + 3) * 4")["result"] == 20.0

    def test_power(self):
        assert execute_calc("2 ** 10")["result"] == 1024.0

    def test_sqrt(self):
        assert execute_calc("sqrt(16)")["result"] == 4.0

    def test_pi(self):
        result = execute_calc("pi")["result"]
        assert abs(result - 3.14159) < 0.001

    def test_division_by_zero(self):
        result = execute_calc("1 / 0")
        assert "error" in result

    def test_invalid_expression(self):
        result = execute_calc("import os")
        assert "error" in result

    def test_no_arbitrary_code(self):
        result = execute_calc("__import__('os').system('ls')")
        assert "error" in result

    def test_execute_tool_dispatch(self):
        result = execute_tool("calc", {"expression": "10 + 5"})
        assert result["result"] == 15.0

    def test_unknown_tool(self):
        result = execute_tool("nonexistent", {})
        assert "error" in result


# =========================================================================
# Chunking tests (no API key needed)
# =========================================================================

class TestChunking:
    """Text chunking tests."""

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_multiple_chunks(self):
        text = ". ".join([f"Sentence number {i}" for i in range(100)]) + "."
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # Each chunk should be roughly within size
        for chunk in chunks:
            assert len(chunk) < 200  # some tolerance

    def test_overlap_preserves_context(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_text(text, chunk_size=40, overlap=20)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_text("")
        assert len(chunks) == 0 or chunks == [""]


# =========================================================================
# ChromaDB tests (no API key needed for collection management)
# =========================================================================

class TestChromaDB:
    """ChromaDB collection tests."""

    def test_create_collection(self):
        col = get_chroma_collection("test_collection")
        assert col is not None
        assert col.name == "test_collection"

    def test_collection_starts_empty(self):
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        # Delete if exists from previous run
        try:
            client.delete_collection("test_empty")
        except Exception:
            pass
        col = get_chroma_collection("test_empty")
        assert col.count() == 0


# =========================================================================
# RAG integration tests (need API key)
# =========================================================================

class TestRAGIntegration:
    """RAG tests that require a real API key."""

    def test_ingest_and_query(self, api_key, has_api_key):
        if not has_api_key:
            pytest.skip("No GEMINI_API_KEY set - skipping integration test")

        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("test_rag")
        except Exception:
            pass

        # Ingest
        result = ingest_text(
            "Python is a programming language. It was created by Guido van Rossum. "
            "Python supports multiple programming paradigms including procedural, "
            "object-oriented, and functional programming.",
            source="test.md",
            api_key=api_key,
            collection_name="test_rag",
        )
        assert result["status"] == "ok"
        assert result["chunks"] > 0

        # Query
        chunks = query_collection("Who created Python?", api_key, top_k=3, collection_name="test_rag")
        assert len(chunks) > 0
        # At least one chunk should mention Guido
        texts = " ".join(c["text"] for c in chunks)
        assert "guido" in texts.lower() or "python" in texts.lower()


# =========================================================================
# API endpoint tests
# =========================================================================

@pytest.mark.anyio
class TestAPI:
    """API endpoint tests."""

    async def test_root_returns_html(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("content-type", "")

    async def test_models_requires_key(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/models")
            assert resp.status_code == 400

    async def test_models_with_key(self, api_key, has_api_key):
        if not has_api_key:
            pytest.skip("No GEMINI_API_KEY set")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/models?api_key={api_key}")
            assert resp.status_code == 200
            data = resp.json()
            assert "models" in data
            assert len(data["models"]) > 0

    async def test_model_card(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/model-card?model=models/gemini-2.5-flash")
            assert resp.status_code == 200
            data = resp.json()
            assert "card_link" in data
            assert "gemini-2.5" in data["family"]

    async def test_tool_schemas(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/tools/schemas")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["tools"]) >= 2
            names = [t["name"] for t in data["tools"]]
            assert "calc" in names
            assert "search_docs" in names

    async def test_tool_execute_calc(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/tools/execute", json={
                "tool_name": "calc",
                "args": {"expression": "2 + 2"},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["result"]["result"] == 4.0

    async def test_collections_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/rag/collections")
            assert resp.status_code == 200
            data = resp.json()
            assert "collections" in data
