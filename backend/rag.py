"""RAG pipeline: ingest, chunk, embed, store, retrieve.

Uses ChromaDB for vector storage with persistent disk-based collections.
"""

import os
import re

import chromadb

from backend.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    RAG_TOP_K,
)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """Get embeddings from Gemini embedding model."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )
    return [e.values for e in result.embeddings]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by sentences, respecting size limits."""
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_text = ""
            overlap_sents: list[str] = []
            for s in reversed(current):
                if len(overlap_text) + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_text = " ".join(overlap_sents)
            current = overlap_sents
            current_len = len(overlap_text)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB collection
# ---------------------------------------------------------------------------

def _get_chroma_client():
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_chroma_collection(collection_name: str = COLLECTION_NAME):
    """Get or create a ChromaDB collection with persistent storage."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_text(
    text: str,
    source: str,
    api_key: str,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Chunk text, embed, and store in ChromaDB. Returns ingest stats."""
    chunks = chunk_text(text)
    if not chunks:
        return {"status": "empty", "chunks": 0}

    embeddings = get_embeddings(chunks, api_key)
    collection = get_chroma_collection(collection_name)

    # Create deterministic IDs from source + chunk index
    base = re.sub(r"[^a-zA-Z0-9]", "_", source)[:50]
    ids = [f"{base}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return {"status": "ok", "chunks": len(chunks), "source": source}


def ingest_file(file_path: str, api_key: str, collection_name: str = COLLECTION_NAME) -> dict:
    """Read a file and ingest its contents."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    source = os.path.basename(file_path)
    return ingest_text(text, source, api_key, collection_name)


# ---------------------------------------------------------------------------
# Query / Retrieve
# ---------------------------------------------------------------------------

def query_collection(
    query: str,
    api_key: str,
    top_k: int = RAG_TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Retrieve top_k relevant chunks for a query."""
    collection = get_chroma_collection(collection_name)

    if collection.count() == 0:
        return []

    query_embedding = get_embeddings([query], api_key)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })

    return chunks


# ---------------------------------------------------------------------------
# Collection info
# ---------------------------------------------------------------------------

def list_collections() -> list[dict]:
    """List all ChromaDB collections with counts."""
    client = _get_chroma_client()
    collections = client.list_collections()
    result = []
    for col in collections:
        result.append({"name": col.name, "count": col.count()})
    return result


def delete_collection(collection_name: str = COLLECTION_NAME) -> dict:
    """Delete a collection."""
    client = _get_chroma_client()
    client.delete_collection(collection_name)
    return {"status": "deleted", "collection": collection_name}


# ---------------------------------------------------------------------------
# RAG generation prompt builder
# ---------------------------------------------------------------------------

def build_rag_prompt(query: str, chunks: list[dict], grounded: bool = True) -> str:
    """Build a prompt with retrieved context for the model."""
    context_parts = []
    for c in chunks:
        context_parts.append(f"[Chunk {c['id']}]\n{c['text']}")
    context = "\n\n".join(context_parts)

    if grounded:
        return (
            "You are a helpful assistant. Answer the question using ONLY the context below. "
            "Cite chunk IDs in square brackets like [chunk_id] after each claim. "
            "If the context does not contain enough information, say 'I don't know based on the provided context.'\n\n"
            f"## Context\n{context}\n\n"
            f"## Question\n{query}"
        )
    else:
        return (
            "You are a helpful assistant. Use the context below to help answer the question, "
            "but you may also use your own knowledge. Cite chunk IDs when using context.\n\n"
            f"## Context\n{context}\n\n"
            f"## Question\n{query}"
        )
