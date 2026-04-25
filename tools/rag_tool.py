"""
tools/rag_tool.py
=================
Dua tool yang pakai Qdrant:
1. semantic_search_tool  — rekomendasi film by vibe/deskripsi
2. similar_movie_tool    — cari film mirip judul tertentu
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from qdrant_client import QdrantClient

load_dotenv()

# ── Setup client & vector store
_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)
_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
_vector_store = QdrantVectorStore(
    client=_client,
    collection_name="cinema_intelligence",
    embedding=_embeddings
)


def _format_results(docs: list, max_results: int = 5) -> str:
    """Format hasil retrieval jadi string ringkas."""
    if not docs:
        return "Tidak ada film yang ditemukan."

    lines = []
    for i, doc in enumerate(docs[:max_results], 1):
        m = doc.metadata
        gross = f"${m['gross']:,}" if m.get("gross") else "N/A"
        lines.append(
            f"{i}. {m['title']} ({m['year']}) | "
            f"⭐ {m['rating']} | {m['genre']}\n"
            f"   Director: {m['director']} | Cast: {m['cast'][:60]}...\n"
            f"   {m['overview'][:120]}..."
        )
    return "\n\n".join(lines)


@tool
def semantic_search_tool(query: str) -> str:
    """
    Cari film berdasarkan deskripsi, vibe, tema, atau mood.
    Gunakan tool ini untuk pertanyaan seperti:
    - 'Film tentang persahabatan yang mengharukan'
    - 'Rekomendasiin film sci-fi yang mind-blowing'
    - 'Film dengan plot twist yang gila'
    """
    docs = _vector_store.similarity_search(query, k=5)
    return _format_results(docs)


@tool
def similar_movie_tool(movie_title: str) -> str:
    """
    Cari film yang mirip dengan judul film tertentu.
    Gunakan tool ini untuk pertanyaan seperti:
    - 'Film yang mirip Inception'
    - 'Rekomendasi film seperti The Dark Knight'
    - 'Film serupa dengan Interstellar'
    """
    # Cari dulu film yang dimaksud untuk ambil overview-nya
    title_docs = _vector_store.similarity_search(
        f"Title: {movie_title}", k=1
    )

    if not title_docs:
        return f"Film '{movie_title}' tidak ditemukan di database."

    # Pakai overview film tersebut sebagai query similarity
    source_movie = title_docs[0]
    query = source_movie.metadata.get("overview", movie_title)
    source_title = source_movie.metadata.get("title", movie_title)

    # Cari film mirip (k=6 lalu exclude film itu sendiri)
    similar_docs = _vector_store.similarity_search(query, k=6)
    similar_docs = [
        d for d in similar_docs
        if d.metadata.get("title", "").lower() != source_title.lower()
    ][:5]

    result = f"Film mirip '{source_title}':\n\n"
    result += _format_results(similar_docs)
    return result
