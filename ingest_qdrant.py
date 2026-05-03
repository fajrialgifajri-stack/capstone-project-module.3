import pickle
import os
import time
from dotenv import load_dotenv
 
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "cinema_intelligence"
QDRANT_DOCS     = "qdrant_docs.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE      = 50
MAX_RETRIES     = 3
RETRY_DELAY     = 5  # detik

# ─────────────────────────────────────────────
# 1. LOAD DOKUMEN
# ─────────────────────────────────────────────
def load_docs(path: str) -> list[Document]:
    with open(path, "rb") as f:
        raw_docs = pickle.load(f)
 
    docs = [
        Document(
            page_content=d["page_content"],
            metadata=d["metadata"]
        )
        for d in raw_docs
    ]
    print(f"[LOAD] {len(docs)} dokumen di-load dari '{path}'")
    return docs
 
 
# ─────────────────────────────────────────────
# 2. CEK BERAPA YANG UDAH KE-INSERT (RESUME)
# ─────────────────────────────────────────────
def get_already_inserted(client: QdrantClient, collection_name: str) -> int:
    try:
        info = client.get_collection(collection_name)
        count = info.points_count
        return count if count else 0
    except Exception:
        return 0
 
 
# ─────────────────────────────────────────────
# 3. SETUP COLLECTION
# ─────────────────────────────────────────────
def setup_collection(client: QdrantClient, collection_name: str):
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        print(f"[Qdrant] Menghapus collection lama '{collection_name}'...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print(f"[Qdrant] Collection '{collection_name}' baru dibuat.")
 
# ─────────────────────────────────────────────
# 4. EMBED & INSERT DENGAN RESUME + RETRY
# ─────────────────────────────────────────────
def ingest(docs: list[Document], collection_name: str):
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
 
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60  
    )
 
    # Setup collection 
    setup_collection(client, collection_name)
 
    start_index = 0
    print(f"[Ingest] Fresh ingest {len(docs)} dokumen...\n")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
 
    remaining_docs = docs[start_index:]
    batches = [
        remaining_docs[i:i + BATCH_SIZE]
        for i in range(0, len(remaining_docs), BATCH_SIZE)
    ]
 
    total_batches = len(batches)
 
    for i, batch in enumerate(batches):
        batch_num   = i + 1
        docs_so_far = min(start_index + (i + 1) * BATCH_SIZE, len(docs))
 
        for attempt in range(MAX_RETRIES):
            try:
                vector_store.add_documents(batch)
                print(f"  ✅ Batch {batch_num}/{total_batches} — {docs_so_far}/{len(docs)} film terupload")
                break
 
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠️  Batch {batch_num} timeout, retry {attempt + 1}/{MAX_RETRIES - 1} dalam {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  ❌ Batch {batch_num} gagal setelah {MAX_RETRIES}x percobaan.")
                    print(f"     Error: {e}")
                    print(f"\n💡 Jalankan ulang script ini — akan otomatis lanjut dari batch yang gagal.")
                    raise
 
    print(f"\n✅ Ingest complete! {len(docs)} dokumen berhasil di-insert.")
    print(f"   Collection : '{collection_name}'")
 
 
# ─────────────────────────────────────────────
# 5. VERIFIKASI
# ─────────────────────────────────────────────
def verify(collection_name: str):
    print("\n[Verify] Menjalankan test query...")
 
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
 
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
 
    info = client.get_collection(collection_name)
    print(f"[Verify] Total dokumen di Qdrant: {info.points_count}")
 
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
 
    results = vector_store.similarity_search(
        "a mind-bending thriller about dreams and reality",
        k=3
    )
 
    print("\n[Verify] Top 3 hasil query 'a mind-bending thriller about dreams and reality':")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.metadata}")
    print("\n✅ Qdrant siap dipakai!")
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_docs(QDRANT_DOCS)
    ingest(docs, COLLECTION_NAME)
    verify(COLLECTION_NAME)