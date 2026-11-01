"""
Module 7 Project — Semantic Search Engine
==========================================
ingest.py — document loading, chunking, and ChromaDB storage

Run with:
    python ingest.py
    python ingest.py --chunk-size 200 --overlap 50
"""

import argparse
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_DIR        = Path("docs")
CHROMA_PATH     = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME      = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP    = 100


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text:       Full document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    
    chunks = []
    start = 0 
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

        start += step

    return chunks


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Read all .txt and .md files from docs_dir.

    Returns:
        List of dicts: {"filename": str, "text": str}
    """
    
    documents = []

    for file_path in sorted(docs_dir.iterdir()):
        if file_path.suffix.lower() not in (".txt", ".md"):
            continue

        text = file_path.read_text(encoding="utf-8")

        if text.strip():
            documents.append({
                "filename": file_path.name,
                "text": text
            })

    return documents



def get_collection(chroma_path: Path, collection_name: str):
    """Create (or retrieve) a persistent ChromaDB collection."""

    client = chromadb.PersistentClient(path=str(chroma_path))

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection


def ingest(chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    """
    Full ingestion pipeline: load → chunk → embed → upsert.

    Each chunk is stored with metadata: source filename, chunk index,
    and the chunk size used — so experiments with different sizes can
    be compared without ambiguity.
    """
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    documents = load_documents(DOCS_DIR)
    model = SentenceTransformer(MODEL_NAME)

    
    # Reset the collection before indexing to prevent mixing chunk configurations.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    texts = []
    metadatas = []

    for document in documents:
        chunks = chunk_text(document["text"], chunk_size, overlap)

        for index, chunk in enumerate(chunks):
            ids.append(f"{document['filename']}_{chunk_size}_{overlap}_{index}")
            texts.append(chunk)
            metadatas.append({
                "source": document["filename"],
                "chunk_index": index,
                "chunk_size": chunk_size,
                "overlap": overlap
            })

    if not texts:
        print("No document chunks found.")
        return

    
    batch_size = 16

    for start in range(0, len(texts), batch_size):
        end = start + batch_size

        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_metadata = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            batch_size=8,
            show_progress_bar=False
        ).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadata
        )

        print(f"Stored {min(end, len(texts))}/{len(texts)} chunks")


    print(f"Indexed {len(documents)} documents into {len(texts)} chunks.")
    print(f"Collection now contains {collection.count()} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index docs/ into ChromaDB")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap",    type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    ingest(chunk_size=args.chunk_size, overlap=args.overlap)
