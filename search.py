"""
Module 7 Project — Semantic Search Engine
==========================================
search.py — query ChromaDB and return ranked results

Import this module into app.py:
    from search import search, get_collection_stats
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration (must match ingest.py) ─────────────────────────────────────
CHROMA_PATH     = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME      = "all-MiniLM-L6-v2"


def get_collection():
    """Return the persistent ChromaDB collection."""
    
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection



def search(
    query: str,
    n_results: int = 5,
    sources: list[str] = None,
    distance_threshold: float = None,
) -> list[dict]:
    """
    Search the ChromaDB collection and return ranked results.

    Args:
        query:              Natural language search query.
        n_results:          Maximum number of results to return.
        sources:            If provided, only return chunks from these filenames.
        distance_threshold: If provided, exclude results with distance above this
                            value (lower = more similar).

    Returns:
        List of result dicts sorted by distance ascending (best first):
            {
                "text":        str,
                "source":      str,
                "chunk_index": int,
                "distance":    float,
                "score":       float,  # 1 - distance
            }
        Returns [] for empty queries or if the collection has no documents.
    """
    if not query.strip() or n_results <= 0:
        return []

    collection = get_collection()

    if collection.count() == 0:
        return[]

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query).tolist()
    
    where_filter = None

    if sources:
        where_filter = {"source": {"$in": sources}}

    matching_count = (
        len(collection.get(where=where_filter)["ids"])
        if where_filter
        else collection.count()
    )

    if matching_count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, matching_count),
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )


    matches = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if sources and metadata["source"] not in sources:
            continue

        if distance_threshold is not None and distance > distance_threshold:
            continue

        matches.append({
            "text": text,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": float(distance),
            "score": 1 - float(distance)
        })

    return matches


def get_collection_stats() -> dict:
    """
    Return basic stats about the indexed collection.

    Returns:
        {
            "total_chunks":   int,
            "unique_sources": int,
            "source_names":   list[str],
        }
    """
    
    collection = get_collection()

    data = collection.get(include=["metadatas"])
    metadatas = data["metadatas"]

    sources = sorted({
        metadata["source"]
        for metadata in metadatas
        if metadata and "source" in metadata
    })

    return {
        "total_chunks": collection.count(),
        "unique_sources": len(sources),
        "source_names": sources
    }

