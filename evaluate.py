"""
Module 7 Project — Semantic Search Engine
==========================================
evaluate.py — precision and recall for your search system

Run with:
    python evaluate.py
    python evaluate.py --n-results 5 --threshold 0.4

Define your evaluation set in EVAL_SET, then run this script against
different index configurations (different chunk sizes) to compare results.
"""

import argparse
from search import search


# Define your evaluation set here.
# Each entry needs a query and the source filenames you expect to be relevant.
# Use at least 5 queries for the chunking experiment.
# Define five test queries and their expected relevant document sources.

EVAL_SET = [
    {
        "query": "What are Python functions and variables?",
        "relevant_sources": ["python-fundamentals.txt"]
    },
    {
        "query": "How does FastAPI handle API requests?",
        "relevant_sources": ["fastapi.txt"]
    },
    {
        "query": "What are SQL joins and relational databases?",
        "relevant_sources": ["sql-databases.txt"]
    },
    {
        "query": "How do embeddings represent text as vectors?",
        "relevant_sources": ["embeddings-and-vectors.txt"]
    },
    {
        "query": "How does Streamlit create interactive web apps?",
        "relevant_sources": ["streamlit.txt"]
    }
]


def precision_recall(
    retrieved_sources: list[str], relevant_sources: list[str]
) -> tuple[float, float]:
    """
    Compute source-level precision and recall.

    Returns (precision, recall) as floats in [0, 1].
    """
    # Count relevant retrieved results and calculate precision and recall.
    relevant = set(relevant_sources)

    correct = sum(
        1 for source in retrieved_sources
        if source in relevant
    )

    precision = correct / len(retrieved_sources) if retrieved_sources else 0.0

    recall = (
        len(set(retrieved_sources) & relevant) / len(relevant)
        if relevant else 0.0
    )

    return precision, recall


def evaluate(n_results: int = 5, distance_threshold: float = None):
    """
    Run every query in EVAL_SET and print per-query and average precision/recall.
    """
    # Evaluate each test query and calculate individual retrieval metrics.
    total_precision = 0.0
    total_recall = 0.0

    for item in EVAL_SET:
        query = item["query"]
        relevant_sources = item["relevant_sources"]

        results = search(
            query=query,
            n_results=n_results,
            distance_threshold=distance_threshold
        )

        retrieved_sources = [
            result["source"] for result in results
        ]

        precision, recall = precision_recall(
            retrieved_sources,
            relevant_sources
        )

        total_precision += precision
        total_recall += recall

        print(f"\nQuery: {query}")
        print(f"Retrieved: {retrieved_sources}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")

    # Calculate average precision and recall across all evaluation queries.
    count = len(EVAL_SET)

    if count == 0:
        print("No evaluation queries defined.")
        return

    avg_precision = total_precision / count
    avg_recall = total_recall / count

    print("\n--- Evaluation Summary ---")
    print(f"Average Precision: {avg_precision:.2%}")
    print(f"Average Recall: {avg_recall:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--n-results", type=int,   default=5)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    evaluate(n_results=args.n_results, distance_threshold=args.threshold)
