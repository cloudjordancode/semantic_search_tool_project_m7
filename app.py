"""
Module 7 Project — Semantic Search Engine
==========================================
app.py — Streamlit search interface

Run with:
    streamlit run app.py

Make sure you've indexed documents first:
    python ingest.py
"""

import streamlit as st
from search import search, get_collection_stats
from ingest import ingest, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

# Configure the Streamlit interface and retrieve indexed document statistics.
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔎",
    layout="wide"
)

st.title("Semantic Search Engine")
st.caption("Search technical documents using AI-powered semantic similarity.")

stats = get_collection_stats()


# Display the number of indexed documents and searchable chunks.
col1, col2 = st.columns(2)

with col1:
    st.metric("Indexed Documents", stats["unique_sources"])

with col2:
    st.metric("Searchable Chunks", stats["total_chunks"])


# Configure sidebar controls for search results and documnet filtering. 
with st.sidebar:
    st.header("Search Settings")

    st.metric("Indexed Documents", stats["unique_sources"])

    n_results = st.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=5
    )

    # Configure the maximum distance allowed for search results.
    use_threshold = st.checkbox(
        "Enable Distance Threshold",
        value=False
    )

    distance_threshold = None

    if use_threshold:
        distance_threshold = st.slider(
            "Maximum Distance",
            min_value=0.0,
            max_value=2.0,
            value=1.2,
            step=0.1
        )

    # Allow users to filter search results by document source. 
    selected_sources = st.multiselect(
        "Filter by Source",
        options=stats["source_names"],
        default=stats["source_names"]
    )

    # Configure document chunking and rebuild the search index.
    st.divider()
    st.subheader("Document Indexing")

    chunk_size = st.number_input(
        "Chunk Size",
        min_value=50,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=50
    )

    overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=DEFAULT_OVERLAP,
        step=25
    )

    if st.button("Re-index Documents"):
        if overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
        else:
            with st.spinner("Re-indexing documents..."):
                ingest(
                    chunk_size=int(chunk_size),
                    overlap=int(overlap)
                )

            st.success("Documents re-indexed successfully!")
            st.rerun()

# Accept a natural language search query from the user. 
st.subheader("Search Documents")

query = st.text_input(
    "Enter your search query",
    placeholder="e.g., How does FastAPI handle API requests?"
)

if query.strip():
    if selected_sources:
        results = search(
            query=query,
            n_results=n_results,
            sources=selected_sources,
            distance_threshold=distance_threshold
        )
    else:
        results = []

# Display ranked search results with their source and relevance information. 
    if results: 
        for rank, result in enumerate(results, start=1):
            distance = result["distance"]

            if distance < 0.8:
                relevance = "High"
            elif distance < 1.2:
                relevance = "Moderate"
            else: 
                relevance = "Low"

            with st.expander(
                f"#{rank} - {result['source']} | Relevance: {relevance}"
            ):
                st.write(result["text"])
                st.caption(
                    f"Chunk: {result['chunk_index']} | "
                    f"Distance: {distance:.3f}"
                )

    else:
        st.info("No matching documents found. Try another query or adjust your filters.")