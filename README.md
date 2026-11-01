# Module 7 Project — Semantic Search Engine

---

## Overview

Build a semantic search tool over a document collection. The tool loads, chunks,
embeds, and stores documents in ChromaDB, then provides a Streamlit interface
where users can type a question and see ranked results. You'll also run a
chunking experiment and document your findings.

---

## System Architecture

The Semantic Search Engine uses a modular architecture that separates document ingestion, semantic retrieval, user interaction, and evaluation.

### Document Ingestion

`ingest.py` loads eight technical documents from the `docs/` directory, splits their text into overlapping chunks, and converts each chunk into a 384-dimensional embedding using the `all-MiniLM-L6-v2` SentenceTransformer model.

The embeddings, original text, source filenames, and chunk indices are stored in a persistent ChromaDB collection.

### Semantic Search

`search.py` converts each user query into an embedding using the same model. ChromaDB compares the query vector against stored document vectors and returns the closest matches.

The search function supports configurable result counts, source filtering, and distance thresholds.

### Streamlit Interface

`app.py` provides the interactive user interface. Users can enter natural-language questions, filter documents, adjust result counts, inspect ranked passages, and rebuild the index with different chunk sizes and overlap values.

### Search Evaluation

`evaluate.py` runs five predefined queries and measures precision and recall against expected document sources.

These metrics help compare retrieval behavior across different chunking configurations.

### Processing Workflow

1. Documents are loaded from `docs/`.
2. `ingest.py` splits documents into chunks and generates embeddings.
3. ChromaDB stores the chunks, vectors, and metadata.
4. The user submits a question through `app.py`.
5. `search.py` embeds the question and retrieves ranked matches.
6. Streamlit displays the matched passages and relevance information.
7. `evaluate.py` measures retrieval performance using test queries.

---

## Features

- Natural-language semantic search across eight technical documents.
- SentenceTransformer embeddings with persistent ChromaDB storage.
- Configurable number of ranked search results.
- Source-file filtering through a multiselect control.
- Optional distance threshold for filtering weaker matches.
- Expandable results displaying source filenames, chunk indices, matched text, and distances.
- Re-indexing controls for configurable chunk size and overlap.
- Document and chunk statistics displayed with Streamlit metrics.
- Precision and recall evaluation using five predefined queries.
- Chunking experiment comparing retrieval performance and passage quality.

---

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Index the document corpus:

   ```bash
   python ingest.py
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Files

| File               | Purpose                                               |
| ------------------ | ----------------------------------------------------- |
| `app.py`           | Streamlit search interface — your primary work file   |
| `ingest.py`        | Document loading, chunking, and ChromaDB storage      |
| `search.py`        | Search functions: query ChromaDB and rank results     |
| `evaluate.py`      | Evaluation framework: precision and recall            |
| `docs/`            | Starter document corpus (add your own documents here) |
| `chroma_data/`     | Persistent ChromaDB storage (created on first run)    |
| `requirements.txt` | Python dependencies                                   |

## Requirements

See the project brief on the course platform for the full rubric. Key sections:

1. **Document Ingestion** — load `docs/` folder, configurable chunk size/overlap,
   store in ChromaDB with source and chunk-index metadata
2. **Semantic Search** — ranked results with scores, configurable `n_results` and
   threshold, edge-case handling
3. **Streamlit Interface** — search input, sidebar controls (filter, re-index,
   result count), at least one `st.metric()`
4. **Chunking Experiment** — re-index at 2+ chunk sizes, run 5 identical queries,
   compare and document findings in this README
5. **Code Quality** — docstrings, modular files, working `requirements.txt`

---

## Chunking Experiment Results

### Objective

Compare two document chunking configurations to determine how chunk size affects semantic search quality, retrieval accuracy, and the usefulness of returned passages.

### Configurations Tested

| Configuration | Chunk Size | Overlap | Total Chunks |
| ------------- | ---------: | ------: | -----------: |
| A             |        500 |     100 |           53 |
| B             |        200 |      50 |          137 |

Both configurations indexed the same eight technical documents using the `all-MiniLM-L6-v2` embedding model and persistent ChromaDB storage.

### Five Evaluation Queries

1. What are Python functions and variables?
2. How does FastAPI handle API requests?
3. What are SQL joins and relational databases?
4. How do embeddings represent text as vectors?
5. How does Streamlit create interactive web apps?

Each query retrieved the top three matching chunks without a distance threshold.

### Quantitative Results

| Metric            | Configuration A | Configuration B |
| ----------------- | --------------: | --------------: |
| Average Precision |            100% |            100% |
| Average Recall    |            100% |            100% |
| Indexed Chunks    |              53 |             137 |

Both configurations returned three chunks from the expected document source for every evaluation query.

Precision counted the proportion of retrieved chunks belonging to expected source files. Recall measured whether the expected source files appeared among retrieved results.

These results measure source retrieval rather than independently verified answer correctness.

### Qualitative Comparison

The query "How does FastAPI handle API requests?" was examined in both configurations.

**Configuration A — 500 characters / 100 overlap**

Top-three distances: 0.601, 0.703, 0.864.

The returned passages contained broader explanations covering FastAPI request validation, middleware, testing, and dependency injection.

**Configuration B — 200 characters / 50 overlap**

Top-three distances: 0.568, 0.631, 0.713.

The returned passages were shorter and more focused on individual concepts, including middleware and request-body validation.

However, some passages ended in the middle of sentences, reducing readability and context.

Distances were recorded for comparison but should not be interpreted as calibrated relevance percentages or definitive evidence that one configuration is superior.

### Findings

Both chunking configurations achieved identical precision and recall on the five question evaluation set.

Smaller chunks produced more searchable units and sometimes more focused passages, while larger chunks preserved more surrounding context.

Neither configuration eliminated incomplete sentences at chunk boundaries because the current implementation uses fixed character based chunking.

**Preferred configuration:** 500 character chunks with 100 character overlap.

This configuration offers a practical balance between search relevance, readable context, and index size for the current document collection.

### Limitations and Future Improvements

- Evaluate more queries, including ambiguous and cross document questions.
- Use semantic or sentence aware chunking to avoid splitting sentences.
- Measure passage level relevance rather than relying only on source filenames.
- Consider reranking retrieved passages to improve result ordering.
- Cache the embedding model to reduce repeated initialization overhead.
