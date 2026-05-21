# practice_rag

A comparative study of RAG (Retrieval-Augmented Generation) strategies over SEC filings from EDGAR.

## What This Does

This project downloads SEC filings (10-K, 10-Q) for several companies from the EDGAR database, normalizes and indexes them using four different chunking/embedding strategies, and then lets you query all indices side-by-side to compare retrieval quality and LLM response accuracy.

The goal is to understand how adding contextual metadata to chunks (document titles, filing summaries, section headings) affects retrieval relevance compared to plain chunking.

## The Four RAG Strategies

| Index | Chunk Header | Description |
|-------|-------------|-------------|
| **Simple** | None | Plain 1000-char chunks, no metadata |
| **CCH** | Title + section heading | Each chunk prefixed with document title and section heading |
| **CCH Sections** | Title + filing summary + section heading | Uses AI-generated filing analysis to add a filing-level summary |
| **CCH Summary+Sections** | Title + filing summary + section heading + section summary | Also includes an AI-generated section-level summary |

All indices use the same chunking parameters (1000 chars, 200 overlap) and the same embedding model (OpenAI `text-embedding-3-small`, 1536 dims). Vectors are stored in FAISS with SQLite metadata.

## Pipeline Overview

1. **Download** — Fetch 152 SEC filings from EDGAR, strip HTML to plain text
2. **Analyze** — Submit filings to OpenAI Batch API to extract title, summary, and section structure (headings, summaries, start-text anchors)
3. **Index** — Chunk filings and build four FAISS indices with different header strategies (note: basic cch rag doesn't use the openai analysis but rather grabs the filing's title from text using basic heuristics)
4. **Query** — Ask questions against any combination of indices, compare retrieved chunks and LLM responses side-by-side

## Setup

```bash
poetry install
```

Required environment variables:
- `OPENAI_API_KEY` — for embeddings and filing analysis
- `MINIMAX_API_KEY` — for the LLM (MiniMax M2.7)

## Usage

```bash
# Query specific indices with a question
python scripts/call_model.py --simple --cch-sections "What drove AMR's operating expenses in 1999?"

# Query all four indices
python scripts/call_model.py --simple --cch --cch-sections --cch-ss "Your question here"

# Batch questions from a file
python scripts/call_model.py --cch-sections --cch-ss --questions-file questions.txt --output-file results.txt
```

## Project Structure

```
scripts/
  download_edgar.py                      # Download + clean SEC filings
  simple_rag.py                          # Build Simple RAG index
  cch_rag.py                             # Build CCH RAG index
  summary_and_sections_rag.py            # Build CCH Sections / Summary+Sections indices
  summary_and_sections_batch_submit.py   # Submit filings for AI analysis
  summary_and_sections_batch_collect.py  # Collect analysis results
  call_model.py                          # Query interface

handlers/
  embedder.py                            # OpenAI embedding wrapper
  vector_db_handlers.py                  # FAISS + SQLite store/search

edgar_data/
  cik_list.xlsx                          # Source list of filings to download
  filings/                               # Raw downloaded filings
  normalized_filings/                    # Cleaned plain text filings
  indexed_filings/                       # The 100 filings used in indices

output/
  analyses/                              # JSON analysis files (title, summary, sections)
```