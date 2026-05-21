# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`practice_rag` is a Python project that compares multiple RAG (Retrieval-Augmented Generation) strategies over a corpus of 152 SEC filings downloaded from EDGAR. Four index variants are built and evaluated side-by-side.

## Setup

```bash
poetry install
```

Requires environment variables: `OPENAI_API_KEY`, `MINIMAX_API_KEY`.

## Remote

`git@github.com:izhargo/practice_rag.git`

## Architecture

### 1. Data Pipeline
- `scripts/download_edgar.py` — downloads SEC filings listed in `edgar_data/cik_list.xlsx` from EDGAR, strips embedded HTML to plain text, and saves to `edgar_data/filings/`
- 152 reports total; older filings are plain SGML, newer ones embed HTML that is cleaned via BeautifulSoup + lxml
- Normalized (clean) filings live in `edgar_data/normalized_filings/`
- The 100 filings that have analyses and are indexed live in `edgar_data/indexed_filings/`

### 2. Analysis Pipeline (for CCH indices)
- `scripts/summary_and_sections_batch_submit.py` — submits filings to OpenAI Batch API for structured analysis (title, summary, sections with headings/summaries/start_text anchors)
- `scripts/summary_and_sections_batch_collect.py` — collects completed batch results
- Analysis JSON files stored in `output/analyses/`
- 100 of 152 filings have completed analyses; 5 were skipped (repeated truncation), 47 remain unsubmitted

### 3. Four FAISS Index Variants
All use 1000-char chunks with 200-char overlap (`RecursiveCharacterTextSplitter`) and OpenAI `text-embedding-3-small` (1536 dims).

#### Store A: Simple RAG (`scripts/simple_rag.py`)
- Plain chunks, no metadata headers

#### Store B: CCH RAG (`scripts/cch_rag.py`)
- Each chunk prefixed with: `{document title} > {section heading}`

#### Store C: CCH Sections RAG (`scripts/summary_and_sections_rag.py`)
- Per-section chunking using start_text anchors from analysis
- Each chunk prefixed with: `{title}\n{filing summary}\n{section heading}`

#### Store D: CCH Summary+Sections RAG (`scripts/summary_and_sections_rag.py --section-summary`)
- Same as Store C but also includes section summary in the header

### 4. Query Interface
- Entry point: `scripts/call_model.py`
- Select indices via flags: `--simple`, `--cch`, `--cch-sections`, `--cch-ss`
- Input: a question as CLI arg, or `--questions-file` for batch
- Flow per question per selected index:
  1. Embed the question
  2. Retrieve top-5 chunks from selected index
  3. Build prompt: `retrieved_chunks + question`
  4. Send to MiniMax model
  5. Output all chunks then all responses
- Output: printed to stdout, or `--output-file` for file

### 5. Models
- **LLM**: MiniMax M2.7 via `https://api.minimax.io/v1` (OpenAI-compatible API) — requires `MINIMAX_API_KEY`
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dims) — requires `OPENAI_API_KEY`
- **Analysis**: OpenAI Batch API (GPT) for filing analysis — uses `OPENAI_API_KEY`

## Key Files
- `scripts/download_edgar.py` — data download + HTML cleaning
- `scripts/simple_rag.py` — builds Simple RAG index
- `scripts/cch_rag.py` — builds CCH RAG index
- `scripts/summary_and_sections_rag.py` — builds CCH Sections / CCH Summary+Sections indices
- `scripts/summary_and_sections_batch_submit.py` — submits filings for analysis
- `scripts/summary_and_sections_batch_collect.py` — collects analysis results
- `scripts/call_model.py` — query interface across all indices
- `handlers/embedder.py` — OpenAI embedding wrapper
- `handlers/vector_db_handlers.py` — `FaissHandler` (FAISS + SQLite store/search)
- `utils.py` — LLM call helper
- `prompt.py` — analysis prompt template
- `edgar_data/cik_list.xlsx` — source list of filings to download
- `edgar_data/normalized_filings/` — cleaned filing texts
- `edgar_data/indexed_filings/` — the 100 filings used in the indices
- `output/analyses/` — JSON analysis files (title, summary, sections)