# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`practice_rag` is a Python project implementing a dual-index Retrieval-Augmented Generation (RAG) pipeline over a corpus of 152 SEC filings downloaded from EDGAR.

## Setup

```bash
poetry install
```

## Remote

`git@github.com:izhargo/practice_rag.git`

## Architecture

### 1. Data Pipeline (complete)
- `download_edgar.py` — downloads SEC filings listed in `edgar_data/cik_list.xlsx` from EDGAR, strips embedded HTML to plain text, and saves to `edgar_data/filings/`
- 152 reports total; older filings are plain SGML, newer ones embed HTML that is cleaned via BeautifulSoup + lxml

### 2. Vector DB — Store A: Simple RAG
- Chunk each filing into 1000-char chunks with 200-char overlap (`RecursiveCharacterTextSplitter`)
- Embed chunks and store in a persisted FAISS index
- Index saved to disk so it survives across runs

### 3. Vector DB — Store B: Contextual Chunk Headers (CCH)
- Same chunking parameters (1000 chars / 200 overlap)
- Each chunk is prefixed with: `[Document title] > [Section heading]`
- Stored in a separate persisted FAISS index

### 4. User Interface
- Entry point: `ask.py`
- Input: a question typed in the terminal, OR a path to a text file containing one question per line
- Flow per question:
  1. Embed the question
  2. Retrieve top-k chunks from Store A and Store B independently
  3. Build two prompts: `retrieved_chunks_A + question` and `retrieved_chunks_B + question`
  4. Send both prompts to the MiniMax model
  5. Return both responses side-by-side
- Output: printed to terminal, OR written to a response text file (mirrors input mode)

### 5. Model
- MiniMax M2.7 via `https://api.minimax.io/v1` (OpenAI-compatible API)
- Requires `MINIMAX_API_KEY` env var
- Embedding model: `google/embeddinggemma-300M` via `sentence-transformers`

## Key Files
- `download_edgar.py` — data download + HTML cleaning
- `document_handler.py` — `TxtHandler` / `PDFHandler` for chunking
- `embedder.py` — wraps sentence-transformers embedding model
- `vector_db_handlers.py` — `FaissHandler` (store + search)
- `rag.py` — single-document RAG pipeline (reference implementation)
- `ask.py` — user-facing Q&A interface
- `edgar_data/filings/` — downloaded and cleaned SEC filings
- `edgar_data/cik_list.xlsx` — source list of filings to download