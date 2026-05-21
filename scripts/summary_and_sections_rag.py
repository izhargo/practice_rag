"""
summary_and_sections_rag.py — Build the CCH summary+section RAG index from analyses.

Reads analysis JSON files from output/analyses/, divides each filing into
sections using start_text anchors, chunks each section, prepends contextual
headers (title, filing summary, section heading, section summary), embeds,
and stores in a FAISS index.

Usage:
    python scripts/summary_and_sections_rag.py
    python scripts/summary_and_sections_rag.py --input-dir edgar_data/normalized_filings
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from handlers.embedder import OpenAIEmbedder
from handlers.vector_db_handlers import FaissHandler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILINGS_DIR = ROOT / "edgar_data" / "normalized_filings"
ANALYSES_DIR = ROOT / "output" / "analyses"
FAISS_PATH = str(ROOT / "cch_sections_index.faiss")
DB_PATH = str(ROOT / "cch_sections_index.db")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _find_anchor(text: str, anchor: str, search_from: int = 0) -> int:
    """Find an anchor string in text, tolerant of whitespace differences."""
    idx = text.find(anchor, search_from)
    if idx != -1:
        return idx

    words = re.sub(r"\s+", " ", anchor).strip().split()
    if len(words) >= 2:
        pattern = r"\s+".join(re.escape(w) for w in words)
        match = re.search(pattern, text[search_from:], re.IGNORECASE)
        if match:
            return search_from + match.start()

    return -1


def find_section_boundaries(text: str, sections: list[dict]) -> list[tuple[str, str, str]]:
    """Return list of (heading, summary, section_text) by locating start_text anchors."""
    located = []
    for section in sections:
        heading = section["heading"]
        summary = section["summary"]
        start_anchor = section.get("start_text", "")

        if not start_anchor:
            print(f"    WARNING: no start_text for section '{heading}', skipping")
            continue

        search_from = located[-1][2] if located else 0
        idx = _find_anchor(text, start_anchor, search_from)

        if idx == -1:
            print(f"    WARNING: could not locate section '{heading}', skipping")
            continue

        located.append((heading, summary, idx))

    results = []
    for i, (heading, summary, start_idx) in enumerate(located):
        end_idx = located[i + 1][2] if i + 1 < len(located) else len(text)
        section_text = text[start_idx:end_idx]
        if section_text.strip():
            results.append((heading, summary, section_text))

    return results


def build_chunk_header(title: str, filing_summary: str, section_heading: str, section_summary: str, include_section_summary: bool = True) -> str:
    header = f"{title}\n{filing_summary}\n{section_heading}\n"
    if include_section_summary:
        header += f"{section_summary}\n"
    return header


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CCH summary+section index from analyses.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_FILINGS_DIR,
                        help="Directory containing normalized .txt filings")
    parser.add_argument("--section-summary", action="store_true",
                        help="Include section summary in chunk headers")
    args = parser.parse_args()

    analysis_files = sorted(ANALYSES_DIR.glob("*.json"))
    # Filter out debug files
    analysis_files = [f for f in analysis_files if not f.stem.endswith("_raw_response")]

    if not analysis_files:
        print(f"No analysis files found in {ANALYSES_DIR}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embedder = OpenAIEmbedder()
    db = FaissHandler(
        vector_size=OpenAIEmbedder.VECTOR_SIZE,
        faiss_path=FAISS_PATH,
        db_path=DB_PATH,
    )

    total = len(analysis_files)
    print(f"Found {total} analyses in {ANALYSES_DIR}")

    for i, analysis_path in enumerate(analysis_files, 1):
        filing_name = analysis_path.stem
        source = f"{filing_name}.txt"

        if db.is_indexed(source):
            print(f"[{i}/{total}] SKIP — {source}")
            continue

        filing_path = args.input_dir / source
        if not filing_path.exists():
            print(f"[{i}/{total}] FILE NOT FOUND — {filing_path}")
            continue

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        filing_text = filing_path.read_text(encoding="utf-8", errors="replace")
        title = analysis["title"]
        filing_summary = analysis["summary"]
        sections = analysis.get("sections", [])

        if not sections:
            chunks = splitter.split_text(filing_text)
            header = build_chunk_header(title, filing_summary, "Full Document", filing_summary, args.section_summary)
            headed_chunks = [f"{header}{chunk}" for chunk in chunks]
        else:
            section_data = find_section_boundaries(filing_text, sections)
            headed_chunks = []
            for heading, summary, section_text in section_data:
                chunks = splitter.split_text(section_text)
                header = build_chunk_header(title, filing_summary, heading, summary, args.section_summary)
                headed_chunks.extend(f"{header}{chunk}" for chunk in chunks)

        if not headed_chunks:
            print(f"[{i}/{total}] EMPTY — {source}")
            continue

        embeddings = embedder.embed_documents(headed_chunks)
        db.store(embeddings, headed_chunks, source)
        print(f"[{i}/{total}] OK ({len(headed_chunks)} chunks) — {title}")

    print(f"\nDone. Total vectors in index: {db.index.ntotal}")


if __name__ == "__main__":
    main()
