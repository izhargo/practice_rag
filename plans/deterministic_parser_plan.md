# Plan: Deterministic Filing Parser for CCH RAG

## Context

We're building the Contextual Chunk Headers (CCH) RAG index. The previous approach used MiniMax LLM to extract titles and sections — we're replacing it with a fully deterministic parser. Every filing needs:
1. A **title** — `"COMPANY NAME — Form TYPE"` (e.g. `"Allis Chalmers Energy Inc. — Form 10-K"`)
2. **Section boundaries** — `[(heading, start_char, end_char), ...]` with char ranges in the cleaned file text

## Files to Create / Modify

### 1. NEW: `handlers/filing_parser.py` — deterministic parser

Two functions:

**`extract_title(file_path) -> dict`**
- Reads first 100 words (59 is enough, 100 gives margin)
- Regex extracts `COMPANY CONFORMED NAME:\s*(.+?)(?:\s+CENTRAL INDEX|$)` and `CONFORMED SUBMISSION TYPE:\s*(.+?)$`
- Returns `{"company_name": str, "form_type": str, "report_category": str, "period": str|None, "title": str}`
- `title` = `f"{company_name} — Form {form_type}"`
- `report_category` mapped from form_type (10-K→Annual, 10-Q→Quarterly, etc.)
- Also extract `CONFORMED PERIOD OF REPORT:\s*(\d+)` for period

**`extract_sections(file_path) -> list[dict]`**
- Reads full file text
- Step 1: Find the end of SGML header. Look for `</SEC-HEADER>` or `<TEXT>`. If neither found, use line 60 as fallback. All content before this marker is skipped (avoids false positives).
- Step 2: From the content start, scan line by line for PART/ITEM headings using regex:
  ```
  ^\s*(PART|Part)\s+([IVX]+)         → PART heading
  ^\s*(ITEM|Item)\s+(\d+[A-Z]?)\.?   → ITEM heading
  ```
- Step 3: For each match, record `heading` (human-readable) and `start_char` (char offset in the full file text)
- Step 4: Compute `end_char` for each section = next section's `start_char` (or len(text) for last section)
- Filtering: skip standalone `ITEM` or `PART` without a number/roman numeral (these are ToC fragments)
- Deduplication: if the same heading appears multiple times (once in ToC, once as actual heading), keep the later occurrence — the ToC appears in the first ~20% of content, actual headings come after. Strategy: when we see a duplicate heading, replace the earlier one.
- NT filings: if `CONFORMED SUBMISSION TYPE` starts with `NT`, return empty sections list
- Returns `[{"heading": "PART I", "start_char": 3659, "end_char": 5012}, ...]`

### 2. MODIFY: `scripts/cch_rag.py` — replace LLM approach with deterministic parser

- Remove the `ANALYSIS_PROMPT`, `build_prompt()`, `parse_response()`, `call_llm_using_openai_client` import
- Import `extract_title`, `extract_sections` from `handlers.filing_parser`
- Keep `resolve_section_offsets` concept but it's now built into `extract_sections`
- New `analyze_filing(path)` just calls both functions and returns combined result
- Smoke test at bottom: run on all files, report any files where parsing failed (no sections and not NT)

### 3. Keep: `utils.py`, `handlers/document_handler.py`, `handlers/vector_db_handlers.py`, `handlers/embedder.py` — unchanged

These will be used later when we chunk and store.

## Heading Detection — Detailed Logic

```
For each line after SGML header:
  1. Strip leading/trailing whitespace
  2. Match against PART pattern: ^\s*(PART|Part)\s+([IVX]+)\s*[.:\-—]?\s*(.*)$
     → heading = "PART {roman}" or "PART {roman} — {description}" if description exists
  3. Match against ITEM pattern: ^\s*(ITEM|Item)\s+(\d+[A-Z]?)\s*[.:\-—]?\s*(.*)$
     → heading = "Item {num}" or "Item {num}. {description}" if description exists
  4. Record (heading, char_offset_of_line_start)
```

Edge cases handled:
- `PART I:   FINANCIAL INFORMATION` → `"PART I — FINANCIAL INFORMATION"`
- `ITEM 7A.  QUANTITATIVE...` → `"Item 7A. QUANTITATIVE..."`
- `Part I: Financial Information` (title case) → matched by case-insensitive pattern
- `ITEM 2  PROPERTIES` (no period) → matched; optional period in pattern
- NT filings → detected via form_type, return empty sections

## Verification

1. Run `scripts/cch_rag.py` on all 152 files
2. Expect: 142+ files with sections, ~10 NT filings with empty sections
3. Print any files where title extraction failed (should be 0)
4. Print any non-NT files where 0 sections found (investigate these)
5. For a few files, manually verify char ranges match actual section content