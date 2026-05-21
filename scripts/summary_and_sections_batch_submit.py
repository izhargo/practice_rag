"""
summary_and_sections_batch_submit.py — Submit filings to OpenAI Batch API for analysis.

Builds a JSONL file of requests (one per filing), uploads it to OpenAI,
creates a batch job, and saves a state file to disk.

Run batch_collect.py afterwards to check results and build the index.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from prompt import FILING_ANALYSIS_PROMPT
from utils import clean_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILINGS_DIR = ROOT / "edgar_data" / "filings"
DEFAULT_STATE_DIR = ROOT / "output" / "batch_state"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 65536

STRUCTURED_FORMAT = {
    "type": "json_schema",
    "name": "filing_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "5-6 sentence filing summary."},
            "title": {"type": "string", "description": "Human-readable filing title with full period."},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "summary": {"type": "string"},
                        "start_text": {"type": "string"},
                        "end_text": {"type": "string"},
                    },
                    "required": ["heading", "summary", "start_text", "end_text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "title", "sections"],
        "additionalProperties": False,
    },
}


def _build_request(
    custom_id: str,
    filing_text: str,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    prompt = FILING_ANALYSIS_PROMPT.format(text=filing_text)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "text": {"format": STRUCTURED_FORMAT},
        },
    }


def _write_jsonl(path: Path, requests: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit filings to OpenAI Batch API for analysis.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_FILINGS_DIR,
                        help="Directory containing .txt filings")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--max-filings", type=int, default=None,
                        help="Max filings per batch (to stay within token quota). Skips already-submitted filings.")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {args.input_dir}")
        return

    # Skip filings that were successfully analyzed in a previous batch
    already_submitted = set()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    for state_file in args.state_dir.glob("batch_state_*.json"):
        prev_state = json.loads(state_file.read_text(encoding="utf-8"))
        for name, meta in prev_state.get("filings", {}).items():
            if meta.get("error") is None and meta.get("analysis") is not None:
                already_submitted.add(name)

    files = [f for f in files if f.stem not in already_submitted]
    if not files:
        print("All filings already submitted in previous batches.")
        return

    if args.max_filings:
        files = files[:args.max_filings]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Model: {args.model}")
    print(f"Filings to submit: {len(files)} (skipped {len(already_submitted)} already submitted)")

    requests: list[dict[str, Any]] = []
    filings_meta: dict[str, Any] = {}

    for path in files:
        filing_name = path.stem
        filing_text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        custom_id = f"filing_{filing_name}"
        requests.append(_build_request(
            custom_id=custom_id,
            filing_text=filing_text,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        ))
        filings_meta[filing_name] = {
            "filing_path": str(path),
            "custom_id": custom_id,
            "analysis": None,
            "error": None,
        }

    # Write JSONL
    jsonl_dir = ROOT / "output" / "batch_jsonl"
    jsonl_path = jsonl_dir / f"batch_filings_{timestamp}.jsonl"
    _write_jsonl(jsonl_path, requests)
    print(f"JSONL written: {jsonl_path}")

    # Upload and submit batch
    client = OpenAI()

    print("Uploading batch file...")
    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded file ID: {uploaded.id}")

    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"Batch ID: {batch.id}  |  Status: {batch.status}")

    # Save state file
    state = {
        "version": 1,
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "timestamp": timestamp,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "input_dir": str(args.input_dir),
        "filings": filings_meta,
    }

    state_path = args.state_dir / f"batch_state_{timestamp}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nState saved: {state_path}")
    print(f"\nRun batch_collect.py --state-file {state_path} to check results.")


if __name__ == "__main__":
    main()