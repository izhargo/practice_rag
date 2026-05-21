"""
summary_and_sections_batch_collect.py — Collect batch results and save analyses.

Checks the batch status. If completed:
  1. Downloads and parses all filing analyses
  2. Saves each analysis as a JSON file to output/analyses/

If not ready: prints the current status and exits.

Usage:
    python scripts/summary_and_sections_batch_collect.py --state-file output/batch_state/batch_state_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSES_DIR = ROOT / "output" / "analyses"


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict, path: Path) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def download_results(client, output_file_id: str) -> list[dict]:
    content = client.files.content(output_file_id).text
    return [json.loads(line) for line in content.strip().splitlines()]


def extract_output_text(response_body: dict) -> str:
    """Extract the last non-empty output_text from the response.

    Some responses contain multiple message items with empty text
    before the actual content — take the last non-empty one.
    """
    last_text = ""
    for item in response_body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    last_text = content["text"]
    return last_text


def process_results(client, state: dict, state_path: Path, output_file_id: str) -> None:
    print("\nDownloading batch results...")
    results = download_results(client, output_file_id)
    results_by_id = {r["custom_id"]: r for r in results}

    parsed: dict[str, dict] = {}
    failed = 0

    for filing_name, meta in state["filings"].items():
        custom_id = meta["custom_id"]
        result = results_by_id.get(custom_id)

        if result is None:
            meta["error"] = "no result returned"
            print(f"  [{filing_name}] MISSING")
            failed += 1
            continue

        if result.get("error"):
            meta["error"] = str(result["error"])
            print(f"  [{filing_name}] FAILED — {result['error']}")
            failed += 1
            continue

        response_body = result.get("response", {}).get("body", {})
        output_text = extract_output_text(response_body)

        if not output_text:
            debug_path = ANALYSES_DIR / f"{filing_name}_raw_response.json"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            meta["error"] = "empty output text"
            print(f"  [{filing_name}] EMPTY RESPONSE — raw saved to {debug_path}")
            failed += 1
            continue

        try:
            analysis = json.loads(output_text)
        except json.JSONDecodeError as e:
            meta["error"] = f"JSON parse error: {e}"
            print(f"  [{filing_name}] PARSE ERROR — {e}")
            failed += 1
            continue

        n_sections = len(analysis.get("sections", []))
        parsed[filing_name] = analysis
        print(f"  [{filing_name}] OK — {n_sections} sections")

    print(f"\nParsed: {len(parsed)} succeeded, {failed} failed")

    if failed > 0:
        print(f"\n{failed} filing(s) failed. Continuing with successful ones.")

    # Save analyses to disk
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    for filing_name, analysis in parsed.items():
        analysis_path = ANALYSES_DIR / f"{filing_name}.json"
        analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Analyses saved to {ANALYSES_DIR}")

    # Update state
    for filing_name, analysis in parsed.items():
        state["filings"][filing_name]["analysis"] = analysis
    state["status"] = "collected"
    save_state(state, state_path)
    print(f"State updated: {state_path}")


def main() -> None:
    from openai import OpenAI

    parser = argparse.ArgumentParser(description="Collect batch results and save analyses.")
    parser.add_argument("--state-file", type=Path, required=True)
    args = parser.parse_args()

    if not args.state_file.exists():
        print(f"State file not found: {args.state_file}", file=sys.stderr)
        sys.exit(1)

    state = load_state(args.state_file)
    client = OpenAI()
    batch = client.batches.retrieve(state["batch_id"])

    print(f"Batch ID: {batch.id}")
    print(f"Status:   {batch.status}")
    if batch.request_counts:
        print(f"Requests: {batch.request_counts.completed}/{batch.request_counts.total} completed, "
              f"{batch.request_counts.failed} failed")

    if batch.status in ("validating", "in_progress", "finalizing"):
        print("\nNot ready yet — run again later.")
        return

    if batch.status == "failed":
        print(f"\nBatch failed: {batch.errors}", file=sys.stderr)
        sys.exit(1)

    if batch.status == "cancelled":
        print("\nBatch was cancelled.", file=sys.stderr)
        sys.exit(1)

    if batch.status != "completed":
        print(f"\nUnexpected status: {batch.status}", file=sys.stderr)
        sys.exit(1)

    if not batch.output_file_id:
        # All requests failed — mark them in state and save
        print("\nBatch completed but no output file (all requests failed).")
        if batch.error_file_id:
            error_content = client.files.content(batch.error_file_id).text
            errors_by_id = {}
            for line in error_content.strip().splitlines():
                err = json.loads(line)
                errors_by_id[err["custom_id"]] = err
            for filing_name, meta in state["filings"].items():
                err = errors_by_id.get(meta["custom_id"])
                if err:
                    error_msg = err.get("response", {}).get("body", {}).get("error", {}).get("message", "unknown error")
                    meta["error"] = error_msg
                    print(f"  [{filing_name}] {error_msg}")
                else:
                    meta["error"] = "no result and no error details"
        else:
            for meta in state["filings"].values():
                meta["error"] = "batch produced no output"
        save_state(state, args.state_file)
        print(f"State updated: {args.state_file}")
        sys.exit(1)

    process_results(client, state, args.state_file, batch.output_file_id)


if __name__ == "__main__":
    main()