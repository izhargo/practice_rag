import argparse
import os
import sys
from pathlib import Path

from handlers.embedder import OpenAIEmbedder
from handlers.vector_db_handlers import FaissHandler
from utils import call_llm_using_openai_client

ROOT = Path(__file__).parent.parent

INDICES = {
    "simple": {
        "label": "Simple RAG",
        "faiss": str(ROOT / "simple_index_100.faiss"),
        "db": str(ROOT / "simple_index_100.db"),
    },
    "cch": {
        "label": "CCH RAG",
        "faiss": str(ROOT / "cch_index_100.faiss"),
        "db": str(ROOT / "cch_index_100.db"),
    },
    "cch-sections": {
        "label": "CCH Sections RAG",
        "faiss": str(ROOT / "cch_sections_index.faiss"),
        "db": str(ROOT / "cch_sections_index.db"),
    },
    "cch-ss": {
        "label": "CCH Summary+Sections RAG",
        "faiss": str(ROOT / "cch_summary_sections_index.faiss"),
        "db": str(ROOT / "cch_summary_sections_index.db"),
    },
}

MINIMAX_API_KEY = os.environ["MINIMAX_API_KEY"]
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M2.7"

PROMPT_TEMPLATE = """\
Based on the following excerpts from SEC filings, answer the question below.

{chunks}

Question: {question}

Provide your reasoning step by step, then give your final answer."""


def process_question(question: str, embedder: OpenAIEmbedder, active_indices: list[tuple[str, FaissHandler]], out):
    out.write(f"\n{'='*80}\n")
    out.write(f"Question: {question}\n")
    out.write(f"{'='*80}\n")

    query_embedding = embedder.embed_query(question)

    results = []
    for label, db in active_indices:
        chunks = db.search(query_embedding, k=5)
        results.append((label, chunks))

    # Print all retrieved chunks first
    for label, chunks in results:
        out.write(f"\n--- {label}: Top 5 chunks ---\n")
        for j, chunk_text in enumerate(chunks, 1):
            out.write(f"\n[Chunk {j}] ({len(chunk_text)} chars)\n{chunk_text}\n")

    # Print all model responses
    out.write(f"\n{'~'*80}\n")
    out.write(f"MODEL RESPONSES\n")
    out.write(f"{'~'*80}\n")

    for label, chunks in results:
        context = "\n\n---\n\n".join(chunks)
        prompt = PROMPT_TEMPLATE.format(chunks=context, question=question)
        response = call_llm_using_openai_client(
            prompt=prompt,
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
            model_name=MINIMAX_MODEL,
        )
        out.write(f"\n--- {label} Response ---\n{response}\n")


def main():
    parser = argparse.ArgumentParser(description="Query SEC filings using RAG indices.")
    parser.add_argument("--simple", action="store_true", help="Use Simple RAG index")
    parser.add_argument("--cch", action="store_true", help="Use CCH RAG index")
    parser.add_argument("--cch-sections", action="store_true", help="Use CCH Sections RAG index (no section summary)")
    parser.add_argument("--cch-ss", action="store_true", help="Use CCH Summary+Sections RAG index")
    parser.add_argument("--questions-file", type=Path,
                        help="Path to a .txt file with questions (one per line)")
    parser.add_argument("--output-file", type=Path,
                        help="Path to a .txt file to write results")
    parser.add_argument("question", nargs="?",
                        help="A question string")
    args = parser.parse_args()

    selected = []
    for key in ("simple", "cch", "cch_sections", "cch_ss"):
        if getattr(args, key):
            idx = INDICES[key.replace("_", "-")]
            selected.append(idx)

    if not selected:
        print("Error: specify at least one index flag: --simple, --cch, --cch-sections, --cch-ss")
        sys.exit(1)

    embedder = OpenAIEmbedder()
    active_indices = []
    for idx in selected:
        db = FaissHandler(
            vector_size=OpenAIEmbedder.VECTOR_SIZE,
            faiss_path=idx["faiss"],
            db_path=idx["db"],
        )
        active_indices.append((idx["label"], db))

    if args.questions_file:
        questions = [line.strip() for line in args.questions_file.read_text().splitlines() if line.strip()]
    elif args.question:
        questions = [args.question]
    else:
        print("Enter your question (or 'quit' to exit):")
        questions = []
        while True:
            q = input("> ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                questions.append(q)

    out = open(args.output_file, "w") if args.output_file else sys.stdout

    try:
        for question in questions:
            process_question(question, embedder, active_indices, out)
    finally:
        if out is not sys.stdout:
            out.close()
            print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()