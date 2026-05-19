import argparse
import os
import sys
from pathlib import Path

from handlers.embedder import OpenAIEmbedder
from handlers.vector_db_handlers import FaissHandler
from utils import call_llm_using_openai_client

ROOT = Path(__file__).parent
SIMPLE_FAISS = str(ROOT / "simple_index.faiss")
SIMPLE_DB = str(ROOT / "simple_index.db")
CCH_FAISS = str(ROOT / "cch_index.faiss")
CCH_DB = str(ROOT / "cch_index.db")

MINIMAX_API_KEY = os.environ["MINIMAX_API_KEY"]
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M2.7"

PROMPT_TEMPLATE = """\
Based on the following excerpts from SEC filings, answer the question below.

{chunks}

Question: {question}

Provide your reasoning step by step, then give your final answer."""


def process_question(question: str, embedder: OpenAIEmbedder, simple_db: FaissHandler, cch_db: FaissHandler, out):
    out.write(f"\n{'='*80}\n")
    out.write(f"Question: {question}\n")
    out.write(f"{'='*80}\n")

    query_embedding = embedder.embed_query(question)

    # Simple RAG
    simple_chunks = simple_db.search(query_embedding, k=3)
    out.write(f"\n--- Simple RAG: Top 3 chunks ---\n")
    for j, chunk_text in enumerate(simple_chunks, 1):
        out.write(f"\n[Chunk {j}] ({len(chunk_text)} chars)\n{chunk_text}\n")

    # CCH RAG
    cch_chunks = cch_db.search(query_embedding, k=3)
    out.write(f"\n--- CCH RAG: Top 3 chunks ---\n")
    for j, chunk_text in enumerate(cch_chunks, 1):
        out.write(f"\n[Chunk {j}] ({len(chunk_text)} chars)\n{chunk_text}\n")

    # Call MiniMax for Simple RAG
    simple_context = "\n\n---\n\n".join(simple_chunks)
    simple_prompt = PROMPT_TEMPLATE.format(chunks=simple_context, question=question)
    simple_response = call_llm_using_openai_client(
        prompt=simple_prompt,
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
        model_name=MINIMAX_MODEL,
    )
    out.write(f"\n--- Simple RAG Response ---\n{simple_response}\n")

    # Call MiniMax for CCH RAG
    cch_context = "\n\n---\n\n".join(cch_chunks)
    cch_prompt = PROMPT_TEMPLATE.format(chunks=cch_context, question=question)
    cch_response = call_llm_using_openai_client(
        prompt=cch_prompt,
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
        model_name=MINIMAX_MODEL,
    )
    out.write(f"\n--- CCH RAG Response ---\n{cch_response}\n")


def main():
    parser = argparse.ArgumentParser(description="Query SEC filings using dual RAG indices.")
    parser.add_argument("--questions-file", type=Path,
                        help="Path to a .txt file with questions (one per line)")
    parser.add_argument("--output-file", type=Path,
                        help="Path to a .txt file to write results")
    parser.add_argument("question", nargs="?",
                        help="A question string")
    args = parser.parse_args()

    embedder = OpenAIEmbedder()
    simple_db = FaissHandler(
        vector_size=OpenAIEmbedder.VECTOR_SIZE,
        faiss_path=SIMPLE_FAISS,
        db_path=SIMPLE_DB,
    )
    cch_db = FaissHandler(
        vector_size=OpenAIEmbedder.VECTOR_SIZE,
        faiss_path=CCH_FAISS,
        db_path=CCH_DB,
    )

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
            process_question(question, embedder, simple_db, cch_db, out)
    finally:
        if out is not sys.stdout:
            out.close()
            print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()