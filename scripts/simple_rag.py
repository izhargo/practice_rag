import argparse
from pathlib import Path

from handlers.embedder import OpenAIEmbedder
from handlers.vector_db_handlers import FaissHandler
from utils import chunk

ROOT = Path(__file__).parent.parent
DEFAULT_FILINGS_DIR = ROOT / "edgar_data" / "filings_to_index"
FAISS_PATH = str(ROOT / "simple_index.faiss")
DB_PATH = str(ROOT / "simple_index.db")


def main():
    parser = argparse.ArgumentParser(description="Simple RAG: index filings as plain chunks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_FILINGS_DIR,
                        help="Directory containing .txt filings")
    args = parser.parse_args()

    embedder = OpenAIEmbedder()
    db = FaissHandler(
        vector_size=OpenAIEmbedder.VECTOR_SIZE,
        faiss_path=FAISS_PATH,
        db_path=DB_PATH,
    )
    files = sorted(args.input_dir.glob("*.txt"))
    total = len(files)
    print(f"Found {total} filings in {args.input_dir}")

    for i, path in enumerate(files, 1):
        source = path.name
        if db.is_indexed(source):
            print(f"[{i}/{total}] SKIP — {source}")
            continue

        chunks = chunk(str(path))
        if not chunks:
            print(f"[{i}/{total}] EMPTY — {source}")
            continue

        embeddings = embedder.embed_documents(chunks)
        db.store(embeddings, chunks, source)
        print(f"[{i}/{total}] OK ({len(chunks)} chunks) — {source}")

    print(f"\nDone. Total vectors in index: {db.index.ntotal}")


if __name__ == "__main__":
    main()
