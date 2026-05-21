import argparse
from pathlib import Path

from handlers.embedder import OpenAIEmbedder
from handlers.vector_db_handlers import FaissHandler
from utils import chunk, extract_title

ROOT = Path(__file__).parent.parent
DEFAULT_FILINGS_DIR = ROOT / "edgar_data" / "filings_to_index"
FAISS_PATH = str(ROOT / "cch_index_100.faiss")
DB_PATH = str(ROOT / "cch_index_100.db")


def main():
    parser = argparse.ArgumentParser(description="CCH RAG: index filings with titles prepended to chunks.")
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

        metadata = extract_title(path)
        title = metadata["title"]

        chunks = chunk(str(path))
        if not chunks:
            print(f"[{i}/{total}] EMPTY — {source}")
            continue

        titled_chunks = [f"{title}\n{chunk}" for chunk in chunks]

        embeddings = embedder.embed_documents(titled_chunks)
        db.store(embeddings, titled_chunks, source)
        print(f"[{i}/{total}] OK ({len(chunks)} chunks) — {title}")

    print(f"\nDone. Total vectors in index: {db.index.ntotal}")


if __name__ == "__main__":
    main()