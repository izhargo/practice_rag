import argparse

from embedder import Embedder
from pdf_handler import PDFHandler
from vector_db_handlers import FaissHandler

EMBEDDING_MODEL = "google/embeddinggemma-300M"


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline: chunk a PDF and embed the chunks.")
    parser.add_argument("--filename", help="Path to the PDF file")
    args = parser.parse_args()

    chunks = PDFHandler().chunk_pdf(args.filename)
    print(f"Created {len(chunks)} chunks")

    embedder = Embedder(EMBEDDING_MODEL, doc_prompt="Retrieval-document", query_prompt="Retrieval-query")
    embeddings = embedder.embed_documents(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    db = FaissHandler(vector_size=embeddings.shape[1])
    db.store(embeddings)
    print(f"Stored {db.index.ntotal} vectors in FAISS")

    test_query = "What is the main cause of climate change?"
    query_embedding = embedder.embed_query(test_query)
    indices = db.search(query_embedding, k=2)
    for i, idx in enumerate(indices):
        print(f"\n--- Result {i + 1} ---\n{chunks[idx]}")

    return chunks, db


if __name__ == "__main__":
    main()