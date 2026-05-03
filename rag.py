import argparse

from sentence_transformers import SentenceTransformer

from pdf_handler import PDFHandler

EMBEDDING_MODEL = "google/embeddinggemma-300M"


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline: chunk a PDF and embed the chunks.")
    parser.add_argument("--filename", help="Path to the PDF file")
    args = parser.parse_args()

    chunks = PDFHandler().chunk_pdf(args.filename)
    print(f"Created {len(chunks)} chunks")

    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    embeddings = model.encode(chunks, prompt_name="Retrieval-document", show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    return chunks, embeddings


if __name__ == "__main__":
    main()