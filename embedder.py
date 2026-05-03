from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model: str, doc_prompt: str | None = None, query_prompt: str | None = None):
        self.model = SentenceTransformer(model, trust_remote_code=True)
        self.doc_prompt = doc_prompt
        self.query_prompt = query_prompt

    def embed_documents(self, chunks: list[str]):
        return self.model.encode(chunks, prompt_name=self.doc_prompt, show_progress_bar=True)

    def embed_query(self, query: str):
        return self.model.encode([query], prompt_name=self.query_prompt)[0]