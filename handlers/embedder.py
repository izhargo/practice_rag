import time

import numpy as np
from openai import OpenAI


class OpenAIEmbedder:
    VECTOR_SIZE = 1536  # text-embedding-3-small output dimension

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def embed_documents(self, chunks: list[str], batch_size: int = 100) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            for attempt in range(5):
                try:
                    response = self.client.embeddings.create(input=batch, model=self.model)
                    all_embeddings.extend(item.embedding for item in response.data)
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait = 2 ** attempt
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            else:
                raise RuntimeError(f"Failed after 5 retries on batch starting at index {i}")
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(input=[query], model=self.model)
        return np.array(response.data[0].embedding, dtype=np.float32)
