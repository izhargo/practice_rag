from abc import ABC, abstractmethod

import faiss
import numpy as np


class VectorDBHandler(ABC):
    @abstractmethod
    def store(self, embeddings: np.ndarray) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> np.ndarray:
        pass


class FaissHandler(VectorDBHandler):
    def __init__(self, vector_size: int):
        self.index = faiss.IndexFlatL2(vector_size)

    def store(self, embeddings: np.ndarray) -> None:
        self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, k: int) -> np.ndarray:
        _, indices = self.index.search(query_embedding.astype(np.float32).reshape(1, -1), k)
        return indices[0]