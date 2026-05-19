import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np


class VectorDBHandler(ABC):
    @abstractmethod
    def store(self, embeddings: np.ndarray, chunks: list[str], source_file: str) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> list[str]:
        pass


class FaissHandler(VectorDBHandler):
    def __init__(self, vector_size: int, faiss_path: str, db_path: str):
        self.faiss_path = Path(faiss_path)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
        else:
            self.index = faiss.IndexFlatIP(vector_size)

    def _init_db(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                source_file TEXT    NOT NULL,
                chunk_text  TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS indexed_files (
                source_file TEXT PRIMARY KEY
            );
        """)
        self.conn.commit()

    def is_indexed(self, source_file: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM indexed_files WHERE source_file = ?", (source_file,)
        ).fetchone()
        return row is not None

    def store(self, embeddings: np.ndarray, chunks: list[str], source_file: str) -> None:
        start_id = self.index.ntotal
        self.conn.executemany(
            "INSERT INTO chunks (id, source_file, chunk_text) VALUES (?, ?, ?)",
            [(start_id + i, source_file, chunk) for i, chunk in enumerate(chunks)],
        )
        self.conn.execute(
            "INSERT INTO indexed_files (source_file) VALUES (?)", (source_file,)
        )
        self.conn.commit()
        self.index.add(embeddings.astype(np.float32))
        faiss.write_index(self.index, str(self.faiss_path))

    def search(self, query_embedding: np.ndarray, k: int) -> list[str]:
        _, indices = self.index.search(
            query_embedding.astype(np.float32).reshape(1, -1), k
        )
        ids = indices[0].tolist()
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, chunk_text FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
        id_to_text = {r[0]: r[1] for r in rows}
        return [id_to_text[i] for i in ids if i in id_to_text]