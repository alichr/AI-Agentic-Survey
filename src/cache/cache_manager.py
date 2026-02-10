"""SQLite-based cache for pipeline intermediate results."""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """Persistent cache backed by SQLite for PDF text, metadata, embeddings, and citations."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "pipeline_cache.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pdf_text (
                pdf_hash TEXT PRIMARY KEY,
                text_content TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                pdf_hash TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors_json TEXT,
                year INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                pdf_hash TEXT,
                model_name TEXT,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (pdf_hash, model_name)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                title TEXT PRIMARY KEY,
                citation_count INTEGER,
                year INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    @staticmethod
    def compute_pdf_hash(pdf_path: Path) -> str:
        """Compute SHA-256 hash of a PDF file."""
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- PDF text cache --

    def get_pdf_text(self, pdf_hash: str) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT text_content FROM pdf_text WHERE pdf_hash = ?", (pdf_hash,)
        )
        row = cur.fetchone()
        return row["text_content"] if row else None

    def set_pdf_text(self, pdf_hash: str, text: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO pdf_text (pdf_hash, text_content) VALUES (?, ?)",
            (pdf_hash, text),
        )
        self.conn.commit()

    # -- Metadata cache --

    def get_metadata(self, pdf_hash: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT title, abstract, authors_json, year FROM metadata WHERE pdf_hash = ?",
            (pdf_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        authors = json.loads(row["authors_json"]) if row["authors_json"] else []
        return {
            "title": row["title"],
            "abstract": row["abstract"],
            "authors": authors,
            "year": row["year"],
        }

    def set_metadata(self, pdf_hash: str, title: str, abstract: str,
                     authors: list[dict], year: Optional[int]):
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (pdf_hash, title, abstract, authors_json, year) "
            "VALUES (?, ?, ?, ?, ?)",
            (pdf_hash, title, abstract, json.dumps(authors), year),
        )
        self.conn.commit()

    # -- Embedding cache --

    def get_embedding(self, pdf_hash: str, model_name: str) -> Optional[np.ndarray]:
        cur = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE pdf_hash = ? AND model_name = ?",
            (pdf_hash, model_name),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32)

    def set_embedding(self, pdf_hash: str, embedding: np.ndarray, model_name: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (pdf_hash, model_name, embedding) "
            "VALUES (?, ?, ?)",
            (pdf_hash, model_name, embedding.astype(np.float32).tobytes()),
        )
        self.conn.commit()

    # -- Citation cache --

    def get_citation(self, title: str, max_age_days: int = 30) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT citation_count, year, created_at FROM citations WHERE title = ?",
            (title,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        created = datetime.fromisoformat(row["created_at"])
        age_days = (datetime.now() - created).days
        if age_days > max_age_days:
            return None
        return {"citation_count": row["citation_count"], "year": row["year"]}

    def set_citation(self, title: str, citation_count: int, year: Optional[int]):
        self.conn.execute(
            "INSERT OR REPLACE INTO citations (title, citation_count, year, created_at) "
            "VALUES (?, ?, ?, ?)",
            (title, citation_count, year, datetime.now().isoformat()),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
