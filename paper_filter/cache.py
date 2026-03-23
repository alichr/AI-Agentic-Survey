"""SQLite caches for download and process pipelines (fully independent)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from paper_filter.models import Paper


class DownloadCache:
    """Cache for the download pipeline — tracks fetched papers and download status."""

    def __init__(self, cache_dir: str) -> None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        db_path = str(Path(cache_dir) / "download.db")
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                source_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                abstract TEXT NOT NULL,
                conference TEXT NOT NULL,
                year INTEGER NOT NULL,
                pdf_url TEXT
            );
            CREATE TABLE IF NOT EXISTS downloads (
                source_id TEXT PRIMARY KEY,
                pdf_path TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES papers(source_id)
            );
        """)

    def save_papers(self, papers: list[Paper]) -> None:
        self._conn.executemany(
            """INSERT OR IGNORE INTO papers
               (source_id, title, authors, abstract, conference, year, pdf_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(p.source_id, p.title, p.authors_json(),
              p.abstract, p.conference, p.year, p.pdf_url) for p in papers],
        )
        self._conn.commit()

    def paper_count(self, conference: str, year: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM papers WHERE conference = ? AND year = ?",
            (conference, year),
        ).fetchone()
        return row[0]

    def get_undownloaded_papers(self, conference: str, year: int) -> list[Paper]:
        rows = self._conn.execute(
            """SELECT p.* FROM papers p
               LEFT JOIN downloads d ON p.source_id = d.source_id
               WHERE d.source_id IS NULL AND p.conference = ? AND p.year = ?""",
            (conference, year),
        ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def save_download(self, source_id: str, pdf_path: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO downloads (source_id, pdf_path) VALUES (?, ?)",
            (source_id, pdf_path),
        )
        self._conn.commit()

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        return Paper(
            source_id=row["source_id"],
            title=row["title"],
            authors=Paper.authors_from_json(row["authors"]),
            abstract=row["abstract"],
            conference=row["conference"],
            year=row["year"],
            pdf_url=row["pdf_url"],
        )

    def close(self) -> None:
        self._conn.close()


class ProcessCache:
    """Cache for the process pipeline — tracks extraction and classification of PDFs."""

    def __init__(self, cache_dir: str, conference: str, year: int) -> None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        db_path = str(Path(cache_dir) / f"process_{conference}_{year}.db")
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                pdf_path TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                authors TEXT NOT NULL DEFAULT '[]',
                abstract TEXT NOT NULL DEFAULT '',
                introduction TEXT NOT NULL DEFAULT '',
                conference TEXT NOT NULL DEFAULT '',
                year INTEGER NOT NULL DEFAULT 0,
                pdf_url TEXT DEFAULT '',
                extracted INTEGER NOT NULL DEFAULT 0,
                relevance_score REAL,
                relevance_reasoning TEXT
            );
        """)
        # Migration: add introduction column if missing (existing DBs)
        try:
            self._conn.execute("SELECT introduction FROM papers LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute("ALTER TABLE papers ADD COLUMN introduction TEXT NOT NULL DEFAULT ''")
            self._conn.commit()

    def register_pdf(self, pdf_path: str) -> bool:
        """Register a PDF. Returns True if newly added."""
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO papers (pdf_path) VALUES (?)",
                (pdf_path,),
            )
            self._conn.commit()
            return self._conn.total_changes > 0
        except sqlite3.Error:
            return False

    def get_unextracted(self) -> list[str]:
        """Get PDF paths that haven't been extracted yet."""
        rows = self._conn.execute(
            "SELECT pdf_path FROM papers WHERE extracted = 0"
        ).fetchall()
        return [r["pdf_path"] for r in rows]

    def save_extraction(self, pdf_path: str, title: str, abstract: str, introduction: str = "") -> None:
        self._conn.execute(
            "UPDATE papers SET title = ?, abstract = ?, introduction = ?, extracted = 1 WHERE pdf_path = ?",
            (title, abstract, introduction, pdf_path),
        )
        self._conn.commit()

    def get_unclassified(self) -> list[dict]:
        """Get extracted papers that haven't been classified yet."""
        rows = self._conn.execute(
            "SELECT pdf_path, title, abstract, introduction FROM papers WHERE extracted = 1 AND relevance_score IS NULL"
        ).fetchall()
        return [{"pdf_path": r["pdf_path"], "title": r["title"], "abstract": r["abstract"], "introduction": r["introduction"]} for r in rows]

    def save_classification(self, pdf_path: str, score: float, reasoning: str) -> None:
        self._conn.execute(
            "UPDATE papers SET relevance_score = ?, relevance_reasoning = ? WHERE pdf_path = ?",
            (score, reasoning, pdf_path),
        )
        self._conn.commit()

    def get_all(self) -> list[dict]:
        """Get all papers with their scores, sorted by score descending."""
        rows = self._conn.execute(
            "SELECT * FROM papers ORDER BY relevance_score DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_below_threshold(self, threshold: float) -> list[str]:
        """Get PDF paths of papers below threshold."""
        rows = self._conn.execute(
            "SELECT pdf_path FROM papers WHERE relevance_score IS NOT NULL AND relevance_score < ?",
            (threshold,),
        ).fetchall()
        return [r["pdf_path"] for r in rows]

    def get_above_threshold(self, threshold: float) -> list[dict]:
        """Get papers above threshold."""
        rows = self._conn.execute(
            "SELECT * FROM papers WHERE relevance_score IS NOT NULL AND relevance_score >= ?",
            (threshold,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
