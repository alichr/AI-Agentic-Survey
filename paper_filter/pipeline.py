from __future__ import annotations

import asyncio
import csv
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from paper_filter.cache import DownloadCache, ProcessCache
from paper_filter.classifier import RelevanceClassifier
from paper_filter.config import Config
from paper_filter.downloader import PDFDownloader
from paper_filter.extractor import PDFExtractor
from paper_filter.fetchers.registry import get_fetcher
from paper_filter.models import Paper

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Download — fetch paper list + download all PDFs
# Output: a folder of PDFs, nothing else
# ═══════════════════════════════════════════════════════════════════════════════

class DownloadPipeline:
    def __init__(self, config: Config, conference: str, year: int) -> None:
        self._config = config
        self._conference = conference.upper()
        self._year = year
        self._cache = DownloadCache(config.pipeline.cache_dir)

    async def run(self) -> str:
        """Fetch + download. Returns path to PDF directory."""
        try:
            await self._fetch()
            pdf_dir = await self._download()
            return pdf_dir
        finally:
            self._cache.close()

    def _get_auth_token(self) -> str:
        """Get an OpenReview auth token if credentials are available."""
        try:
            from paper_filter.fetchers.openreview_api_fetcher import get_openreview_token
            return get_openreview_token()
        except Exception:
            return ""

    async def _fetch(self) -> None:
        existing = self._cache.paper_count(self._conference, self._year)
        if existing > 0:
            logger.info("[FETCH] Found %d cached papers for %s %d", existing, self._conference, self._year)
            return

        logger.info("[FETCH] Fetching papers for %s %d...", self._conference, self._year)
        fetcher = get_fetcher(self._conference, self._year)
        papers = await fetcher.fetch(self._conference, self._year)

        if not papers:
            logger.warning("[FETCH] No papers found")
            return

        self._cache.save_papers(papers)
        logger.info("[FETCH] Saved %d papers", len(papers))

    async def _download(self) -> str:
        pdf_dir = str(
            Path(self._config.pipeline.output_dir) / "pdfs" / f"{self._conference}_{self._year}"
        )
        papers = self._cache.get_undownloaded_papers(self._conference, self._year)

        if not papers:
            logger.info("[DOWNLOAD] All papers already downloaded")
            return pdf_dir

        downloadable = [p for p in papers if p.pdf_url and p.pdf_url.startswith("http")]
        skipped = len(papers) - len(downloadable)
        if skipped:
            logger.info("[DOWNLOAD] Skipping %d papers without valid PDF URLs", skipped)
        if not downloadable:
            return pdf_dir

        logger.info("[DOWNLOAD] Downloading %d papers...", len(downloadable))

        # Use OpenReview auth token if papers are from openreview.net
        auth_token = ""
        if any("openreview.net" in p.pdf_url for p in downloadable[:5]):
            auth_token = self._get_auth_token()
            if auth_token:
                logger.info("[DOWNLOAD] Using authenticated OpenReview session")

        downloader = PDFDownloader(
            config=self._config.download,
            output_dir=self._config.pipeline.output_dir,
            auth_token=auth_token,
        )
        pbar = tqdm(total=len(downloadable), desc="Downloading", unit="pdf")

        async def download_one(paper: Paper) -> None:
            pdf_path = await downloader.download(paper)
            if pdf_path:
                self._cache.save_download(paper.source_id, pdf_path)
            pbar.update(1)

        await asyncio.gather(*[download_one(p) for p in downloadable])
        pbar.close()
        await downloader.close()

        logger.info("[DOWNLOAD] Complete → %s", pdf_dir)
        return pdf_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Process — takes a folder of PDFs, fully independent
# Input: a folder of PDFs (from download, or manually placed)
# Output: CSV + processed/ folder with relevant PDFs
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessPipeline:
    def __init__(
        self,
        config: Config,
        conference: str,
        year: int,
        topic: str,
        topic_description: str | None = None,
        pdf_dir: str | None = None,
    ) -> None:
        self._config = config
        self._conference = conference.upper()
        self._year = year
        self._topic = topic
        self._topic_description = topic_description
        self._cache = ProcessCache(config.pipeline.cache_dir, self._conference, self._year)

        if pdf_dir:
            self._pdf_dir = Path(pdf_dir)
        else:
            self._pdf_dir = Path(config.pipeline.output_dir) / "pdfs" / f"{self._conference}_{self._year}"

    async def run(self) -> str:
        """Scan → Extract → Classify → Cleanup → Output. Returns CSV path."""
        try:
            self._scan()
            await self._extract()
            await self._classify()
            self._cleanup()
            csv_path = self._output()
            return csv_path
        finally:
            self._cache.close()

    def _scan(self) -> None:
        """Register all PDFs in the input folder."""
        if not self._pdf_dir.exists():
            logger.warning("[SCAN] PDF directory not found: %s", self._pdf_dir)
            logger.info("[SCAN] Run 'download' first or provide --pdf-dir")
            return

        pdf_files = sorted(self._pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("[SCAN] No PDFs found in %s", self._pdf_dir)
            return

        new_count = 0
        for pdf_path in pdf_files:
            if self._cache.register_pdf(str(pdf_path)):
                new_count += 1

        logger.info("[SCAN] %d PDFs in %s (%d new)", len(pdf_files), self._pdf_dir, new_count)

    async def _extract(self) -> None:
        unextracted = self._cache.get_unextracted()
        if not unextracted:
            logger.info("[EXTRACT] All PDFs already extracted")
            return

        logger.info("[EXTRACT] Extracting metadata from %d PDFs...", len(unextracted))
        extractor = PDFExtractor(
            config=self._config.vllm,
            max_concurrent=self._config.pipeline.max_classify_concurrent,
        )

        try:
            pbar = tqdm(total=len(unextracted), desc="Extracting", unit="pdf")

            async def extract_one(pdf_path: str) -> None:
                meta = await extractor.extract(pdf_path)
                title = meta.title if meta.title else Path(pdf_path).stem.replace("_", " ")
                self._cache.save_extraction(pdf_path, title, meta.abstract, meta.introduction)
                pbar.update(1)

            await asyncio.gather(*[extract_one(p) for p in unextracted])
            pbar.close()
        finally:
            await extractor.close()

        logger.info("[EXTRACT] Complete")

    async def _classify(self) -> None:
        unclassified = self._cache.get_unclassified()
        if not unclassified:
            logger.info("[CLASSIFY] All papers already classified")
            return

        logger.info("[CLASSIFY] Classifying %d papers for topic: '%s'", len(unclassified), self._topic)
        classifier = RelevanceClassifier(
            config=self._config.vllm,
            topic=self._topic,
            topic_description=self._topic_description,
            max_concurrent=self._config.pipeline.max_classify_concurrent,
        )

        try:
            pbar = tqdm(total=len(unclassified), desc="Classifying", unit="paper")

            async def classify_one(paper: dict) -> None:
                p = Paper(
                    source_id="",
                    title=paper["title"],
                    authors=[],
                    abstract=paper["abstract"],
                    introduction=paper.get("introduction", ""),
                    conference=self._conference,
                    year=self._year,
                )
                score, reasoning = await classifier.classify(p)
                self._cache.save_classification(paper["pdf_path"], score, reasoning)
                pbar.update(1)

            await asyncio.gather(*[classify_one(p) for p in unclassified])
            pbar.close()
        finally:
            await classifier.close()

        logger.info("[CLASSIFY] Complete")

    def _cleanup(self) -> None:
        """Copy relevant PDFs to processed/. Never touches the input PDF folder."""
        threshold = self._config.pipeline.relevance_threshold

        processed_dir = Path(self._config.pipeline.output_dir) / "processed" / f"{self._conference}_{self._year}"
        processed_dir.mkdir(parents=True, exist_ok=True)

        above = self._cache.get_above_threshold(threshold)
        copied = 0
        for paper in above:
            src = Path(paper["pdf_path"])
            if not src.exists():
                continue
            dest = processed_dir / src.name
            if not dest.exists():
                shutil.copy2(str(src), str(dest))
                copied += 1

        total = len(self._cache.get_all())
        relevant = len(above)

        logger.info(
            "[CLEANUP] Copied %d/%d relevant PDFs to %s (threshold %.1f)",
            copied, relevant, processed_dir, threshold,
        )

    def _output(self) -> str:
        all_papers = self._cache.get_all()
        if not all_papers:
            logger.warning("[OUTPUT] No papers to write")
            return ""

        output_dir = Path(self._config.pipeline.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(output_dir / f"{self._conference}_{self._year}_results.csv")

        threshold = self._config.pipeline.relevance_threshold

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "title", "abstract", "introduction", "relevance_score", "relevance_reasoning", "pdf_path",
            ])
            writer.writeheader()
            for paper in all_papers:
                writer.writerow({
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "introduction": paper.get("introduction", ""),
                    "relevance_score": paper.get("relevance_score", ""),
                    "relevance_reasoning": paper.get("relevance_reasoning", ""),
                    "pdf_path": paper.get("pdf_path", ""),
                })

        total = len(all_papers)
        classified = sum(1 for p in all_papers if p.get("relevance_score") is not None)
        relevant = sum(1 for p in all_papers if (p.get("relevance_score") or 0) >= threshold)

        logger.info("[OUTPUT] %d total, %d classified, %d relevant (>= %.1f)", total, classified, relevant, threshold)
        logger.info("[OUTPUT] CSV → %s", csv_path)

        return csv_path
