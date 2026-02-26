"""Multi-view pipeline orchestrator."""

import asyncio
import logging
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.cache.cache_manager import CacheManager
from src.config import Config
from src.embedding.embedding_model import EmbeddingModel
from src.embedding.kmeans_clustering import MultiViewKMeans
from src.extraction.metadata_extractor import MetadataExtractor
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.section_extractor import SectionExtractor
from src.extraction.section_splitter import split_sections
from src.extraction.section_summarizer import SectionSummarizer
from src.external.company_tiers import CompanyTiers
from src.external.openalex import OpenAlexClient
from src.external.semantic_scholar import SemanticScholarClient
from src.external.university_rankings import UniversityRankings
from src.models.paper import (
    Author, PaperMetadata, PaperSections, SectionType, Paper,
)
from src.output.section_diagnostics_csv import write_section_diagnostics_csv
from src.output.csv_writer import write_results_csv
from src.scoring.affiliation_scorer import AffiliationScorer
from src.scoring.citation_scorer import CitationScorer
from src.scoring.hindex_scorer import HIndexScorer

logger = logging.getLogger(__name__)
console = Console()


class Pipeline:
    """Orchestrates the multi-view paper selection pipeline.

    Stages:
        1.  Discovery — scan PDFs, create Paper objects
        2.  Full-Text Extraction — extract all pages via PyMuPDF
        3.  Metadata Extraction — title/abstract/authors/year via vLLM
        4a. Section Splitting — heuristic regex-based splitting (instant)
        4b. Section Summarization — LLM ~400-word summaries per section
        5.  Multi-View Embedding — embed section summaries independently
        6.  K-Means Clustering — K-Means per section view
        7.  Scoring — h-index + affiliation + citation
        8.  Output — CSV report with all raw signals
    """

    def __init__(self, config: Config):
        self.config = config
        self.cache = CacheManager(config.pipeline.cache_dir)
        self.papers: list[Paper] = []
        self.incomplete_papers: list[Paper] = []

    def run(self):
        """Execute the full pipeline."""
        console.rule("[bold blue]Multi-View Paper Selection Pipeline")

        self._stage1_discovery()
        self._stage2_full_text_extraction()
        asyncio.run(self._stage3_metadata_extraction())
        self._stage4a_section_splitting()
        asyncio.run(self._stage4b_section_summarization())
        self._filter_incomplete_papers()
        self._stage5_embedding()
        self._stage6_kmeans_clustering()

        # Stage 7: global quality scores
        self._stage7a_affiliation_scoring()
        asyncio.run(self._stage7b_citation_scoring())
        asyncio.run(self._stage7c_hindex_scoring())

        self._stage8_output()

        self.cache.close()
        console.rule("[bold green]Pipeline Complete")

    # -------------------------------------------------------------------------
    # Stage 1: Discovery
    # -------------------------------------------------------------------------

    def _stage1_discovery(self):
        """Scan for PDFs and create Paper objects."""
        console.rule("[bold]Stage 1: Discovery")

        papers_dir = Path(self.config.pipeline.papers_dir)

        if papers_dir.exists():
            for pdf_path in sorted(papers_dir.rglob("*.pdf")):
                venue = pdf_path.parent.name if pdf_path.parent != papers_dir else "Unknown"
                pdf_hash = CacheManager.compute_pdf_hash(pdf_path)
                self.papers.append(Paper(
                    pdf_path=pdf_path,
                    venue=venue,
                    pdf_hash=pdf_hash,
                ))

        console.print(f"  Found {len(self.papers)} candidate papers")
        logger.info("Discovery: %d candidates", len(self.papers))

    # -------------------------------------------------------------------------
    # Stage 2: Full-Text Extraction
    # -------------------------------------------------------------------------

    def _stage2_full_text_extraction(self):
        """Extract full text from all PDFs."""
        console.rule("[bold]Stage 2: Full-Text Extraction")

        skipped = 0
        extracted = 0

        for paper in tqdm(self.papers, desc="Extracting full text"):
            # Check cache
            cached_text = self.cache.get_full_text(paper.pdf_hash)
            if cached_text is not None:
                paper.full_text = cached_text
                skipped += 1
                continue

            # Extract full text
            text = PDFExtractor.extract_full_text(paper.pdf_path)
            paper.full_text = text
            self.cache.set_full_text(paper.pdf_hash, text)
            extracted += 1

            # Also extract first page for metadata extraction
            if not paper.first_page_text:
                cached_fp = self.cache.get_pdf_text(paper.pdf_hash)
                if cached_fp is not None:
                    paper.first_page_text = cached_fp
                else:
                    fp_text = PDFExtractor.extract_first_page(paper.pdf_path)
                    paper.first_page_text = fp_text
                    self.cache.set_pdf_text(paper.pdf_hash, fp_text)

        # Fill first_page_text for cached full-text papers
        for paper in self.papers:
            if paper.first_page_text is None:
                cached_fp = self.cache.get_pdf_text(paper.pdf_hash)
                if cached_fp is not None:
                    paper.first_page_text = cached_fp
                else:
                    fp_text = PDFExtractor.extract_first_page(paper.pdf_path)
                    paper.first_page_text = fp_text
                    self.cache.set_pdf_text(paper.pdf_hash, fp_text)

        console.print(f"  Extracted: {extracted}, Cached: {skipped}")
        logger.info("Full-text extraction: %d new, %d cached", extracted, skipped)

    # -------------------------------------------------------------------------
    # Stage 3: LLM Metadata Extraction
    # -------------------------------------------------------------------------

    async def _stage3_metadata_extraction(self):
        """Extract metadata from papers using vLLM."""
        console.rule("[bold]Stage 3: LLM Metadata Extraction")

        extractor = MetadataExtractor(
            base_url=self.config.vllm.base_url,
            model_name=self.config.vllm.model_name,
        )

        to_extract = []
        skipped = 0

        for paper in self.papers:
            cached = self.cache.get_metadata(paper.pdf_hash)
            if cached is not None:
                authors = [Author(name=a["name"], affiliation=a.get("affiliation"))
                           for a in cached.get("authors", [])]
                paper.metadata = PaperMetadata(
                    title=cached.get("title"),
                    abstract=cached.get("abstract"),
                    authors=authors,
                    year=cached.get("year"),
                )
                skipped += 1
            else:
                to_extract.append(paper)

        console.print(f"  Need extraction: {len(to_extract)}, Cached: {skipped}")

        if to_extract:
            texts = [(p.pdf_hash, p.first_page_text or "") for p in to_extract]
            results = await extractor.extract_batch(texts)

            success = 0
            failed = 0
            for paper in to_extract:
                metadata = results.get(paper.pdf_hash)
                if metadata:
                    paper.metadata = metadata
                    authors_dicts = [
                        {"name": a.name, "affiliation": a.affiliation}
                        for a in metadata.authors
                    ]
                    self.cache.set_metadata(
                        paper.pdf_hash,
                        metadata.title or "",
                        metadata.abstract or "",
                        authors_dicts,
                        metadata.year,
                    )
                    success += 1
                else:
                    paper.metadata = PaperMetadata()
                    failed += 1

            console.print(f"  Extracted: {success}, Failed: {failed}, Cached: {skipped}")
            logger.info("Metadata extraction: %d success, %d failed, %d cached",
                        success, failed, skipped)
        else:
            console.print(f"  All {skipped} papers cached, no extraction needed")

    # -------------------------------------------------------------------------
    # Stage 4a: Heuristic Section Splitting
    # -------------------------------------------------------------------------

    # Minimum number of non-empty sections from heuristic split before
    # falling back to LLM section-start detection.  Papers with fewer
    # than 5 sections after summarization get filtered out, so use 5
    # here to aggressively route incomplete splits to LLM fallback.
    MIN_HEURISTIC_SECTIONS = 5

    def _stage4a_section_splitting(self):
        """Split full text into 5 sections using heuristic regex, with LLM fallback."""
        console.rule("[bold]Stage 4a: Section Splitting (heuristic + LLM fallback)")

        skipped = 0
        heuristic_ok = 0
        needs_llm = []

        for paper in tqdm(self.papers, desc="Splitting sections"):
            # Check cache
            cached = self.cache.get_sections(paper.pdf_hash)
            if cached is not None:
                paper.sections = PaperSections(
                    title_abstract_conclusion=cached.get("title_abstract_conclusion", ""),
                    introduction=cached.get("introduction", ""),
                    related_work=cached.get("related_work", ""),
                    method=cached.get("method", ""),
                    experiments=cached.get("experiments", ""),
                )
                skipped += 1
                continue

            # Heuristic split — instant, no LLM
            paper.sections = split_sections(paper.full_text or "")

            # Count non-empty sections
            section_names = [
                "title_abstract_conclusion", "introduction",
                "related_work", "method", "experiments",
            ]
            n_populated = sum(
                1 for name in section_names
                if (getattr(paper.sections, name, "") or "").strip()
            )

            if n_populated >= self.MIN_HEURISTIC_SECTIONS:
                heuristic_ok += 1
                self._cache_sections(paper)
            else:
                needs_llm.append(paper)
                logger.info(
                    "Paper '%s' has only %d/5 heuristic sections, queuing LLM fallback",
                    paper.title or paper.pdf_path.name, n_populated,
                )

        console.print(
            f"  Heuristic OK: {heuristic_ok}, Need LLM fallback: {len(needs_llm)}, "
            f"Cached: {skipped}"
        )

        # LLM fallback for papers with too few heuristic sections
        if needs_llm:
            console.print(f"  Running LLM section-start detection for {len(needs_llm)} papers...")
            asyncio.run(self._llm_section_fallback(needs_llm))

        logger.info(
            "Section splitting: %d heuristic, %d LLM fallback, %d cached",
            heuristic_ok, len(needs_llm), skipped,
        )

    async def _llm_section_fallback(self, papers: list[Paper]):
        """Use LLM section-start detection for papers where heuristic failed."""
        sec_cfg = self.config.section_extraction
        extractor = SectionExtractor(
            base_url=self.config.vllm.base_url,
            model_name=self.config.vllm.model_name,
            max_concurrent=sec_cfg.max_concurrent,
            temperature=sec_cfg.temperature,
            max_tokens=sec_cfg.max_tokens,
            max_text_chars=sec_cfg.max_text_chars,
        )

        texts = [(p.pdf_hash, p.full_text or "") for p in papers]
        results = await extractor.extract_batch(texts)

        for paper in papers:
            llm_sections = results.get(paper.pdf_hash)
            if llm_sections:
                # Merge: use LLM result for sections that heuristic missed
                section_names = [
                    "title_abstract_conclusion", "introduction",
                    "related_work", "method", "experiments",
                ]
                for name in section_names:
                    heuristic_text = getattr(paper.sections, name, "") or ""
                    llm_text = getattr(llm_sections, name, "") or ""
                    # Use LLM result if heuristic section is empty
                    if not heuristic_text.strip() and llm_text.strip():
                        setattr(paper.sections, name, llm_text)

            self._cache_sections(paper)

    def _cache_sections(self, paper: Paper):
        """Cache a paper's sections."""
        self.cache.set_sections(
            paper.pdf_hash,
            paper.sections.title_abstract_conclusion or "",
            paper.sections.introduction or "",
            paper.sections.related_work or "",
            paper.sections.method or "",
            paper.sections.experiments or "",
        )

    # -------------------------------------------------------------------------
    # Stage 4b: LLM Section Summarization
    # -------------------------------------------------------------------------

    async def _stage4b_section_summarization(self):
        """Summarize each section (~200 words) using vLLM for embedding."""
        console.rule("[bold]Stage 4b: LLM Section Summarization")

        sec_cfg = self.config.section_extraction
        summarizer = SectionSummarizer(
            base_url=self.config.vllm.base_url,
            model_name=self.config.vllm.model_name,
            max_concurrent=sec_cfg.max_concurrent,
            temperature=sec_cfg.temperature,
            max_tokens=1024,
        )

        section_names = [
            "title_abstract_conclusion", "introduction",
            "related_work", "method", "experiments",
        ]
        section_type_map = {
            "title_abstract_conclusion": SectionType.TITLE_ABSTRACT_CONCLUSION,
            "introduction": SectionType.INTRODUCTION,
            "related_work": SectionType.RELATED_WORK,
            "method": SectionType.METHOD,
            "experiments": SectionType.EXPERIMENTS,
        }

        to_summarize = []
        skipped = 0

        for paper in self.papers:
            cached = self.cache.get_section_summaries(paper.pdf_hash)
            if cached is not None:
                for name in section_names:
                    st = section_type_map[name]
                    summary = cached.get(name, "")
                    if summary:
                        paper.section_summaries[st.value] = summary
                skipped += 1
            else:
                to_summarize.append(paper)

        console.print(f"  Need summarization: {len(to_summarize)}, Cached: {skipped}")

        if to_summarize:
            # Build batch: (pdf_hash, {section_name: raw_text})
            batch = []
            for paper in to_summarize:
                sections_dict = {}
                if paper.sections:
                    for name in section_names:
                        text = getattr(paper.sections, name, "") or ""
                        if text.strip():
                            sections_dict[name] = text
                batch.append((paper.pdf_hash, sections_dict))

            results = await summarizer.summarize_batch(batch)

            success = 0
            for paper in to_summarize:
                summaries = results.get(paper.pdf_hash, {})
                for name in section_names:
                    st = section_type_map[name]
                    summary = summaries.get(name, "")
                    if summary:
                        paper.section_summaries[st.value] = summary

                # Cache
                self.cache.set_section_summaries(
                    paper.pdf_hash,
                    summaries.get("title_abstract_conclusion", ""),
                    summaries.get("introduction", ""),
                    summaries.get("related_work", ""),
                    summaries.get("method", ""),
                    summaries.get("experiments", ""),
                )

                n_summaries = sum(1 for name in section_names if summaries.get(name))
                if n_summaries > 0:
                    success += 1

            console.print(f"  Summarized: {success}/{len(to_summarize)} papers")
            logger.info("Section summarization: %d success, %d cached",
                        success, skipped)
        else:
            console.print(f"  All {skipped} papers cached, no summarization needed")

        # Write section diagnostics CSV for verification (all papers,
        # including those that will be filtered for incomplete sections)
        output_dir = Path(self.config.pipeline.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        diag_path = output_dir / "section_diagnostics.csv"
        write_section_diagnostics_csv(self.papers, diag_path)
        console.print(f"  Section diagnostics written to {diag_path}")

    # -------------------------------------------------------------------------
    # Section Completeness Filter
    # -------------------------------------------------------------------------

    def _filter_incomplete_papers(self):
        """Remove papers with missing section summaries from the active list.

        Papers that don't have summaries for all enabled views are moved to
        self.incomplete_papers and excluded from downstream stages.
        """
        console.rule("[bold]Section Completeness Filter")

        enabled_views = self.config.kmeans.enabled_views
        required_count = len(enabled_views)
        complete = []
        incomplete = []

        for paper in self.papers:
            n_summaries = sum(
                1 for v in enabled_views
                if paper.section_summaries.get(v, "").strip()
            )
            if n_summaries >= required_count:
                complete.append(paper)
            else:
                incomplete.append(paper)

        self.incomplete_papers.extend(incomplete)
        self.papers = complete

        console.print(
            f"  Complete: {len(complete)}, "
            f"Incomplete (removed): {len(incomplete)}"
        )
        if incomplete:
            for p in incomplete[:5]:
                title = (p.title or p.pdf_path.name)[:60]
                n = sum(1 for v in enabled_views
                        if p.section_summaries.get(v, "").strip())
                console.print(f"    - {title} ({n}/5 summaries)")
            if len(incomplete) > 5:
                console.print(f"    ... and {len(incomplete) - 5} more")

        logger.info("Section filter: %d complete, %d incomplete (removed)",
                     len(complete), len(incomplete))

    # -------------------------------------------------------------------------
    # Stage 5: Section Embedding
    # -------------------------------------------------------------------------

    def _stage5_embedding(self):
        """Embed each section summary independently."""
        console.rule("[bold]Stage 5: Section Embedding")

        model = EmbeddingModel(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
        )
        model_name = self.config.embedding.model_name
        enabled_views = self.config.kmeans.enabled_views

        CACHE_KEY_PREFIX = f"{model_name}__summary_v2_view_"

        for view in enabled_views:
            section_type = SectionType(view)
            view_model_key = f"{CACHE_KEY_PREFIX}{view}"

            texts_to_embed = []
            indices_to_embed = []

            for i, paper in enumerate(self.papers):
                cached = self.cache.get_embedding(paper.pdf_hash, view_model_key)
                if cached is not None:
                    paper.section_embeddings[view] = cached.tolist()
                else:
                    text = paper.section_summaries.get(view, "")
                    if not text or len(text.strip()) < 20:
                        if paper.sections:
                            text = paper.sections.get_section(section_type) or ""
                    if not text or len(text.strip()) < 20:
                        text = paper.embedding_text or ""
                    if text and len(text.strip()) >= 20:
                        texts_to_embed.append(text)
                        indices_to_embed.append(i)

            cached_count = len(self.papers) - len(texts_to_embed)
            console.print(f"  View {view} ({section_type.name}): "
                          f"embed {len(texts_to_embed)}, cached {cached_count}")

            if texts_to_embed:
                embeddings = model.embed(texts_to_embed)
                for idx, emb in zip(indices_to_embed, embeddings):
                    self.papers[idx].section_embeddings[view] = emb.tolist()
                    self.cache.set_embedding(
                        self.papers[idx].pdf_hash, emb, view_model_key
                    )

    # -------------------------------------------------------------------------
    # Stage 6: Multi-View K-Means Clustering
    # -------------------------------------------------------------------------

    def _stage6_kmeans_clustering(self):
        """Run independent K-Means per section view."""
        console.rule("[bold]Stage 6: Multi-View K-Means Clustering")

        pca = self.config.kmeans.pca_components or None
        kmeans = MultiViewKMeans(
            n_clusters=self.config.kmeans.n_clusters,
            n_init=self.config.kmeans.n_init,
            random_state=self.config.kmeans.random_state,
            weak_member_percentile=self.config.kmeans.weak_member_percentile,
            pca_components=pca,
        )

        enabled_views = self.config.kmeans.enabled_views

        for view in enabled_views:
            valid_indices = [
                i for i, p in enumerate(self.papers)
                if view in p.section_embeddings
            ]

            if not valid_indices:
                console.print(f"  [yellow]View {view}: no valid embeddings, skipping")
                continue

            embeddings = np.array(
                [self.papers[i].section_embeddings[view] for i in valid_indices],
                dtype=np.float32,
            )

            result = kmeans.fit_predict(embeddings, view)

            # Store per-view cluster assignments
            for j, paper_idx in enumerate(valid_indices):
                paper = self.papers[paper_idx]
                paper.scores.cluster_ids[view] = int(result.cluster_ids[j])
                paper.scores.centroid_distances[view] = float(result.centroid_distances[j])
                paper.scores.weak_member[view] = bool(result.weak_members[j])

            from collections import Counter
            sizes = Counter(int(result.cluster_ids[j]) for j in range(len(valid_indices)))
            console.print(
                f"  View {view}: {len(valid_indices)} papers → "
                f"{result.n_clusters} clusters  "
                f"(sizes: {sorted(sizes.values(), reverse=True)})"
            )

    # -------------------------------------------------------------------------
    # Stage 7: Scoring (affiliation + citation + h-index)
    # -------------------------------------------------------------------------

    def _stage7a_affiliation_scoring(self):
        """Score papers based on author affiliations."""
        console.rule("[bold]Stage 7a: Affiliation Scoring")

        if not self.config.affiliation.enabled:
            console.print("  [yellow]Affiliation scoring disabled")
            return

        uni_rankings = UniversityRankings(self.config.affiliation.university_rankings_path)
        company_tiers = CompanyTiers(self.config.affiliation.company_tiers_path)

        scorer = AffiliationScorer(
            university_rankings=uni_rankings,
            company_tiers=company_tiers,
            first_author_weight=self.config.affiliation.first_author_weight,
            last_author_weight=self.config.affiliation.last_author_weight,
        )

        for paper in self.papers:
            paper.scores.first_author_affiliation_score = scorer.score_author(
                paper.first_author
            )
            paper.scores.last_author_affiliation_score = scorer.score_author(
                paper.last_author
            )

        scored = sum(1 for p in self.papers
                     if p.scores.first_author_affiliation_score is not None)
        console.print(f"  Scored {scored}/{len(self.papers)} papers with affiliation data")

    async def _stage7b_citation_scoring(self):
        """Fetch citations and score papers."""
        console.rule("[bold]Stage 7b: Citation Scoring")

        if not self.config.citation.enabled:
            console.print("  [yellow]Citation scoring disabled")
            return

        source = self.config.citation.source
        fallback_client = None

        if source == "openalex":
            client = OpenAlexClient(
                api_key=self.config.citation.openalex_api_key,
                email=self.config.citation.openalex_email,
                rate_limit_rps=self.config.citation.rate_limit_rps,
                max_retries=self.config.citation.max_retries,
            )
            # Use Semantic Scholar as fallback — it merges preprint/published
            # records and often has higher citation counts.
            fallback_client = SemanticScholarClient(
                api_key=self.config.citation.semantic_scholar_api_key,
                rate_limit_rps=self.config.hindex.rate_limit_rps,  # S2 rate
                max_retries=self.config.citation.max_retries,
            )
            console.print("  Using OpenAlex + Semantic Scholar fallback for citation data")
        else:
            client = SemanticScholarClient(
                api_key=self.config.citation.semantic_scholar_api_key,
                rate_limit_rps=self.config.citation.rate_limit_rps,
                max_retries=self.config.citation.max_retries,
            )
            console.print("  Using Semantic Scholar API for citation data")

        scorer = CitationScorer(
            client=client,
            cache=self.cache,
            expected_citations=self.config.citation.expected_citations,
            expected_citations_slope=self.config.citation.expected_citations_slope,
            cache_max_age_days=self.config.citation.cache_max_age_days,
            fallback_client=fallback_client,
        )

        try:
            results = await scorer.score_batch(self.papers)

            for paper, (score, count) in zip(self.papers, results):
                paper.scores.citation_count = count

            scored = sum(1 for s, _ in results if s is not None)
            console.print(f"  Scored {scored}/{len(self.papers)} papers with citation data")
        finally:
            await client.close()
            if fallback_client:
                await fallback_client.close()

    async def _stage7c_hindex_scoring(self):
        """Fetch author h-indices and score papers."""
        console.rule("[bold]Stage 7c: H-Index Scoring")

        if not self.config.hindex.enabled:
            console.print("  [yellow]H-index scoring disabled")
            return

        # Use OpenAlex for h-index data (10 RPS vs S2's 1 RPS)
        client = OpenAlexClient(
            api_key=self.config.citation.openalex_api_key,
            email=self.config.citation.openalex_email,
            rate_limit_rps=self.config.citation.rate_limit_rps,
            max_retries=self.config.citation.max_retries,
        )

        scorer = HIndexScorer(
            client=client,
            cache=self.cache,
            max_hindex_baseline=self.config.hindex.max_hindex_baseline,
            default_score=self.config.hindex.default_score,
            cache_max_age_days=self.config.hindex.cache_max_age_days,
        )

        try:
            # Build papers_data, reusing paper_ids from citation cache (stage 7b)
            papers_data = []
            from_cache = 0
            need_lookup = []

            for i, paper in enumerate(self.papers):
                title = paper.title or ""
                paper_id = None
                if title:
                    cached = self.cache.get_citation(
                        title, self.config.citation.cache_max_age_days
                    )
                    if cached and cached.get("paper_id"):
                        paper_id = cached["paper_id"]
                        from_cache += 1
                    else:
                        need_lookup.append(i)
                papers_data.append((title, paper_id))

            console.print(
                f"  Paper IDs: {from_cache} from citation cache, "
                f"{len(need_lookup)} need lookup"
            )

            # Look up missing paper_ids via OpenAlex
            if need_lookup:
                async def _fetch_paper_id(idx: int, title: str):
                    result = await client.search_paper(title)
                    if result and result.paper_id:
                        return idx, title, result.paper_id
                    return idx, title, None

                id_tasks = [
                    _fetch_paper_id(i, self.papers[i].title)
                    for i in need_lookup
                    if self.papers[i].title
                ]
                id_results = await asyncio.gather(*id_tasks, return_exceptions=True)
                found = 0
                for r in id_results:
                    if isinstance(r, Exception):
                        logger.error("Paper ID lookup failed: %s", r)
                        continue
                    idx, title, paper_id = r
                    if paper_id:
                        papers_data[idx] = (title, paper_id)
                        found += 1
                console.print(f"  Looked up {found}/{len(need_lookup)} paper IDs via OpenAlex")

            results = await scorer.score_batch(papers_data)

            for paper, (_score, max_hindex) in zip(self.papers, results):
                paper.scores.max_hindex = max_hindex

            scored = sum(1 for s, h in results if h is not None)
            console.print(f"  Scored {scored}/{len(self.papers)} papers with h-index data")

        finally:
            await client.close()

    # -------------------------------------------------------------------------
    # Stage 8: Output
    # -------------------------------------------------------------------------

    def _stage8_output(self):
        """Generate CSV reports with local and global scores."""
        console.rule("[bold]Stage 9: Output")

        output_dir = Path(self.config.pipeline.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write main results CSV
        all_papers = self.papers + self.incomplete_papers
        csv_path = output_dir / "results.csv"
        write_results_csv(all_papers, csv_path)
        console.print(f"  Wrote results to {csv_path} "
                       f"({len(self.papers)} complete + "
                       f"{len(self.incomplete_papers)} incomplete)")

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print a summary table of results."""
        table = Table(title="Pipeline Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        total = len(self.papers) + len(self.incomplete_papers)
        incomplete = len(self.incomplete_papers)

        table.add_row("Total candidates", str(total))
        table.add_row("Scored", str(len(self.papers)))
        table.add_row("Incomplete (filtered)", str(incomplete))

        # H-index summary
        hindices = [p.scores.max_hindex for p in self.papers
                    if p.scores.max_hindex is not None]
        if hindices:
            table.add_row("---", "---")
            table.add_row("[bold]Max H-Index", "")
            table.add_row("  Min", str(min(hindices)))
            table.add_row("  Max", str(max(hindices)))
            table.add_row("  Mean", f"{sum(hindices)/len(hindices):.1f}")

        # Citation summary
        citations = [p.scores.citation_count for p in self.papers
                     if p.scores.citation_count is not None]
        if citations:
            table.add_row("---", "---")
            table.add_row("[bold]Citation Count", "")
            table.add_row("  Min", str(min(citations)))
            table.add_row("  Max", str(max(citations)))
            table.add_row("  Mean", f"{sum(citations)/len(citations):.1f}")

        console.print(table)
