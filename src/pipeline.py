"""Pipeline orchestrator: coordinates all stages of paper selection."""

import asyncio
import logging
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.cache.cache_manager import CacheManager
from src.config import Config, load_seed_papers_config
from src.embedding.clustering import PaperClusterer
from src.embedding.embedding_model import EmbeddingModel
from src.extraction.metadata_extractor import MetadataExtractor
from src.extraction.pdf_extractor import PDFExtractor
from src.external.company_tiers import CompanyTiers
from src.external.semantic_scholar import SemanticScholarClient
from src.external.university_rankings import UniversityRankings
from src.models.paper import Author, Paper, PaperMetadata
from src.output.csv_writer import write_results_csv
from src.output.file_organizer import organize_accepted_papers
from src.scoring.affiliation_scorer import AffiliationScorer
from src.scoring.aggregator import ScoreAggregator
from src.scoring.citation_scorer import CitationScorer
from src.scoring.relevance_scorer import RelevanceScorer

logger = logging.getLogger(__name__)
console = Console()


class Pipeline:
    """Orchestrates the full paper selection pipeline.

    Stages:
        1. Discovery - find all PDFs
        2. PDF text extraction
        3. LLM metadata extraction
        4. Embedding + clustering (if relevance enabled)
        5. Affiliation scoring (if enabled)
        6. Citation scoring (if enabled)
        7. Aggregation & decision
        8. Output
    """

    def __init__(self, config: Config):
        self.config = config
        self.cache = CacheManager(config.pipeline.cache_dir)
        self.papers: list[Paper] = []
        self.seed_papers: list[Paper] = []

    def run(self):
        """Execute the full pipeline."""
        console.rule("[bold blue]AI Survey Paper Selection Pipeline")

        self._stage1_discovery()
        self._stage2_pdf_extraction()
        asyncio.run(self._stage3_metadata_extraction())

        if self.config.relevance.enabled:
            self._stage4_embedding_clustering()

        if self.config.affiliation.enabled:
            self._stage5_affiliation_scoring()

        if self.config.citation.enabled:
            asyncio.run(self._stage6_citation_scoring())

        self._stage7_aggregation()
        self._stage8_output()

        self.cache.close()
        console.rule("[bold green]Pipeline Complete")

    # -------------------------------------------------------------------------
    # Stage 1: Discovery
    # -------------------------------------------------------------------------

    def _stage1_discovery(self):
        """Scan for PDFs and load seed paper config."""
        console.rule("[bold]Stage 1: Discovery")

        papers_dir = Path(self.config.pipeline.papers_dir)
        seed_dir = Path(self.config.pipeline.seed_papers_dir)
        seed_config = load_seed_papers_config(self.config.pipeline.seed_papers_config)

        # Discover candidate papers
        if papers_dir.exists():
            for pdf_path in sorted(papers_dir.rglob("*.pdf")):
                venue = pdf_path.parent.name if pdf_path.parent != papers_dir else "Unknown"
                pdf_hash = CacheManager.compute_pdf_hash(pdf_path)
                self.papers.append(Paper(
                    pdf_path=pdf_path,
                    venue=venue,
                    pdf_hash=pdf_hash,
                ))

        # Discover seed papers
        if seed_dir.exists():
            for pdf_path in sorted(seed_dir.rglob("*.pdf")):
                pdf_hash = CacheManager.compute_pdf_hash(pdf_path)
                label = seed_config.get(pdf_path.name)
                self.seed_papers.append(Paper(
                    pdf_path=pdf_path,
                    venue="seed",
                    pdf_hash=pdf_hash,
                    is_seed=True,
                    seed_label=label,
                ))

        console.print(f"  Found {len(self.papers)} candidate papers")
        console.print(f"  Found {len(self.seed_papers)} seed papers")
        logger.info("Discovery: %d candidates, %d seeds", len(self.papers), len(self.seed_papers))

    # -------------------------------------------------------------------------
    # Stage 2: PDF Text Extraction
    # -------------------------------------------------------------------------

    def _stage2_pdf_extraction(self):
        """Extract first-page text from all PDFs."""
        console.rule("[bold]Stage 2: PDF Text Extraction")

        all_papers = self.seed_papers + self.papers
        skipped = 0
        extracted = 0

        for paper in tqdm(all_papers, desc="Extracting PDF text"):
            # Check cache
            cached_text = self.cache.get_pdf_text(paper.pdf_hash)
            if cached_text is not None:
                paper.first_page_text = cached_text
                skipped += 1
                continue

            # Extract
            text = PDFExtractor.extract_first_page(paper.pdf_path)
            paper.first_page_text = text
            self.cache.set_pdf_text(paper.pdf_hash, text)
            extracted += 1

        console.print(f"  Extracted: {extracted}, Cached: {skipped}")
        logger.info("PDF extraction: %d new, %d cached", extracted, skipped)

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

        all_papers = self.seed_papers + self.papers
        to_extract = []
        skipped = 0

        for paper in all_papers:
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
            logger.info("Metadata extraction: %d success, %d failed, %d cached", success, failed, skipped)
        else:
            console.print(f"  All {skipped} papers cached, no extraction needed")
            logger.info("Metadata extraction: 0 new, %d cached", skipped)

    # -------------------------------------------------------------------------
    # Stage 4: Embedding + Clustering
    # -------------------------------------------------------------------------

    def _stage4_embedding_clustering(self):
        """Compute embeddings and perform GMM clustering."""
        console.rule("[bold]Stage 4: Embedding & Clustering")

        model = EmbeddingModel(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
        )

        all_papers = self.seed_papers + self.papers
        model_name = self.config.embedding.model_name

        # Collect texts and check cache
        texts_to_embed = []
        indices_to_embed = []

        for i, paper in enumerate(all_papers):
            cached = self.cache.get_embedding(paper.pdf_hash, model_name)
            if cached is not None:
                paper.embedding = cached.tolist()
            else:
                text = paper.embedding_text
                if text:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)

        console.print(f"  Need embedding: {len(texts_to_embed)}, "
                       f"Cached: {len(all_papers) - len(texts_to_embed)}")

        # Compute new embeddings
        if texts_to_embed:
            embeddings = model.embed(texts_to_embed)
            for idx, emb in zip(indices_to_embed, embeddings):
                all_papers[idx].embedding = emb.tolist()
                self.cache.set_embedding(
                    all_papers[idx].pdf_hash, emb, model_name
                )

        # Build embedding matrix for clustering
        valid_papers = [p for p in all_papers if p.embedding is not None]
        if not valid_papers:
            console.print("  [red]No papers with valid embeddings, skipping clustering")
            return

        embedding_matrix = np.array([p.embedding for p in valid_papers], dtype=np.float32)

        # Identify seed paper indices and labels in the valid set
        seed_indices = []
        seed_labels = []
        for i, p in enumerate(valid_papers):
            if p.is_seed and p.seed_label:
                seed_indices.append(i)
                seed_labels.append(p.seed_label)

        # Cluster
        clusterer = PaperClusterer(
            n_clusters=self.config.clustering.n_clusters,
        )
        result = clusterer.fit_predict(embedding_matrix, seed_indices, seed_labels)

        # Apply relevance scores
        relevance_scores = RelevanceScorer.score(result.cluster_probabilities)

        for i, paper in enumerate(valid_papers):
            paper.scores.cluster_id = int(result.cluster_ids[i])
            paper.scores.cluster_label = result.cluster_labels.get(
                int(result.cluster_ids[i]), "Unknown"
            )
            paper.scores.relevance_score = relevance_scores[i]

        console.print(f"  Clustered {len(valid_papers)} papers into "
                       f"{len(result.cluster_labels)} clusters")

    # -------------------------------------------------------------------------
    # Stage 5: Affiliation Scoring
    # -------------------------------------------------------------------------

    def _stage5_affiliation_scoring(self):
        """Score papers based on author affiliations."""
        console.rule("[bold]Stage 5: Affiliation Scoring")

        uni_rankings = UniversityRankings(self.config.affiliation.university_rankings_path)
        company_tiers = CompanyTiers(self.config.affiliation.company_tiers_path)

        scorer = AffiliationScorer(
            university_rankings=uni_rankings,
            company_tiers=company_tiers,
            first_author_weight=self.config.affiliation.first_author_weight,
            last_author_weight=self.config.affiliation.last_author_weight,
        )

        for paper in self.papers:
            paper.scores.affiliation_score = scorer.score(
                paper.first_author, paper.last_author
            )

        scores = [p.scores.affiliation_score for p in self.papers
                   if p.scores.affiliation_score is not None]
        if scores:
            console.print(f"  Affiliation scores: min={min(scores):.3f}, "
                           f"max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}")

    # -------------------------------------------------------------------------
    # Stage 6: Citation Scoring
    # -------------------------------------------------------------------------

    async def _stage6_citation_scoring(self):
        """Fetch citations and score papers."""
        console.rule("[bold]Stage 6: Citation + Recency Scoring")

        client = SemanticScholarClient(
            api_key=self.config.citation.semantic_scholar_api_key,
            rate_limit_rps=self.config.citation.rate_limit_rps,
            max_retries=self.config.citation.max_retries,
        )

        scorer = CitationScorer(
            client=client,
            cache=self.cache,
            expected_citations=self.config.citation.expected_citations,
            expected_citations_slope=self.config.citation.expected_citations_slope,
            cache_max_age_days=self.config.citation.cache_max_age_days,
        )

        try:
            results = await scorer.score_batch(self.papers)

            for paper, (score, count) in zip(self.papers, results):
                paper.scores.citation_score = score
                paper.scores.citation_count = count

            scored = sum(1 for s, _ in results if s is not None)
            console.print(f"  Scored {scored}/{len(self.papers)} papers with citation data")

        finally:
            await client.close()

    # -------------------------------------------------------------------------
    # Stage 7: Aggregation
    # -------------------------------------------------------------------------

    def _stage7_aggregation(self):
        """Apply threshold-based acceptance decisions."""
        console.rule("[bold]Stage 7: Aggregation & Decision")

        aggregator = ScoreAggregator(self.config)
        aggregator.decide_all(self.papers)

        accepted = sum(1 for p in self.papers if p.accepted)
        console.print(f"  Accepted: {accepted}/{len(self.papers)} papers")

    # -------------------------------------------------------------------------
    # Stage 8: Output
    # -------------------------------------------------------------------------

    def _stage8_output(self):
        """Generate CSV report and organize accepted papers."""
        console.rule("[bold]Stage 8: Output")

        output_dir = Path(self.config.pipeline.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write CSV
        csv_path = output_dir / "results.csv"
        write_results_csv(self.papers, csv_path)

        # Copy accepted papers
        copied = organize_accepted_papers(self.papers, output_dir)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print a summary table of results."""
        table = Table(title="Pipeline Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        total = len(self.papers)
        accepted = sum(1 for p in self.papers if p.accepted)
        rejected = total - accepted

        table.add_row("Total candidates", str(total))
        table.add_row("Seed papers", str(len(self.seed_papers)))
        table.add_row("Accepted", str(accepted))
        table.add_row("Rejected", str(rejected))

        if total > 0:
            table.add_row("Acceptance rate", f"{100 * accepted / total:.1f}%")

        # Cluster distribution
        if self.config.relevance.enabled:
            cluster_counts: dict[str, int] = {}
            for p in self.papers:
                if p.accepted and p.scores.cluster_label:
                    label = p.scores.cluster_label
                    cluster_counts[label] = cluster_counts.get(label, 0) + 1
            if cluster_counts:
                table.add_row("---", "---")
                table.add_row("[bold]Cluster Distribution", "[bold]Accepted")
                for label, count in sorted(cluster_counts.items(), key=lambda x: -x[1]):
                    table.add_row(f"  {label}", str(count))

        # Score summaries
        for name, attr in [("Relevance", "relevance_score"),
                           ("Affiliation", "affiliation_score"),
                           ("Citation", "citation_score")]:
            scores = [getattr(p.scores, attr) for p in self.papers
                      if getattr(p.scores, attr) is not None]
            if scores:
                table.add_row("---", "---")
                table.add_row(f"[bold]{name} scores", "")
                table.add_row(f"  Min", f"{min(scores):.3f}")
                table.add_row(f"  Max", f"{max(scores):.3f}")
                table.add_row(f"  Mean", f"{sum(scores)/len(scores):.3f}")

        console.print(table)
