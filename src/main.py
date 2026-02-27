"""Entry point for the Multi-View Paper Selection Pipeline."""

import argparse
import logging
import sys

from src.config import Config, load_config
from src.pipeline import Pipeline


def setup_logging(level: str):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by all subcommands."""
    parser.add_argument(
        "--config", "-c",
        default="config/default_config.yaml",
        help="Path to YAML configuration file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )


def _add_scoring_args(parser: argparse.ArgumentParser):
    """Add scoring toggle arguments (used by prepare and full run)."""
    parser.add_argument(
        "--no-hindex", action="store_true",
        help="Disable h-index scoring",
    )
    parser.add_argument(
        "--no-affiliation", action="store_true",
        help="Disable affiliation scoring",
    )
    parser.add_argument(
        "--no-citation", action="store_true",
        help="Disable citation scoring",
    )


def _apply_common_overrides(config: Config, args: argparse.Namespace):
    """Apply common CLI overrides to config."""
    if args.log_level:
        config.pipeline.log_level = args.log_level


def _apply_scoring_overrides(config: Config, args: argparse.Namespace):
    """Apply scoring toggle overrides."""
    if args.no_hindex:
        config.hindex.enabled = False
    if args.no_affiliation:
        config.affiliation.enabled = False
    if args.no_citation:
        config.citation.enabled = False


def main():
    parser = argparse.ArgumentParser(
        description="Multi-View Paper Selection Pipeline - "
                    "Select relevant papers using multi-view K-Means clustering.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── prepare subcommand ──────────────────────────────────────────────
    prep_parser = subparsers.add_parser(
        "prepare",
        help="Extract text, metadata, embeddings, and scores; export a portable bundle.",
    )
    _add_common_args(prep_parser)
    _add_scoring_args(prep_parser)
    prep_parser.add_argument(
        "--export-dir", required=True,
        help="Directory to write papers.json + embeddings.npz",
    )
    prep_parser.add_argument(
        "--papers-dir",
        help="Override papers directory from config",
    )
    prep_parser.add_argument(
        "--output-dir",
        help="Override output directory from config",
    )

    # ── cluster subcommand ──────────────────────────────────────────────
    clust_parser = subparsers.add_parser(
        "cluster",
        help="Import exported bundles, run K-Means clustering, and output CSV.",
    )
    _add_common_args(clust_parser)
    clust_parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing export bundles (scanned recursively for papers.json)",
    )
    clust_parser.add_argument(
        "--output-dir",
        help="Override output directory from config",
    )
    clust_parser.add_argument(
        "--n-clusters", type=int,
        help="Number of K-Means clusters (overrides config)",
    )
    clust_parser.add_argument(
        "--seeds-file",
        help="Path to YAML file with seed topic descriptions for cluster assignment",
    )

    # ── no-subcommand (backward compat) ────────────────────────────────
    _add_common_args(parser)
    _add_scoring_args(parser)
    parser.add_argument(
        "--papers-dir",
        help="Override papers directory from config",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config",
    )
    parser.add_argument(
        "--n-clusters", type=int,
        help="Number of K-Means clusters (default: from config, typically 10)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    _apply_common_overrides(config, args)

    if args.command == "prepare":
        _apply_scoring_overrides(config, args)
        if args.papers_dir:
            config.pipeline.papers_dir = args.papers_dir
        if args.output_dir:
            config.pipeline.output_dir = args.output_dir

        setup_logging(config.pipeline.log_level)
        pipeline = Pipeline(config)
        pipeline.prepare(args.export_dir)

    elif args.command == "cluster":
        if args.output_dir:
            config.pipeline.output_dir = args.output_dir
        if args.n_clusters:
            config.kmeans.n_clusters = args.n_clusters

        setup_logging(config.pipeline.log_level)
        pipeline = Pipeline(config)
        pipeline.cluster(args.input_dir, seeds_file=args.seeds_file)

    else:
        # No subcommand — full pipeline (backward compatible)
        _apply_scoring_overrides(config, args)
        if args.papers_dir:
            config.pipeline.papers_dir = args.papers_dir
        if args.output_dir:
            config.pipeline.output_dir = args.output_dir
        if args.n_clusters:
            config.kmeans.n_clusters = args.n_clusters

        setup_logging(config.pipeline.log_level)
        pipeline = Pipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
