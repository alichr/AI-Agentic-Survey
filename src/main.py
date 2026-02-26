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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-View Paper Selection Pipeline - "
                    "Select relevant papers using multi-view K-Means clustering "
                    "and consensus fusion.",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/default_config.yaml",
        help="Path to YAML configuration file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--papers-dir",
        help="Override papers directory from config",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )
    parser.add_argument(
        "--no-hindex",
        action="store_true",
        help="Disable h-index scoring",
    )
    parser.add_argument(
        "--no-affiliation",
        action="store_true",
        help="Disable affiliation scoring",
    )
    parser.add_argument(
        "--no-citation",
        action="store_true",
        help="Disable citation scoring",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        help="Number of K-Means clusters (default: from config, typically 10)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.papers_dir:
        config.pipeline.papers_dir = args.papers_dir
    if args.output_dir:
        config.pipeline.output_dir = args.output_dir
    if args.log_level:
        config.pipeline.log_level = args.log_level
    if args.no_hindex:
        config.hindex.enabled = False
    if args.no_affiliation:
        config.affiliation.enabled = False
    if args.no_citation:
        config.citation.enabled = False
    if args.n_clusters:
        config.kmeans.n_clusters = args.n_clusters

    # Setup logging
    setup_logging(config.pipeline.log_level)

    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
