from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from paper_filter.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paper_filter",
        description="Filter academic conference papers by topic relevance using a local LLM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── download ───────────────────────────────────────────────────────────
    dl = subparsers.add_parser(
        "download",
        help="Fetch paper lists and download all PDFs for a conference.",
    )
    dl.add_argument("--conference", "-c", required=True,
                    help="Conference name (neurips, iclr, icml, cvpr, iccv, eccv, aaai, emnlp)")
    dl.add_argument("--year", "-y", required=True, type=int,
                    help="Conference year (2021-2026)")
    dl.add_argument("--config", default=None)
    dl.add_argument("--output-dir", default=None)
    dl.add_argument("--log-level", default=None,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ── process ────────────────────────────────────────────────────────────
    proc = subparsers.add_parser(
        "process",
        help="Extract metadata from PDFs, classify relevance, and output CSV.",
    )
    proc.add_argument("--pdf-dir", required=True,
                      help="Path to folder containing PDFs to process")
    proc.add_argument("--conference", "-c", required=True,
                      help="Conference name (used for output naming)")
    proc.add_argument("--year", "-y", required=True, type=int,
                      help="Conference year (used for output naming)")
    proc.add_argument("--topic", "-t", required=True,
                      help="Research topic (short label)")
    proc.add_argument("--topic-description", "-d", default=None,
                      help="Detailed survey scope description for better LLM precision")
    proc.add_argument("--relevance-threshold", type=float, default=None,
                      help="Minimum score to keep PDFs (default from config)")
    proc.add_argument("--config", default=None)
    proc.add_argument("--output-dir", default=None)
    proc.add_argument("--log-level", default=None,
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def _apply_common(args, config):
    if getattr(args, "output_dir", None) is not None:
        config.pipeline.output_dir = args.output_dir
    if getattr(args, "log_level", None) is not None:
        config.pipeline.log_level = args.log_level
    logging.basicConfig(
        level=getattr(logging, config.pipeline.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_download(args) -> None:
    from paper_filter.pipeline import DownloadPipeline

    config = load_config(args.config)
    _apply_common(args, config)

    pipeline = DownloadPipeline(
        config=config,
        conference=args.conference,
        year=args.year,
    )
    pdf_dir = asyncio.run(pipeline.run())
    if pdf_dir:
        print(f"\nPDFs saved to: {pdf_dir}")
    else:
        print("\nNo PDFs to download (metadata-only conference).", file=sys.stderr)


def cmd_process(args) -> None:
    from paper_filter.pipeline import ProcessPipeline

    config = load_config(args.config)
    _apply_common(args, config)
    if args.relevance_threshold is not None:
        config.pipeline.relevance_threshold = args.relevance_threshold

    pipeline = ProcessPipeline(
        config=config,
        conference=args.conference,
        year=args.year,
        topic=args.topic,
        topic_description=args.topic_description,
        pdf_dir=args.pdf_dir,
    )
    csv_path = asyncio.run(pipeline.run())
    if csv_path:
        print(f"\nResults written to: {csv_path}")
    else:
        print("\nNo results generated.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    if args.command == "download":
        cmd_download(args)
    elif args.command == "process":
        cmd_process(args)


if __name__ == "__main__":
    main()
