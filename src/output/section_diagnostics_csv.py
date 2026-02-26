"""Write section segmentation diagnostics to CSV for verification."""

import csv
from pathlib import Path

from src.models.paper import SectionType, Paper

PREVIEW_LEN = 3000  # chars of each section to include as preview


def write_section_diagnostics_csv(papers: list[Paper], csv_path: Path):
    """Write a diagnostics CSV showing section segmentation and summary quality.

    For each paper, shows:
    - Title and PDF filename
    - Full-text char count
    - Per-section char count (raw heuristic split)
    - Per-section summary preview (LLM-generated)
    - Per-section raw text preview

    Args:
        papers: List of papers with sections and summaries populated.
        csv_path: Output CSV path.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    section_names = [
        "title_abstract_conclusion",
        "introduction",
        "related_work",
        "method",
        "experiments",
    ]

    section_type_map = {
        "title_abstract_conclusion": SectionType.TITLE_ABSTRACT_CONCLUSION,
        "introduction": SectionType.INTRODUCTION,
        "related_work": SectionType.RELATED_WORK,
        "method": SectionType.METHOD,
        "experiments": SectionType.EXPERIMENTS,
    }

    fieldnames = ["title", "pdf_file", "full_text_chars"]
    for name in section_names:
        fieldnames.append(f"{name}_chars")
    fieldnames.append("total_section_chars")
    fieldnames.append("coverage_pct")
    for name in section_names:
        fieldnames.append(f"{name}_summary_chars")
    fieldnames.append("summaries_count")
    for name in section_names:
        fieldnames.append(f"{name}_summary_preview")
    for name in section_names:
        fieldnames.append(f"{name}_raw_preview")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for paper in papers:
            full_len = len(paper.full_text) if paper.full_text else 0

            sec_texts = {}
            sec_lens = {}
            total_sec = 0

            for name in section_names:
                text = ""
                if paper.sections:
                    text = getattr(paper.sections, name, "") or ""
                sec_texts[name] = text
                sec_lens[name] = len(text)
                total_sec += len(text)

            coverage = (total_sec / full_len * 100) if full_len > 0 else 0.0

            # Summary info
            summary_count = 0
            summary_lens = {}
            summary_texts = {}
            for name in section_names:
                st = section_type_map[name]
                summary = paper.section_summaries.get(st.value, "")
                summary_texts[name] = summary
                summary_lens[name] = len(summary)
                if summary:
                    summary_count += 1

            row = {
                "title": paper.title or "(no title)",
                "pdf_file": paper.pdf_path.name,
                "full_text_chars": full_len,
                "total_section_chars": total_sec,
                "coverage_pct": f"{coverage:.1f}",
                "summaries_count": summary_count,
            }

            for name in section_names:
                row[f"{name}_chars"] = sec_lens[name]
                row[f"{name}_summary_chars"] = summary_lens[name]
                # Summary preview
                summary_preview = summary_texts[name][:PREVIEW_LEN].replace("\n", " ").strip()
                row[f"{name}_summary_preview"] = summary_preview
                # Raw text preview
                raw_preview = sec_texts[name][:PREVIEW_LEN].replace("\n", " ").strip()
                row[f"{name}_raw_preview"] = raw_preview

            writer.writerow(row)
