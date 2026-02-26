"""Heuristic regex-based section splitting for academic papers.

Identifies section headings via pattern matching and splits the full text
into 5 canonical sections. No LLM needed — instant and deterministic.
"""

import logging
import re
from typing import Optional

from src.models.paper import PaperSections

logger = logging.getLogger(__name__)

# Patterns for each section category.
# Order matters within each group — more specific patterns first.
# All patterns are case-insensitive and match at line start (after optional numbering).
_HEADING_PREFIX = r"^(?:\d+\.?\s*|[IVX]+\.?\s*)?(?:[-–—]\s*)?"

_SECTION_PATTERNS: dict[str, list[str]] = {
    "title_abstract_conclusion": [
        r"(?:abstract)\b",
        r"(?:conclusion)s?\b",
        r"(?:summary)\b",
        r"(?:concluding\s+remarks)\b",
        r"(?:final\s+remarks)\b",
    ],
    "introduction": [
        r"(?:introduction)\b",
        r"(?:motivation)\b",
        r"(?:overview)\b",
    ],
    "related_work": [
        r"(?:related\s+work)\b",
        r"(?:related\s+works)\b",
        r"(?:background)\b",
        r"(?:prior\s+work)\b",
        r"(?:literature\s+review)\b",
        r"(?:previous\s+work)\b",
        r"(?:preliminaries)\b",
    ],
    "method": [
        r"(?:method)(?:ology|s)?\b",
        r"(?:approach)\b",
        r"(?:proposed\s+method)\b",
        r"(?:proposed\s+approach)\b",
        r"(?:framework)\b",
        r"(?:model)s?\b",
        r"(?:architecture)\b",
        r"(?:our\s+approach)\b",
        r"(?:our\s+method)\b",
        r"(?:formulation)\b",
        r"(?:problem\s+(?:definition|formulation|setup|setting))\b",
        r"(?:technical\s+approach)\b",
    ],
    "experiments": [
        r"(?:experiment)s?\b",
        r"(?:experimental\s+(?:results|evaluation|setup|settings))\b",
        r"(?:(?:numerical|computational)\s+experiment)s?\b",
        r"(?:results)\b",
        r"(?:evaluation)\b",
        r"(?:ablation)\b",
        r"(?:empirical\s+(?:results|evaluation|study))\b",
        r"(?:performance\s+(?:on\s+)?(?:benchmarks?|evaluation|comparison))\b",
        r"(?:benchmarks?)\b",
        r"(?:analysis)\b",
    ],
}

# Compile patterns
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {}
for section, patterns in _SECTION_PATTERNS.items():
    _COMPILED_PATTERNS[section] = [
        re.compile(_HEADING_PREFIX + p, re.IGNORECASE | re.MULTILINE)
        for p in patterns
    ]

# Patterns for lines that are likely headings (short, possibly numbered)
_HEADING_LINE_RE = re.compile(
    r"^(?:\d+\.?\s+|[IVX]+\.?\s+)?[A-Z][A-Za-z\s:,\-–—&]+$"
)


def _is_heading_line(line: str) -> bool:
    """Check if a line looks like a section heading (short, title-cased/uppercase)."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120 or len(stripped) < 3:
        return False
    # Headings are typically short
    if len(stripped.split()) > 12:
        return False
    return bool(_HEADING_LINE_RE.match(stripped))


def split_sections(full_text: str) -> PaperSections:
    """Split paper full text into 5 sections using regex heading detection.

    Every line in the text is assigned to exactly one section.
    Lines before the first detected heading go to title_abstract_conclusion.

    Args:
        full_text: Complete paper text from PDF extraction.

    Returns:
        PaperSections with text distributed across sections.
    """
    if not full_text or len(full_text.strip()) < 100:
        return PaperSections(title_abstract_conclusion=full_text or "")

    lines = full_text.split("\n")
    n_lines = len(lines)

    # Find all heading positions: (line_index, section_name)
    boundaries: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not _is_heading_line(stripped):
            continue

        for section_name, patterns in _COMPILED_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(stripped):
                    boundaries.append((i, section_name))
                    break
            else:
                continue
            break  # Found a match for this line, stop checking sections

    if not boundaries:
        logger.warning("No section headings detected, putting all text in title_abstract_conclusion")
        return PaperSections(title_abstract_conclusion=full_text)

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    # Assign lines to sections (contiguous chunks)
    section_names = [
        "title_abstract_conclusion", "introduction", "related_work",
        "method", "experiments",
    ]
    section_lines: dict[str, list[str]] = {name: [] for name in section_names}

    # Lines before first boundary → title_abstract_conclusion
    current_section = "title_abstract_conclusion"
    boundary_idx = 0

    for i in range(n_lines):
        # Check if we've reached the next boundary
        while (boundary_idx < len(boundaries)
               and boundaries[boundary_idx][0] <= i):
            current_section = boundaries[boundary_idx][1]
            boundary_idx += 1

        section_lines[current_section].append(lines[i])

    result = PaperSections(
        title_abstract_conclusion="\n".join(section_lines["title_abstract_conclusion"]),
        introduction="\n".join(section_lines["introduction"]),
        related_work="\n".join(section_lines["related_work"]),
        method="\n".join(section_lines["method"]),
        experiments="\n".join(section_lines["experiments"]),
    )

    # Log distribution
    total = len(full_text)
    for name in section_names:
        text = getattr(result, name) or ""
        pct = len(text) / total * 100 if total > 0 else 0
        logger.debug("  %s: %d chars (%.1f%%)", name, len(text), pct)

    non_empty = sum(1 for name in section_names if getattr(result, name))
    logger.info("Heuristic split: %d/%d sections populated, %d boundaries found",
                non_empty, len(section_names), len(boundaries))

    return result
