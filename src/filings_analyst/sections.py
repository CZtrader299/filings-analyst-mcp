"""10-K section extractor.

10-K formatting varies wildly between filers — some use ``<h2>`` tags, some
use ``<b>`` inside ``<p>``, some inline everything in a single deep
``<div>`` with formatting via CSS classes. This module takes a deliberately
defensive approach: parse to plain text, then locate sections by matching
Item-number + title patterns. Sections that aren't found return empty
strings rather than raising — callers can decide how to surface that.

Targets:

* ``Business`` — Item 1
* ``Risk Factors`` — Item 1A
* ``MD&A`` — Item 7 (Management's Discussion and Analysis)
* ``Financial Statements`` — Item 8 (header marker only, not parsed contents)
"""

from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup


# The canonical section names we return as dict keys.
SECTION_NAMES = ("Business", "Risk Factors", "MD&A", "Financial Statements")


# Each entry: (canonical_name, regex pattern matched against a plaintext line).
# We anchor on Item N (or N + letter) followed by the title. Lots of filers
# use periods, dots, or just a space between the item number and title, so
# the separator is loose. ``re.IGNORECASE`` handles "ITEM 1A." vs "Item 1a".
#
# We compile these once at module import.
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "Business",
        re.compile(r"^\s*ITEM\s*1\b\.?\s*[-:.\s]*BUSINESS\b", re.IGNORECASE),
    ),
    (
        "Risk Factors",
        re.compile(r"^\s*ITEM\s*1A\b\.?\s*[-:.\s]*RISK\s+FACTORS\b", re.IGNORECASE),
    ),
    (
        "MD&A",
        re.compile(
            r"^\s*ITEM\s*7\b\.?\s*[-:.\s]*MANAGEMENT'?S?\s+DISCUSSION",
            re.IGNORECASE,
        ),
    ),
    (
        "Financial Statements",
        re.compile(
            r"^\s*ITEM\s*8\b\.?\s*[-:.\s]*(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS",
            re.IGNORECASE,
        ),
    ),
]


def clean_section_text(text: str) -> str:
    """Normalize whitespace and strip common 10-K HTML artifacts.

    * Collapse runs of whitespace (including non-breaking spaces).
    * Drop bare "(continued)" markers that appear in tables of contents.
    * Strip lines that are purely page numbers.
    """
    if not text:
        return ""
    # Normalize unicode non-breaking space and similar to plain space.
    text = text.replace(" ", " ").replace("​", "")
    # Drop "(continued)" markers, case-insensitive.
    text = re.sub(r"\(\s*continued\s*\)", " ", text, flags=re.IGNORECASE)
    # Strip lines that are just a page number.
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if re.fullmatch(r"-?\s*\d{1,4}\s*-?", stripped):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _html_to_lines(html_text: str) -> list[str]:
    """Parse HTML to a flat list of lines of plain text.

    Uses lxml when available (faster, more forgiving) and falls back to
    the stdlib parser.
    """
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        soup = BeautifulSoup(html_text, "html.parser")
    # Drop script/style noise.
    for tag in soup(["script", "style"]):
        tag.decompose()
    # ``get_text`` with a newline separator gives us roughly line-per-block
    # output, which is what the Item-header regex below expects.
    raw = soup.get_text("\n")
    # Pre-clean to make line matching cheaper.
    lines = [line.replace(" ", " ").strip() for line in raw.splitlines()]
    return lines


def _find_section_starts(lines: Iterable[str]) -> list[tuple[int, str]]:
    """Return [(line_index, canonical_name), ...] sorted by line_index.

    A given canonical name can match multiple times — that's expected
    because 10-Ks repeat Item headings in their table of contents. We
    keep the LATEST match (i.e., the body, not the TOC) by picking the
    last occurrence per section in ``extract_sections``.
    """
    hits: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if not line:
            continue
        for name, pattern in _SECTION_PATTERNS:
            if pattern.search(line):
                hits.append((idx, name))
                break
    return hits


def extract_sections(html_text: str) -> dict[str, str]:
    """Extract known 10-K sections from raw filing HTML.

    Returns a dict mapping each canonical section name in
    :data:`SECTION_NAMES` to its cleaned plain text. Sections that
    couldn't be located return an empty string rather than raising; the
    caller can decide whether to treat that as an error.

    Heuristic:

    1. Flatten the HTML to plain-text lines.
    2. Find every line that matches an Item-header pattern.
    3. For each canonical section, take the LAST matching line as the
       real body start (TOC entries come earlier in the document).
    4. Slice the lines list from that start to the next section's start.
    """
    if not html_text:
        return {name: "" for name in SECTION_NAMES}

    lines = _html_to_lines(html_text)
    hits = _find_section_starts(lines)

    # Map each canonical name to its latest match (skip TOC).
    last_hit: dict[str, int] = {}
    for idx, name in hits:
        last_hit[name] = idx

    # Build an ordered list of (start_idx, name) and walk to next start.
    starts = sorted([(idx, name) for name, idx in last_hit.items()])
    starts_with_sentinel = starts + [(len(lines), "__END__")]

    result: dict[str, str] = {name: "" for name in SECTION_NAMES}
    for i in range(len(starts)):
        start_idx, name = starts_with_sentinel[i]
        end_idx, _ = starts_with_sentinel[i + 1]
        # Skip the matched header line itself.
        body_lines = lines[start_idx + 1 : end_idx]
        body = "\n".join(line for line in body_lines if line is not None)
        result[name] = clean_section_text(body)
    return result
