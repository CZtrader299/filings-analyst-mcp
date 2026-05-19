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

Heading variants seen in real filings (week-3 hardening notes):

* Apple's 2025 10-K (accession ``0000320193-25-000079``) uses
  ``Item 7.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and
  Analysis of Financial Condition and Results of Operations``. After
  BeautifulSoup decodes entities, the apostrophe becomes the curly
  ``’`` rather than ``'``, and the inter-token space becomes
  ``\xa0`` (non-breaking space). The pre-week-3 regex anchored on a
  literal ASCII apostrophe followed by ``\\s+DISCUSSION``, so MD&A
  came back empty.
* Other filers use ALL CAPS (``ITEM 7.``), drop the trailing period,
  wrap the heading in ``<b>``, or omit the space between the item
  number and the title. We normalize whitespace + quotes on the
  parsed lines and use looser regex variants below.
"""

from __future__ import annotations

import re
import warnings
from typing import Iterable

from bs4 import BeautifulSoup

try:  # bs4 only exposes XMLParsedAsHTMLWarning on newer versions.
    from bs4 import XMLParsedAsHTMLWarning  # type: ignore
except ImportError:  # pragma: no cover - very old bs4
    XMLParsedAsHTMLWarning = None  # type: ignore


# The canonical section names we return as dict keys.
SECTION_NAMES = ("Business", "Risk Factors", "MD&A", "Financial Statements")


# Each entry: (canonical_name, regex pattern matched against a plaintext line).
#
# Patterns are matched against pre-normalized lines: curly quotes are folded
# to ASCII apostrophes and non-breaking spaces collapsed to regular spaces
# before regex evaluation (see ``_normalize_heading_line``). We anchor on
# Item N (or N + letter) followed by the title. Real filings use periods,
# dashes, tabs, or just runs of whitespace between the item number and
# title — the separator class is permissive on purpose. ``re.IGNORECASE``
# handles "ITEM 1A." vs "Item 1a".
#
# Compiled once at module import.
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "Business",
        re.compile(
            r"^\s*ITEM\s*1\b\.?\s*[-:.\s]*BUSINESS\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Risk Factors",
        re.compile(
            r"^\s*ITEM\s*1A\b\.?\s*[-:.\s]*RISK\s+FACTORS\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MD&A",
        # Apostrophe is optional and may be any of '/curly variants
        # (we normalize before matching so a literal `'?` is enough).
        # ``S`` may be absent (some filers write "Management Discussion").
        # The separator between "Management's" and "Discussion" may be a
        # space, non-breaking space (already normalized), or even nothing
        # if the filer renders the apostrophe-s as a single token. We
        # accept zero-or-more whitespace there.
        re.compile(
            r"^\s*ITEM\s*7\b\.?\s*[-:.\s]*MANAGEMENT'?S?\s*DISCUSSION",
            re.IGNORECASE,
        ),
    ),
    (
        "Financial Statements",
        re.compile(
            r"^\s*ITEM\s*8\b\.?\s*[-:.\s]*"
            r"(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS",
            re.IGNORECASE,
        ),
    ),
]

# Fallback patterns: used only when the primary Item-anchored patterns
# missed a section. They match the title alone (no Item-number anchor)
# so a stray heading like ``Management's Discussion and Analysis`` still
# gets picked up. Kept separate to keep the primary path strict.
_FALLBACK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "MD&A",
        re.compile(
            r"^\s*MANAGEMENT'?S?\s*DISCUSSION\s+AND\s+ANALYSIS",
            re.IGNORECASE,
        ),
    ),
    (
        "Risk Factors",
        re.compile(r"^\s*RISK\s+FACTORS\s*$", re.IGNORECASE),
    ),
]


# Curly quotes / typographic apostrophes we want to fold to a plain ASCII
# apostrophe before regex matching. Real SEC filings use these constantly
# in section headers ("Management's"); leaving them as-is silently breaks
# header detection for filers that emit them.
_CURLY_QUOTES = {
    "’": "'",  # right single quotation mark (most common)
    "‘": "'",  # left single quotation mark
    "ʼ": "'",  # modifier letter apostrophe
    "“": '"',
    "”": '"',
}


def _normalize_heading_line(line: str) -> str:
    """Fold curly quotes + non-breaking spaces so the heading regexes match.

    Operates on a single line. Cheap, side-effect-free.
    """
    for src, dst in _CURLY_QUOTES.items():
        if src in line:
            line = line.replace(src, dst)
    # Non-breaking and zero-width spaces -> plain space.
    line = line.replace("\xa0", " ").replace("​", "")
    return line


def clean_section_text(text: str) -> str:
    """Normalize whitespace and strip common 10-K HTML artifacts.

    * Collapse runs of whitespace (including non-breaking spaces).
    * Drop bare "(continued)" markers that appear in tables of contents.
    * Strip lines that are purely page numbers.
    """
    if not text:
        return ""
    # Normalize unicode non-breaking space and similar to plain space.
    text = text.replace("\xa0", " ").replace("​", "")
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
    the stdlib parser. ``XMLParsedAsHTMLWarning`` from BeautifulSoup is
    cosmetic noise for our use case (SEC filings are XHTML-ish HTML);
    suppress it scoped to this call rather than globally.
    """
    with warnings.catch_warnings():
        if XMLParsedAsHTMLWarning is not None:
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
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
    lines = [line.replace("\xa0", " ").strip() for line in raw.splitlines()]
    return lines


def _find_section_starts(lines: list[str]) -> list[tuple[int, str]]:
    """Return [(line_index, canonical_name), ...] sorted by line_index.

    A given canonical name can match multiple times — that's expected
    because 10-Ks repeat Item headings in their table of contents. We
    keep the LATEST match (i.e., the body, not the TOC) by picking the
    last occurrence per section in ``extract_sections``.

    Primary Item-anchored patterns run first; title-only fallback
    patterns run as a second pass only for sections still missing,
    which avoids leaking ToC references into the primary signal.
    """
    hits: list[tuple[int, str]] = []
    matched_names: set[str] = set()
    # Microsoft's 2025 10-K (and similar filers) splits Item headings
    # across two lines, e.g. ``ITEM 1. B`` then ``USINESS`` on the next
    # line. To catch these we evaluate a small sliding window: each
    # source line plus the joined ``line + next_line`` if the current
    # line starts with an Item-style prefix. We only run the joined
    # check when the raw line itself doesn't match, so well-formed
    # filings (Apple, our synthetic fixture) keep matching as before.
    _ITEM_PREFIX = re.compile(r"^\s*ITEM\s*\d", re.IGNORECASE)
    for idx, line in enumerate(lines):
        if not line:
            continue
        normalized = _normalize_heading_line(line)
        candidates = [normalized]
        if _ITEM_PREFIX.match(normalized):
            # Greedy join with up to two following non-empty lines.
            tail_bits: list[str] = []
            j = idx + 1
            looked_at = 0
            while j < len(lines) and looked_at < 4:
                nxt = lines[j]
                looked_at += 1
                j += 1
                if not nxt:
                    continue
                tail_bits.append(_normalize_heading_line(nxt))
                if len(tail_bits) >= 2:
                    break
            for combo_count in range(1, len(tail_bits) + 1):
                joined = (normalized + "".join(tail_bits[:combo_count])).strip()
                # Re-collapse spaces so the regex doesn't choke on
                # unexpected gaps from the per-line strip.
                joined = re.sub(r"\s+", " ", joined)
                candidates.append(joined)
        matched = False
        for cand in candidates:
            for name, pattern in _SECTION_PATTERNS:
                if pattern.search(cand):
                    hits.append((idx, name))
                    matched_names.add(name)
                    matched = True
                    break
            if matched:
                break

    # Second pass: title-only fallbacks for sections not yet seen.
    missing = [
        (name, pat) for name, pat in _FALLBACK_PATTERNS if name not in matched_names
    ]
    if missing:
        for idx, line in enumerate(lines):
            if not line:
                continue
            normalized = _normalize_heading_line(line)
            for name, pattern in missing:
                if pattern.search(normalized):
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
    2. Find every line that matches an Item-header pattern (with curly
       quotes and non-breaking spaces normalized first).
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
