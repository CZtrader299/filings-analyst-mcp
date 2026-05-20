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

Heading variants seen in real filings:

* Apple's 2025 10-K (accession ``0000320193-25-000079``) uses
  ``Item 7.&#160;&#160;&#160;&#160;Management&#8217;s Discussion and
  Analysis of Financial Condition and Results of Operations``. After
  BeautifulSoup decodes entities, the apostrophe becomes the curly
  ``’`` rather than ``'``, and the inter-token space becomes
  ``\xa0`` (non-breaking space). An earlier version of this regex
  anchored on a literal ASCII apostrophe followed by ``\\s+DISCUSSION``,
  so MD&A came back empty against this filing.
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


# Minimum reasonable MD&A length in cleaned characters. Real MD&A
# sections in 10-Ks are tens of thousands of characters; anything below
# this threshold is almost certainly the Item-anchor catching a
# forward-reference placeholder, a chapter-divider banner, or a TOC
# entry rather than the real section. Picked to leave Apple/Microsoft
# (both well above 20K) untouched while triggering on JPM/BAC/XOM
# (all under 1K with the Item-anchor strategy alone).
_MDA_MIN_REASONABLE_CHARS = 3000

# Title-only MD&A pattern used by the section-title fallback. Matches
# both ALL-CAPS and Title-Case variants, with or without an "ITEM 7."
# prefix, and tolerates the curly-apostrophe and non-breaking-space
# variants real filers emit (those get normalized to ASCII before
# matching, so a plain "'" is enough here). Anchored to the start of
# a line so we only match heading-like text, not in-prose mentions
# like "see the Management's Discussion and Analysis section".
_MDA_TITLE_FALLBACK = re.compile(
    r"^\s*(?:ITEM\s*7\.?\s*)?MANAGEMENT'?S?\s+DISCUSSION\s+AND\s+ANALYSIS\b",
    re.IGNORECASE,
)

# Markers for the END of an MD&A section in the fallback path. Item 7A
# (quantitative/qualitative disclosures) or Item 8 (financial
# statements) typically follow MD&A. The standalone title variant
# catches filers that drop the "Item 7A." label.
_MDA_END_MARKERS = [
    re.compile(r"^\s*ITEM\s*7A\b", re.IGNORECASE),
    re.compile(r"^\s*ITEM\s*8\b", re.IGNORECASE),
    re.compile(
        r"^\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\b",
        re.IGNORECASE,
    ),
    # JPM-style: the financial-statements block opens with the
    # auditor's report rather than an Item 8 anchor (Item 8 only
    # appears earlier as a forward-reference). These are reliable
    # boundary markers for the start of the financial-statements
    # block in bank filings.
    re.compile(
        r"^\s*REPORT\s+OF\s+INDEPENDENT\s+REGISTERED\s+PUBLIC\s+ACCOUNTING\s+FIRM\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*MANAGEMENT'?S?\s+REPORT\s+ON\s+INTERNAL\s+CONTROL\b",
        re.IGNORECASE,
    ),
]


def _mda_density_score(lines: list[str], start: int, window: int = 30) -> int:
    """Count alphabetic characters in the next ``window`` non-empty lines.

    Used by the section-title fallback to distinguish a real MD&A body
    start (followed by paragraphs of prose) from a TOC entry, a
    chapter-divider banner, or a page-header repeat (followed by
    page numbers, navigation links, or another heading).
    """
    collected: list[str] = []
    i = start + 1
    while i < len(lines) and len(collected) < window:
        if lines[i]:
            collected.append(_normalize_heading_line(lines[i]))
        i += 1
    joined = " ".join(collected)
    return sum(1 for c in joined if c.isalpha())


def _find_mda_fallback_start(
    lines: list[str],
    current_start: int,
) -> list[int] | None:
    """Find candidate MD&A starts by searching for the section title directly.

    Used only when the Item-anchor extraction produced a suspiciously
    short MD&A slice. Three real failure modes motivate this:

    * **JPM**: ``Item 7.`` is a forward-reference placeholder ("MD&A
      appears on pages 46-160. Such information should be read in
      conjunction with...") immediately followed by ``Item 7A.``. The
      actual MD&A content sits under a separately-titled
      ``Management's discussion and analysis`` heading elsewhere.
    * **BAC**: the ``Item 7.`` line is a chapter-divider banner near
      the end of the document; the real content is earlier under a
      ``Management's Discussion and Analysis...`` title.
    * **XOM**: the ``Item 7.`` line *is* the real one (a forward
      reference to the Financial Section), but the actual MD&A body
      that follows is interspersed with repeated page-header lines
      ("MANAGEMENT'S DISCUSSION AND ANALYSIS..."). The Item-anchor
      slice ended too early because Item 7A also appears in the cover
      forward-reference. Picking the title fallback with high prose
      density downstream lands us in the real body.

    Heuristic: scan every line matching :data:`_MDA_TITLE_FALLBACK`,
    skip lines that are clearly Item-anchor variants of the *current*
    too-short slice, and return every candidate whose next 30 non-
    empty lines contain enough alphabetic prose to be a real section
    body (>= 1500 alpha chars is a generous floor — TOC and page-
    header entries score well below this in practice). The caller
    walks the list in order and accepts the first candidate that
    produces a long-enough body; this two-step accept lets us skip
    short forward-reference paragraphs (JPM) that look like real
    section bodies up until the trailing Item 7A end-marker fires.

    Returns the candidate line indices in document order, or ``None``
    if no line clears the prose-density floor.
    """
    PROSE_FLOOR = 1500
    candidates: list[int] = []
    for idx, raw_line in enumerate(lines):
        if not raw_line:
            continue
        line = _normalize_heading_line(raw_line)
        if not _MDA_TITLE_FALLBACK.search(line):
            continue
        # Don't re-pick the same Item-anchor line that already failed.
        if idx == current_start:
            continue
        score = _mda_density_score(lines, idx)
        if score < PROSE_FLOOR:
            continue
        candidates.append(idx)
    return candidates if candidates else None  # type: ignore[return-value]


def _find_mda_fallback_end(lines: list[str], start: int) -> int:
    """Locate the end of the MD&A section under the fallback path.

    Returns the index of the first end-marker line at or after
    ``start + 1``, or ``len(lines)`` if no marker is found. End markers
    are Item 7A, Item 8, or a standalone "Quantitative and Qualitative
    Disclosures" heading.
    """
    for idx in range(start + 1, len(lines)):
        raw = lines[idx]
        if not raw:
            continue
        line = _normalize_heading_line(raw)
        for pat in _MDA_END_MARKERS:
            if pat.search(line):
                return idx
    return len(lines)


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
    5. If MD&A came back implausibly short (< ``_MDA_MIN_REASONABLE_CHARS``)
       run a section-title fallback that searches for the
       ``Management's Discussion and Analysis`` heading directly and
       uses a prose-density signal to skip TOC entries and
       chapter-divider banners. See :func:`_find_mda_fallback_start`
       for the three real failure modes this addresses (JPM forward-
       reference, BAC chapter-divider banner, XOM TOC-anchor repeats).
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

    # --- MD&A section-title fallback ------------------------------------
    # The primary Item-anchor heuristic above fails on three real
    # patterns observed in our corpus (JPM/BAC/XOM). When the MD&A
    # slice is implausibly short we re-run extraction for THIS section
    # only using the section-title fallback. AAPL and MSFT (both well
    # above the threshold) take this branch never; their extraction
    # is unchanged.
    mda_start = last_hit.get("MD&A")
    if len(result["MD&A"]) < _MDA_MIN_REASONABLE_CHARS:
        fallback_candidates = _find_mda_fallback_start(
            lines,
            current_start=mda_start if mda_start is not None else -1,
        )
        if fallback_candidates:
            # Walk candidates in document order; accept the first one
            # whose slice actually clears the reasonable-length floor.
            # This filters out JPM's forward-reference paragraph
            # (which looks dense in the next 30 lines but is followed
            # immediately by an Item 7A end-marker that truncates the
            # slice to a few hundred characters).
            for fallback_start in fallback_candidates:
                fallback_end = _find_mda_fallback_end(lines, fallback_start)
                body_lines = lines[fallback_start + 1 : fallback_end]
                body = "\n".join(line for line in body_lines if line is not None)
                fallback_text = clean_section_text(body)
                if len(fallback_text) >= _MDA_MIN_REASONABLE_CHARS:
                    result["MD&A"] = fallback_text
                    break
            else:
                # No candidate cleared the floor. Take the longest
                # candidate slice we saw — better than the empty/tiny
                # Item-anchor result — but never regress.
                best_text = result["MD&A"]
                for fallback_start in fallback_candidates:
                    fallback_end = _find_mda_fallback_end(lines, fallback_start)
                    body_lines = lines[fallback_start + 1 : fallback_end]
                    body = "\n".join(line for line in body_lines if line is not None)
                    candidate = clean_section_text(body)
                    if len(candidate) > len(best_text):
                        best_text = candidate
                result["MD&A"] = best_text

    return result
