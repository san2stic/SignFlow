"""Markdown helpers supporting Obsidian-style wiki links."""

from __future__ import annotations

import re

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def extract_wikilinks(markdown_text: str | None) -> list[str]:
    """Extract unique wikilink targets from markdown text."""
    if not markdown_text:
        return []
    targets = [match.strip() for match in WIKILINK_PATTERN.findall(markdown_text)]
    deduped: list[str] = []
    seen = set()
    for target in targets:
        key = target.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)
    return deduped
