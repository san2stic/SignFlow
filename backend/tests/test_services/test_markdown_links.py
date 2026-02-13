"""Markdown wikilink parser test."""

from __future__ import annotations

from app.utils.markdown import extract_wikilinks


def test_extract_wikilinks() -> None:
    """Parser should collect unique wiki link targets."""
    text = "Related: [[Bonjour]] and [[Salut]] and [[bonjour]]"
    assert extract_wikilinks(text) == ["Bonjour", "Salut"]
