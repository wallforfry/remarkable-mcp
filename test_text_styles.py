"""Tests for reading paragraph styles back as Markdown line prefixes."""

from __future__ import annotations

import io

import pytest
from rmscene import read_blocks, write_blocks
from rmscene.scene_items import Text

from remarkable_mcp import notebooks as nb
from remarkable_mcp.extract import extract_text_from_rm_file
from remarkable_mcp.strokes import WRITE_VERSION
from remarkable_mcp.text_styles import (
    TextStyle,
    _style_for,
    markdown_lines,
    read_paragraph_styles,
)

_EVERY_STYLE = (
    "# Title\n## Heading\n### Subheading\nBody\n- Bullet\n  - Nested\n"
    "1. One\n2. Two\n- [ ] Todo\n- [x] Done"
)


def _text_item(page: bytes) -> Text:
    return next(
        b.value
        for b in read_blocks(io.BytesIO(page))
        if isinstance(getattr(b, "value", None), Text)
    )


def _read_back(page: bytes) -> str:
    return "\n".join(markdown_lines(_text_item(page), read_paragraph_styles(page)))


class TestReadParagraphStyles:
    def test_every_style_round_trips(self):
        assert _read_back(nb.page_rm_bytes(_EVERY_STYLE)) == _EVERY_STYLE

    def test_heading_and_subheading_are_told_apart(self):
        styles = read_paragraph_styles(nb.page_rm_bytes("## Big\n### Small"))

        assert sorted(s.name for s in styles.values()) == ["HEADING", "SUBHEADING"]

    def test_numbered_items_are_renumbered_per_run(self):
        page = nb.page_rm_bytes("7. a\n9. b\nbreak\n4. c")

        assert _read_back(page) == "1. a\n2. b\nbreak\n1. c"

    def test_page_without_text_has_no_styles(self):
        blocks = [
            b
            for b in read_blocks(io.BytesIO(nb.page_rm_bytes("")))
            if not isinstance(getattr(b, "value", None), Text)
        ]
        buf = io.BytesIO()
        write_blocks(buf, blocks, options={"version": WRITE_VERSION})

        assert read_paragraph_styles(buf.getvalue()) == {}

    @pytest.mark.parametrize(
        "code, extra, expected",
        [
            (42, (), TextStyle.BODY),
            (3, ((2, "byte", 9),), TextStyle.SUBHEADING),
            (3, ((2, "byte", 2), (3, "int", 3)), TextStyle.HEADING),
        ],
        ids=["unknown-code", "unknown-extra-fields", "heading"],
    )
    def test_codes_map_to_styles(self, code, extra, expected):
        assert _style_for(code, extra) is expected


class TestExtractTypedText:
    def test_lines_carry_their_prefixes(self, tmp_path):
        path = tmp_path / "page.rm"
        path.write_bytes(nb.page_rm_bytes("# Plan\n- Ship\nNotes"))

        assert extract_text_from_rm_file(path) == ["# Plan", "- Ship", "Notes"]

    @pytest.mark.parametrize(
        "text, expected",
        [("a\n\nb", ["a", "", "b"]), ("a\nb\n\n", ["a", "b"]), ("", [])],
        ids=["inner-blank-kept", "trailing-blank-dropped", "empty-page"],
    )
    def test_blank_paragraphs(self, tmp_path, text, expected):
        path = tmp_path / "page.rm"
        path.write_bytes(nb.page_rm_bytes(text))

        assert extract_text_from_rm_file(path) == expected
