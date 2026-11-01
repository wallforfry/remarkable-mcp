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


class TestTypedTextPerPage:
    def _zip(self, tmp_path):
        import json
        import zipfile

        path = tmp_path / "doc.zip"
        with zipfile.ZipFile(path, "w") as zf:
            pages = [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}]
            zf.writestr("doc.content", json.dumps({"cPages": {"pages": pages}}))
            zf.writestr("doc/p3.rm", nb.page_rm_bytes("# Third\n- item"))
            zf.writestr("doc/p1.rm", nb.page_rm_bytes("First"))
            zf.writestr("doc/p2.rm", nb.page_rm_bytes(""))
        return path

    def test_extraction_keeps_page_numbers(self, tmp_path):
        from remarkable_mcp.extract import extract_text_from_document_zip

        result = extract_text_from_document_zip(self._zip(tmp_path))

        assert result["typed_pages"] == [
            {"page": 1, "page_id": "p1", "lines": ["First"]},
            {"page": 3, "page_id": "p3", "lines": ["# Third", "- item"]},
        ]
        assert result["typed_text"] == ["First", "# Third", "- item"]

    def test_read_groups_typed_text_under_page_headers(self):
        from remarkable_mcp.tools import _typed_text_parts

        content = {
            "typed_pages": [
                {"page": 2, "page_id": "p2", "lines": ["# Plan", "", "- Ship"]},
                {"page": 5, "page_id": "p5", "lines": ["Later"]},
            ],
            "typed_text": ["# Plan", "", "- Ship", "Later", "from a .txt file"],
        }

        assert _typed_text_parts(content) == [
            "--- Page 2 ---\n# Plan\n\n- Ship",
            "--- Page 5 ---\nLater",
            "from a .txt file",
        ]

    def test_read_without_page_split_keeps_flat_text(self):
        from remarkable_mcp.tools import _typed_text_parts

        assert _typed_text_parts({"typed_text": ["a", "b"]}) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_remarkable_read_shows_page_headers(self, tmp_path):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from remarkable_mcp import tools
        from remarkable_mcp.extract import extract_text_from_document_zip
        from remarkable_mcp.server import mcp

        doc = SimpleNamespace(
            VissibleName="Notes",
            ID="doc",
            Parent="",
            ModifiedClient=None,
            is_folder=False,
            tags=[],
        )
        client = Mock()
        client.get_meta_items.return_value = [doc]
        client.download.return_value = self._zip(tmp_path).read_bytes()

        with (
            patch.object(tools, "get_rmapi", return_value=client),
            patch.object(tools, "get_file_type", return_value="notebook"),
            patch.object(tools, "get_cached_ocr_result", return_value=None),
            patch.object(
                tools,
                "extract_text_from_document_zip",
                lambda path, **_: extract_text_from_document_zip(path),
            ),
        ):
            result = await mcp.call_tool("remarkable_read", {"document": "Notes"})

        content = json.loads(result.content[0].text)["content"]
        assert content == "--- Page 1 ---\nFirst\n\n--- Page 3 ---\n# Third\n- item"
