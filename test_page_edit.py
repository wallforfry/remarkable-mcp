"""Tests for remarkable_mcp.page_edit: erasing ink and replacing typed text."""

from __future__ import annotations

import io
import struct

import pytest
from rmscene import HEADER_V6, read_blocks, read_tree
from rmscene.scene_stream import RootTextBlock, SceneLineItemBlock, SceneTombstoneItemBlock
from rmscene.tagged_block_common import CrdtId

from remarkable_mcp import notebooks as nb
from remarkable_mcp import page_edit
from remarkable_mcp import strokes as strokes_mod
from remarkable_mcp.extract import extract_text_from_rm_file, render_rm_file_to_svg
from remarkable_mcp.text_styles import TextStyle

_TOP_LEFT = {"points": [[0.1, 0.1], [0.3, 0.2]], "tool": "fineliner"}
_BOTTOM_RIGHT = {"points": [[0.7, 0.8], [0.9, 0.9]], "tool": "highlighter", "color": "yellow"}
# A block type rmscene does not know, standing in for newer-firmware data it
# cannot round-trip.
_UNKNOWN_BLOCK = struct.pack("<IBBBB", 3, 0, 1, 1, 0xEE) + b"\x01\x02\x03"


def _page(text: str = "Agenda\nFollow-ups", strokes=(_TOP_LEFT, _BOTTOM_RIGHT)) -> bytes:
    page = nb.page_rm_bytes(text)
    return strokes_mod.append_strokes(page, list(strokes)) if strokes else page


def _blocks(page: bytes) -> list:
    return list(read_blocks(io.BytesIO(page)))


def _live_lines(page: bytes) -> list:
    return [
        b.item.value
        for b in _blocks(page)
        if isinstance(b, SceneLineItemBlock) and b.item.value is not None
    ]


def _text(page: bytes, tmp_path) -> list[str]:
    path = tmp_path / "page.rm"
    path.write_bytes(page)
    return extract_text_from_rm_file(path)


def _root_text(page: bytes):
    return next(b for b in _blocks(page) if isinstance(b, RootTextBlock)).value


def _without_root_text(page: bytes) -> bytes:
    segments = page_edit._split_blocks(page)
    kept = [s for s in segments if not isinstance(page_edit._parse_block(s), RootTextBlock)]
    return HEADER_V6 + b"".join(kept)


class TestEraseStrokes:
    def test_erases_every_stroke_as_tombstones(self, tmp_path):
        page = _page()

        erased_page, erased = page_edit.erase_strokes(page)

        assert erased == 2
        assert _live_lines(erased_page) == []
        line_ids = [b.item.item_id for b in _blocks(page) if isinstance(b, SceneLineItemBlock)]
        tombstones = [b for b in _blocks(erased_page) if isinstance(b, SceneTombstoneItemBlock)]
        assert [t.item.item_id for t in tombstones] == line_ids
        assert all(t.item.deleted_length == 1 for t in tombstones)
        assert _text(erased_page, tmp_path) == _text(page, tmp_path)

    def test_region_erases_only_crossing_strokes(self):
        erased_page, erased = page_edit.erase_strokes(_page(), [0.0, 0.0, 0.5, 0.5])

        assert erased == 1
        [line] = _live_lines(erased_page)
        assert line.tool.name == "HIGHLIGHTER_2"

    def test_no_match_returns_the_page_unchanged(self):
        page = _page()

        assert page_edit.erase_strokes(page, [0.4, 0.4, 0.6, 0.6]) == (page, 0)
        ink_free = _page(strokes=())
        assert page_edit.erase_strokes(ink_free, None) == (ink_free, 0)

    def test_untouched_blocks_stay_byte_identical(self):
        page = _page() + _UNKNOWN_BLOCK

        erased_page, _ = page_edit.erase_strokes(page)

        before = page_edit._split_blocks(page)
        after = page_edit._split_blocks(erased_page)
        assert len(before) == len(after)
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        kinds = [type(page_edit._parse_block(before[i])) for i in changed]
        assert kinds == [SceneLineItemBlock, SceneLineItemBlock]
        assert erased_page.endswith(_UNKNOWN_BLOCK)

    def test_erased_page_still_renders(self, tmp_path):
        erased_page, _ = page_edit.erase_strokes(_page())
        path = tmp_path / "page.rm"
        path.write_bytes(erased_page)

        assert render_rm_file_to_svg(path) is not None

    @pytest.mark.parametrize(
        "region",
        [[0.1, 0.1, 0.2], [0.5, 0.1, 0.2, 0.3], [0, 0, 1, 1.5], ["a", 0, 1, 1]],
        ids=["three-values", "inverted", "out-of-range", "not-numbers"],
    )
    def test_invalid_region_is_rejected(self, region):
        with pytest.raises(page_edit.PageEditError, match="region"):
            page_edit.erase_strokes(_page(), region)


class TestSetPageText:
    def test_replaces_text_and_keeps_ink(self, tmp_path):
        page = _page()

        edited = page_edit.set_page_text(page, "# Plan\n- Ship\nDone")

        assert _text(edited, tmp_path) == ["Plan", "Ship", "Done"]
        assert len(_live_lines(edited)) == 2

    def test_old_characters_stay_as_deleted_runs(self):
        page = _page()
        old = _root_text(page).items.sequence_items()

        edited = page_edit.set_page_text(page, "New")
        items = _root_text(edited).items.sequence_items()

        deleted, [inserted] = items[: len(old)], items[len(old) :]
        assert [d.item_id for d in deleted] == [o.item_id for o in old]
        assert all(d.value == "" and d.deleted_length == len(o.value) for d, o in zip(deleted, old))
        assert inserted.value == "New"
        stroke_ids = [b.item.item_id for b in _blocks(page) if isinstance(b, SceneLineItemBlock)]
        assert inserted.item_id.part2 > max(i.part2 for i in stroke_ids)

    def test_styles_follow_the_line_prefixes(self):
        edited = page_edit.set_page_text(_page(), "# Plan\nbody\n- [ ] todo")
        text = _root_text(edited)
        first = text.items.sequence_items()[-1].item_id

        # rmscene reads the codes back into its own enum.
        codes = {key: value.value.value for key, value in text.styles.items()}
        assert codes == {
            CrdtId(0, 0): TextStyle.TITLE.code,
            CrdtId(first.part1, first.part2 + len("Plan\nbody")): TextStyle.CHECKBOX.code,
        }

    def test_style_stamps_win_over_existing_entries(self):
        page = _page()
        old_stamp = max(v.timestamp.part2 for v in _root_text(page).styles.values())

        edited = page_edit.set_page_text(page, "x")

        assert all(v.timestamp.part2 > old_stamp for v in _root_text(edited).styles.values())

    def test_empty_text_removes_the_typed_text(self, tmp_path):
        edited = page_edit.set_page_text(_page(), "")

        assert _text(edited, tmp_path) in ([], [""])
        assert all(i.deleted_length for i in _root_text(edited).items.sequence_items())

    def test_blank_page_keeps_its_empty_item_live(self, tmp_path):
        page = _page(text="", strokes=())

        edited = page_edit.set_page_text(page, "Hi")

        items = _root_text(edited).items.sequence_items()
        assert [(i.value, i.deleted_length) for i in items] == [("", 0), ("Hi", 0)]
        assert items[-1].left_id == CrdtId(0, 0)
        assert _text(edited, tmp_path) == ["Hi"]

    def test_repeated_edits_stay_readable(self, tmp_path):
        page = _page()
        for text in ("one", "", "two\nthree"):
            page = page_edit.set_page_text(page, text)

        assert _text(page, tmp_path) == ["two", "three"]
        read_tree(io.BytesIO(page))

    def test_adds_text_to_an_ink_only_page(self, tmp_path):
        ink_only = _without_root_text(_page())

        edited = page_edit.set_page_text(ink_only, "Hello")

        assert _text(edited, tmp_path) == ["Hello"]
        assert len(_live_lines(edited)) == 2

    def test_empty_text_on_an_ink_only_page_is_a_no_op(self):
        ink_only = _without_root_text(_page())

        assert page_edit.set_page_text(ink_only, "") == ink_only

    def test_untouched_blocks_stay_byte_identical(self):
        page = _page() + _UNKNOWN_BLOCK

        edited = page_edit.set_page_text(page, "Changed")

        before = page_edit._split_blocks(page)
        after = page_edit._split_blocks(edited)
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert len(before) == len(after)
        assert [type(page_edit._parse_block(before[i])) for i in changed] == [RootTextBlock]


class TestMalformedPages:
    @pytest.mark.parametrize(
        "data",
        [b"not a page", HEADER_V6 + b"\x10\x00", HEADER_V6 + struct.pack("<IBBBB", 50, 0, 1, 1, 5)],
        ids=["no-header", "truncated-header", "truncated-block"],
    )
    def test_rejected_with_page_edit_error(self, data):
        with pytest.raises(page_edit.PageEditError):
            page_edit.erase_strokes(data)
        with pytest.raises(page_edit.PageEditError):
            page_edit.set_page_text(data, "x")
