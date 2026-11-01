"""Pure ``.rm`` edits: erase ink from a page and replace its typed text.

Like :mod:`remarkable_mcp.strokes`, this module never round-trips a whole page
through rmscene, which drops bytes it does not understand on newer firmware.
The page is split on its block headers, only the blocks an edit touches are
serialised again, and every other block is kept byte for byte.

Removals follow the CRDT model the device merges pages with, so nothing is
dropped outright:

- An erased stroke becomes a tombstone block carrying the same item id.
- A text edit deletes and inserts only the characters that change, like the
  device keyboard (see :mod:`remarkable_mcp.text_diff`), its paragraph styles
  stamped later than every existing entry.
"""

from __future__ import annotations

import io
import struct
from dataclasses import replace
from typing import Optional

from rmscene import HEADER_V6, read_blocks
from rmscene.crdt_sequence import CrdtSequence
from rmscene.scene_items import Text
from rmscene.scene_stream import (
    Block,
    RootTextBlock,
    SceneLineItemBlock,
    SceneTombstoneItemBlock,
    TreeNodeBlock,
    UnreadableBlock,
)
from rmscene.tagged_block_common import CrdtId

from remarkable_mcp import strokes as strokes_mod
from remarkable_mcp.extract import ink_group_offsets
from remarkable_mcp.text_diff import TextEdit, edit_text
from remarkable_mcp.text_styles import StyledRootTextBlock

# uint32 body length, then unknown, min version, current version, block type.
_BLOCK_HEADER = struct.Struct("<IBBBB")
_ROOT_TEXT_BLOCK_TYPE = RootTextBlock.BLOCK_TYPE
_END_MARKER = CrdtId(0, 0)
# Root text geometry simple_text_document uses when a page has no text yet.
_DEFAULT_TEXT_GEOMETRY = (-468.0, 234.0, 936.0)


class PageEditError(ValueError):
    """Raised when a page cannot be edited."""


def _split_blocks(data: bytes) -> list[bytes]:
    """Return each top-level block of a v6 page as raw bytes, header included."""
    if not data.startswith(HEADER_V6):
        raise PageEditError("The page is not a reMarkable v6 .rm file.")
    segments = []
    offset = len(HEADER_V6)
    while offset < len(data):
        if offset + _BLOCK_HEADER.size > len(data):
            raise PageEditError("The page ends inside a block header.")
        end = offset + _BLOCK_HEADER.size + _BLOCK_HEADER.unpack_from(data, offset)[0]
        if end > len(data):
            raise PageEditError("The page ends inside a block.")
        segments.append(data[offset:end])
        offset = end
    return segments


def _parse_block(segment: bytes) -> Block:
    return next(read_blocks(io.BytesIO(HEADER_V6 + segment)))


def _serialize_block(block: Block) -> bytes:
    return strokes_mod.serialize_new_blocks([block])


def _region_bounds(region: Optional[list], paper_w: int, paper_h: int) -> Optional[tuple]:
    """Map a normalised top-left ``[x0, y0, x1, y1]`` region to page units."""
    if region is None:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in region)
    except (TypeError, ValueError) as e:
        raise PageEditError("region must be four numbers: [x0, y0, x1, y1].") from e
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise PageEditError("region must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1.")
    return ((x0 - 0.5) * paper_w, y0 * paper_h, (x1 - 0.5) * paper_w, y1 * paper_h)


def _stroke_hit(line, offset: tuple, bounds: Optional[tuple]) -> bool:
    if bounds is None:
        return True
    x0, y0, x1, y1 = bounds
    dx, dy = offset
    return any(x0 <= p.x + dx <= x1 and y0 <= p.y + dy <= y1 for p in line.points)


def erase_strokes(original_bytes: bytes, region: Optional[list] = None) -> tuple[bytes, int]:
    """Erase strokes from a page and return ``(new_bytes, erased_count)``.

    With no ``region`` every stroke is erased. Otherwise a stroke is erased
    when any of its points falls inside the normalised top-left
    ``[x0, y0, x1, y1]`` rectangle, like the device eraser sweeping that area.
    Ink anchored to typed text is placed where the canvas renders it before
    the test. Typed text is left alone. The input bytes come back unchanged when nothing
    matches.
    """
    segments = _split_blocks(original_bytes)
    blocks = [_parse_block(segment) for segment in segments]
    paper_w, paper_h = strokes_mod.get_paper_size(blocks)
    bounds = _region_bounds(region, paper_w, paper_h)
    offsets = ink_group_offsets(blocks) if bounds is not None else {}

    edited = []
    erased = 0
    for segment, block in zip(segments, blocks):
        if (
            isinstance(block, SceneLineItemBlock)
            and block.item.value is not None
            and _stroke_hit(block.item.value, offsets.get(block.parent_id, (0.0, 0.0)), bounds)
        ):
            tombstone = SceneTombstoneItemBlock(
                parent_id=block.parent_id,
                item=replace(block.item, deleted_length=1, value=None),
            )
            edited.append(_serialize_block(tombstone))
            erased += 1
        else:
            edited.append(segment)
    if not erased:
        return original_bytes, 0
    return HEADER_V6 + b"".join(edited), erased


def _root_text(blocks: list) -> Optional[int]:
    """Return the index of the page's root text block, or None when it has none."""
    for index, block in enumerate(blocks):
        if isinstance(block, RootTextBlock):
            return index
        if isinstance(block, UnreadableBlock) and block.info.block_type == _ROOT_TEXT_BLOCK_TYPE:
            raise PageEditError(f"The page's typed text cannot be read: {block.error}")
    return None


def _replacement_text_block(old_block: Optional[RootTextBlock], edit: TextEdit) -> Block:
    """Build the root text block holding ``edit`` in place of ``old_block``."""
    old_text = old_block.value if old_block is not None else None
    pos_x, pos_y, width = (
        (old_text.pos_x, old_text.pos_y, old_text.width) if old_text else _DEFAULT_TEXT_GEOMETRY
    )
    return StyledRootTextBlock(
        block_id=_END_MARKER,
        value=Text(
            items=CrdtSequence(edit.items),
            styles=edit.styles,
            pos_x=pos_x,
            pos_y=pos_y,
            width=width,
        ),
        extra_data=old_block.extra_data if old_block is not None else b"",
    )


def _detached_groups(blocks: list, deleted_ids: frozenset) -> int:
    """Count ink groups anchored to a character the edit deleted."""
    return sum(
        1
        for block in blocks
        if isinstance(block, TreeNodeBlock)
        and block.group.anchor_id is not None
        and block.group.anchor_id.value in deleted_ids
    )


def set_page_text(original_bytes: bytes, text: str) -> tuple[bytes, int]:
    """Replace a page's typed text; return ``(new_bytes, detached_annotations)``.

    ``text`` uses the Markdown line prefixes of
    :func:`remarkable_mcp.text_styles.parse_paragraphs`; an empty string
    removes the typed text. Only the characters that differ are deleted or
    inserted (see :mod:`remarkable_mcp.text_diff`), so ink anchored to a kept
    character follows its line. ``detached_annotations`` counts the ink groups
    anchored to a deleted character instead. The input bytes come back
    unchanged when the page has no typed text and ``text`` is empty. A page
    without typed text gets its new text block before its first layer, the
    order ``simple_text_document`` uses.
    """
    segments = _split_blocks(original_bytes)
    blocks = [_parse_block(segment) for segment in segments]
    root_index = _root_text(blocks)
    if root_index is None and not text:
        return original_bytes, 0

    old_block = blocks[root_index] if root_index is not None else None
    old_text = old_block.value if old_block is not None else None
    edit = edit_text(old_text, text, strokes_mod.max_crdt_id(blocks))
    new_segment = _serialize_block(_replacement_text_block(old_block, edit))

    if root_index is not None:
        segments[root_index] = new_segment
    else:
        layers = [i for i, block in enumerate(blocks) if isinstance(block, TreeNodeBlock)]
        segments.insert(layers[0] if layers else len(segments), new_segment)
    return HEADER_V6 + b"".join(segments), _detached_groups(blocks, edit.deleted_ids)
