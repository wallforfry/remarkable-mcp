"""Pure ``.rm`` edits: erase ink from a page and replace its typed text.

Like :mod:`remarkable_mcp.strokes`, this module never round-trips a whole page
through rmscene, which drops bytes it does not understand on newer firmware.
The page is split on its block headers, only the blocks an edit touches are
serialised again, and every other block is kept byte for byte.

Removals follow the CRDT model the device merges pages with, so nothing is
dropped outright:

- An erased stroke becomes a tombstone block carrying the same item id.
- Replaced text keeps its characters as deleted runs (same ids, no value), and
  the new text is inserted after them as one run with fresh ids, its paragraph
  styles stamped later than every existing entry.
"""

from __future__ import annotations

import io
import struct
from dataclasses import replace
from typing import Optional

from rmscene import HEADER_V6, read_blocks
from rmscene.crdt_sequence import CrdtSequence, CrdtSequenceItem
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
from remarkable_mcp.text_styles import StyledRootTextBlock, paragraph_styles, parse_paragraphs

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


def _stroke_hit(line, bounds: Optional[tuple]) -> bool:
    if bounds is None:
        return True
    x0, y0, x1, y1 = bounds
    return any(x0 <= p.x <= x1 and y0 <= p.y <= y1 for p in line.points)


def erase_strokes(original_bytes: bytes, region: Optional[list] = None) -> tuple[bytes, int]:
    """Erase strokes from a page and return ``(new_bytes, erased_count)``.

    With no ``region`` every stroke is erased. Otherwise a stroke is erased
    when any of its points falls inside the normalised top-left
    ``[x0, y0, x1, y1]`` rectangle, like the device eraser sweeping that area.
    Typed text is left alone. The input bytes come back unchanged when nothing
    matches.
    """
    segments = _split_blocks(original_bytes)
    blocks = [_parse_block(segment) for segment in segments]
    paper_w, paper_h = strokes_mod.get_paper_size(blocks)
    bounds = _region_bounds(region, paper_w, paper_h)

    edited = []
    erased = 0
    for segment, block in zip(segments, blocks):
        if (
            isinstance(block, SceneLineItemBlock)
            and block.item.value is not None
            and _stroke_hit(block.item.value, bounds)
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


def _item_length(item: CrdtSequenceItem) -> int:
    """Return how many character ids a text item spans.

    An unstyled blank page holds one live item with an empty string, which
    spans no character.
    """
    if item.deleted_length:
        return item.deleted_length
    return len(item.value) if isinstance(item.value, str) else 1


def _last_char_id(item: CrdtSequenceItem) -> CrdtId:
    return CrdtId(item.item_id.part1, item.item_id.part2 + max(_item_length(item), 1) - 1)


def _insertion_anchor(items: list) -> CrdtId:
    """Return the id of the last character in ``items``, or the end marker."""
    spanning = [item for item in items if _item_length(item)]
    return _last_char_id(spanning[-1]) if spanning else _END_MARKER


def _max_crdt_id(blocks: list, text: Optional[Text]) -> CrdtId:
    """Return the highest id on the page, text characters and style stamps included."""
    ids = list(strokes_mod._iter_crdt_ids(blocks))
    if text is not None:
        ids += [_last_char_id(item) for item in text.items.sequence_items()]
        ids += [value.timestamp for value in text.styles.values()]
    return max(ids, key=lambda cid: (cid.part1, cid.part2), default=_END_MARKER)


def _deleted_run(item: CrdtSequenceItem) -> CrdtSequenceItem:
    if item.deleted_length or not _item_length(item):
        return item
    return replace(item, deleted_length=_item_length(item), value="")


def _root_text(blocks: list) -> Optional[int]:
    """Return the index of the page's root text block, or None when it has none."""
    for index, block in enumerate(blocks):
        if isinstance(block, RootTextBlock):
            return index
        if isinstance(block, UnreadableBlock) and block.info.block_type == _ROOT_TEXT_BLOCK_TYPE:
            raise PageEditError(f"The page's typed text cannot be read: {block.error}")
    return None


def _replacement_text_block(
    old_block: Optional[RootTextBlock], last: CrdtId, text: str
) -> StyledRootTextBlock:
    """Build a root text block holding ``text`` in place of ``old_block``'s.

    Old characters become deleted runs; the new text is one run inserted after
    them, with ids and a style stamp above ``last``, the page's highest id.
    """
    old_text = old_block.value if old_block is not None else None
    old_items = old_text.items.sequence_items() if old_text else []
    author = last.part1 or 1
    timestamp = CrdtId(author, last.part2 + 1)
    first_char = CrdtId(author, last.part2 + 2)

    paragraphs = parse_paragraphs(text)
    body = "\n".join(content for _, content in paragraphs)
    items = [_deleted_run(item) for item in old_items]
    if body:
        items.append(
            CrdtSequenceItem(
                item_id=first_char,
                left_id=_insertion_anchor(old_items),
                right_id=_END_MARKER,
                deleted_length=0,
                value=body,
            )
        )

    pos_x, pos_y, width = (
        (old_text.pos_x, old_text.pos_y, old_text.width) if old_text else _DEFAULT_TEXT_GEOMETRY
    )
    return StyledRootTextBlock(
        block_id=_END_MARKER,
        value=Text(
            items=CrdtSequence(items),
            styles=paragraph_styles(paragraphs, first_char, timestamp),
            pos_x=pos_x,
            pos_y=pos_y,
            width=width,
        ),
        extra_data=old_block.extra_data if old_block is not None else b"",
    )


def set_page_text(original_bytes: bytes, text: str) -> bytes:
    """Replace a page's typed text and return the new page bytes.

    ``text`` uses the Markdown line prefixes of
    :func:`remarkable_mcp.text_styles.parse_paragraphs`; an empty string
    removes the typed text. Ink is left alone, but a highlight anchored to the
    old characters loses its anchor. The input bytes come back unchanged when
    the page has no typed text and ``text`` is empty. A page without typed
    text gets its new text block before its first layer, the order
    ``simple_text_document`` uses.
    """
    segments = _split_blocks(original_bytes)
    blocks = [_parse_block(segment) for segment in segments]
    root_index = _root_text(blocks)
    if root_index is None and not text:
        return original_bytes

    old_block = blocks[root_index] if root_index is not None else None
    last = _max_crdt_id(blocks, old_block.value if old_block is not None else None)
    new_segment = _serialize_block(_replacement_text_block(old_block, last, text))

    if root_index is not None:
        segments[root_index] = new_segment
    else:
        layers = [i for i, block in enumerate(blocks) if isinstance(block, TreeNodeBlock)]
        segments.insert(layers[0] if layers else len(segments), new_segment)
    return HEADER_V6 + b"".join(segments)
