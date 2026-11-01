"""Turn a page's typed text into new text as a minimal CRDT edit.

The device keyboard never retypes a page: it deletes and inserts only the
characters that change, so every other character keeps its CRDT id. Ink
written over typed text hangs off one of those ids (``Group.anchor_id``) and
follows its line as the text reflows, so an edit that keeps the ids keeps the
annotations in place. :func:`edit_text` reproduces that: the old and new text
are diffed character by character, removed characters become deleted runs
under their old ids, and each inserted span is one new run.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Optional, Union

from rmscene.crdt_sequence import CrdtSequenceItem
from rmscene.scene_items import Text
from rmscene.tagged_block_common import CrdtId, LwwValue

from remarkable_mcp.text_styles import TextStyle, parse_paragraphs

END_MARKER = CrdtId(0, 0)


@dataclass(frozen=True)
class TextEdit:
    """The edited text sequence, its paragraph styles, and the ids it deleted."""

    items: list
    styles: dict
    deleted_ids: frozenset


@dataclass(frozen=True)
class _Char:
    """One live character of an old text item."""

    char_id: CrdtId
    char: str
    item_index: int
    # The item's own left id for its first character, None for the others.
    left_id: Optional[CrdtId]
    right_id: CrdtId


@dataclass(frozen=True)
class _Insert:
    text: str


_Unit = Union[_Char, CrdtSequenceItem, _Insert]


def item_span(item: CrdtSequenceItem) -> int:
    """Return how many character ids an item spans; an empty live string spans none."""
    if item.deleted_length:
        return item.deleted_length
    return len(item.value) if isinstance(item.value, str) else 1


def _flatten(items: list) -> list:
    """Split live string items into characters; keep every other item whole."""
    units: list = []
    for index, item in enumerate(items):
        if item.deleted_length or not isinstance(item.value, str) or not item.value:
            units.append(item)
            continue
        base = item.item_id
        units += [
            _Char(
                CrdtId(base.part1, base.part2 + offset),
                char,
                index,
                item.left_id if offset == 0 else None,
                item.right_id,
            )
            for offset, char in enumerate(item.value)
        ]
    return units


def _plan(chars: list, body: str) -> tuple[set, dict]:
    """Return the indices of ``chars`` to delete and the text to insert before each index."""
    old = "".join(c.char for c in chars)
    deleted: set = set()
    inserts: dict = {}
    matcher = difflib.SequenceMatcher(None, old, body, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            deleted.update(range(i1, i2))
        if tag in ("insert", "replace"):
            inserts[i2] = body[j1:j2]
    return deleted, inserts


def _ordered_units(units: list, deleted: set, inserts: dict) -> list:
    """Interleave the old units with the planned inserts, flagging deletions."""
    ordered: list = []
    index = 0
    for unit in units:
        if isinstance(unit, _Char):
            if index in inserts:
                ordered.append(_Insert(inserts[index]))
            ordered.append((unit, index in deleted))
            index += 1
        else:
            ordered.append(unit)
    if index in inserts:
        ordered.append(_Insert(inserts[index]))
    return ordered


def _first_id(unit) -> Optional[CrdtId]:
    if isinstance(unit, tuple):
        return unit[0].char_id
    if isinstance(unit, CrdtSequenceItem) and item_span(unit):
        return unit.item_id
    return None


def _run_item(run: list) -> CrdtSequenceItem:
    """Build one item from consecutive ``(char, deleted)`` pairs of one old item.

    A run that starts mid-item hangs off the character before it, as when the
    device splits an item around an edit.
    """
    first, is_deleted = run[0]
    left = first.left_id
    if left is None:
        left = CrdtId(first.char_id.part1, first.char_id.part2 - 1)
    text = "".join(char.char for char, _ in run)
    return CrdtSequenceItem(
        item_id=first.char_id,
        left_id=left,
        right_id=first.right_id,
        deleted_length=len(text) if is_deleted else 0,
        value="" if is_deleted else text,
    )


def _continues(run: list, char: _Char, is_deleted: bool) -> bool:
    previous, previous_deleted = run[-1]
    return (
        previous_deleted == is_deleted
        and previous.item_index == char.item_index
        and previous.char_id.part2 + 1 == char.char_id.part2
    )


def _emit(ordered: list, next_part2: int, author: int) -> tuple[list, list]:
    """Build the items, and the ``(char, id)`` list of the live text in order."""
    items: list = []
    live: list = []
    last_id = END_MARKER
    run: list = []

    for position, unit in enumerate(ordered):
        if isinstance(unit, tuple):
            char, is_deleted = unit
            if run and not _continues(run, char, is_deleted):
                items.append(_run_item(run))
                run = []
            run.append(unit)
            if not is_deleted:
                live.append((char.char, char.char_id))
            last_id = char.char_id
            continue
        if run:
            items.append(_run_item(run))
            run = []
        if isinstance(unit, _Insert):
            following = (_first_id(u) for u in ordered[position + 1 :])
            items.append(
                CrdtSequenceItem(
                    item_id=CrdtId(author, next_part2),
                    left_id=last_id,
                    right_id=next((i for i in following if i is not None), END_MARKER),
                    deleted_length=0,
                    value=unit.text,
                )
            )
            live += [(c, CrdtId(author, next_part2 + n)) for n, c in enumerate(unit.text)]
            next_part2 += len(unit.text)
            last_id = CrdtId(author, next_part2 - 1)
        else:
            items.append(unit)
            if item_span(unit):
                last_id = CrdtId(unit.item_id.part1, unit.item_id.part2 + item_span(unit) - 1)
    if run:
        items.append(_run_item(run))
    return items, live


def _styles(
    paragraphs: list, live: list, timestamp: CrdtId, old_styles: dict
) -> dict[CrdtId, LwwValue]:
    """Stamp each paragraph's style on the newline that opens it.

    A body paragraph needs no entry, unless the old text styled that newline:
    the explicit body entry then outranks the old one when pages merge.
    """
    keys = [END_MARKER] + [char_id for char, char_id in live if char == "\n"]
    return {
        key: LwwValue(timestamp=timestamp, value=style)
        for index, ((style, _), key) in enumerate(zip(paragraphs, keys))
        if index == 0 or style != TextStyle.BODY or key in old_styles
    }


def edit_text(old: Optional[Text], text: str, last: CrdtId) -> TextEdit:
    """Return the minimal edit from ``old`` to ``text``.

    ``text`` uses the Markdown line prefixes of
    :func:`remarkable_mcp.text_styles.parse_paragraphs`. New ids and the style
    stamp are allocated above ``last``, the page's highest id. Items that are
    already deleted, formatting markers and empty strings are kept verbatim.
    """
    paragraphs = parse_paragraphs(text)
    body = "\n".join(content for _, content in paragraphs)
    units = _flatten(old.items.sequence_items() if old is not None else [])
    chars = [unit for unit in units if isinstance(unit, _Char)]
    deleted, inserts = _plan(chars, body)

    author = last.part1 or 1
    timestamp = CrdtId(author, last.part2 + 1)
    items, live = _emit(_ordered_units(units, deleted, inserts), last.part2 + 2, author)
    old_styles = old.styles if old is not None else {}
    return TextEdit(
        items=items,
        styles=_styles(paragraphs, live, timestamp, old_styles),
        deleted_ids=frozenset(chars[i].char_id for i in deleted),
    )
