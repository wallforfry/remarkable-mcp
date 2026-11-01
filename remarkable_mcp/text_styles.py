"""Native paragraph styles for typed text, as Markdown line prefixes.

Markdown line prefixes select one of the paragraph styles a Paper Pro offers,
and the styles are written into the page's root text block; reading a page
turns its styles back into the same prefixes. rmscene 0.8 does not know the
numbered-list code (10) and cannot read or write the two extra fields that
tell the larger heading from the smaller one, so the style entries are encoded
and decoded here. Their bytes match a page styled on a Paper Pro and read back from the
cloud; the meaning of the extra fields is unknown.
"""

from __future__ import annotations

import enum
import io
import re
from dataclasses import dataclass, replace
from typing import Optional

from rmscene.scene_items import Text
from rmscene.scene_stream import (
    RootTextBlock,
    text_item_from_stream,
    text_item_to_stream,
)
from rmscene.tagged_block_common import CrdtId, LwwValue, TagType
from rmscene.tagged_block_reader import TaggedBlockReader
from rmscene.tagged_block_writer import TaggedBlockWriter
from rmscene.text import TextDocument


class TextStyle(enum.Enum):
    """Paragraph styles the generated pages can carry.

    Each value is ``(format code, extra fields)``; an extra field is
    ``(tag index, "byte" | "int", value)`` written after the format code.
    """

    BODY = (1, ())
    TITLE = (2, ())
    HEADING = (3, ((2, "byte", 2), (3, "int", 3)))
    SUBHEADING = (3, ())
    BULLET = (4, ())
    BULLET2 = (5, ())
    CHECKBOX = (6, ())
    CHECKBOX_CHECKED = (7, ())
    NUMBERED = (10, ())

    @property
    def code(self) -> int:
        return self.value[0]

    @property
    def extra_fields(self) -> tuple:
        return self.value[1]


# Checked against in order, so longer prefixes come before the ones they
# start with ("- [ ] " before "- ", "### " before "# ").
LINE_PREFIXES = (
    ("- [ ] ", TextStyle.CHECKBOX),
    ("- [x] ", TextStyle.CHECKBOX_CHECKED),
    ("- [X] ", TextStyle.CHECKBOX_CHECKED),
    ("  - ", TextStyle.BULLET2),
    ("  * ", TextStyle.BULLET2),
    ("- ", TextStyle.BULLET),
    ("* ", TextStyle.BULLET),
    ("### ", TextStyle.SUBHEADING),
    ("## ", TextStyle.HEADING),
    ("# ", TextStyle.TITLE),
)
# The device numbers list items itself, so the typed number is dropped.
_NUMBERED_PREFIX = re.compile(r"\d+\. ")

# simple_text_document stores the whole text as one CRDT item whose characters
# get consecutive ids starting here; a paragraph's style is keyed by the id of
# the newline that opens it, and the first paragraph by the (0, 0) end marker.
_TEXT_FIRST_CHAR_ID = CrdtId(1, 16)
_STYLE_TIMESTAMP = CrdtId(1, 15)
# Marker byte that opens every style entry on the device.
_STYLE_ENTRY_MARKER = 17


def parse_paragraphs(text: str) -> list[tuple[TextStyle, str]]:
    """Split ``text`` on newlines into ``(style, content)`` paragraphs.

    A line starting with one of :data:`LINE_PREFIXES`, or with ``<digits>. ``
    for a numbered item, takes that style and loses the prefix; every other
    line is ``BODY``.
    """
    paragraphs = []
    for line in text.split("\n"):
        style, content = TextStyle.BODY, line
        for prefix, prefix_style in LINE_PREFIXES:
            if line.startswith(prefix):
                style, content = prefix_style, line[len(prefix) :]
                break
        else:
            numbered = _NUMBERED_PREFIX.match(line)
            if numbered:
                style, content = TextStyle.NUMBERED, line[numbered.end() :]
        paragraphs.append((style, content))
    return paragraphs


def paragraph_styles(
    paragraphs: list[tuple[TextStyle, str]],
    first_char_id: CrdtId = _TEXT_FIRST_CHAR_ID,
    timestamp: CrdtId = _STYLE_TIMESTAMP,
) -> dict:
    """Return the ``Text.styles`` map for paragraphs joined with newlines.

    ``first_char_id`` is the id of the first character of the joined text, and
    ``timestamp`` the last-writer-wins stamp of every entry. Like
    ``simple_text_document``, later paragraphs get an entry only when they are
    not ``BODY``: an unstyled paragraph renders as body text, and plain input
    keeps the exact bytes of the unstyled builder.
    """
    styles = {CrdtId(0, 0): LwwValue(timestamp=timestamp, value=paragraphs[0][0])}
    offset = len(paragraphs[0][1]) + 1
    for style, content in paragraphs[1:]:
        if style != TextStyle.BODY:
            newline_id = CrdtId(first_char_id.part1, first_char_id.part2 + offset - 1)
            styles[newline_id] = LwwValue(timestamp=timestamp, value=style)
        offset += len(content) + 1
    return styles


def _write_style_entry(char_id: CrdtId, value: LwwValue, writer: TaggedBlockWriter) -> None:
    style = value.value
    writer.data.write_crdt_id(char_id)
    writer.write_id(1, value.timestamp)
    with writer.write_subblock(2):
        writer.data.write_uint8(_STYLE_ENTRY_MARKER)
        writer.data.write_uint8(style.code)
        for index, kind, field_value in style.extra_fields:
            if kind == "byte":
                writer.write_byte(index, field_value)
            else:
                writer.write_int(index, field_value)


@dataclass
class StyledRootTextBlock(RootTextBlock):
    """Root text block whose ``styles`` map to :class:`TextStyle` values.

    Serialised with the same layout as ``RootTextBlock.to_stream``; only the
    style entries are encoded differently. rmscene resolves block types on
    ``RootTextBlock`` before its subclasses, so reading is unaffected.
    """

    def to_stream(self, writer: TaggedBlockWriter) -> None:
        writer.write_id(1, self.block_id)
        with writer.write_subblock(2):
            text_items = self.value.items.sequence_items()
            with writer.write_subblock(1):
                with writer.write_subblock(1):
                    writer.data.write_varuint(len(text_items))
                    for item in text_items:
                        text_item_to_stream(item, writer)
            with writer.write_subblock(2):
                with writer.write_subblock(1):
                    writer.data.write_varuint(len(self.value.styles))
                    for char_id, value in self.value.styles.items():
                        _write_style_entry(char_id, value, writer)
        with writer.write_subblock(3):
            writer.data.write_float64(self.value.pos_x)
            writer.data.write_float64(self.value.pos_y)
        writer.write_float(4, self.value.width)


def styled_root_text(
    block: RootTextBlock, paragraphs: list[tuple[TextStyle, str]]
) -> StyledRootTextBlock:
    """Return ``block`` with its styles replaced by those of ``paragraphs``."""
    return StyledRootTextBlock(
        block_id=block.block_id,
        value=replace(block.value, styles=paragraph_styles(paragraphs)),
        extra_data=block.extra_data,
    )


# The prefix each style is written back as: the first one that selects it.
_STYLE_PREFIXES = {}
for _prefix, _style in LINE_PREFIXES:
    _STYLE_PREFIXES.setdefault(_style, _prefix)


def _style_for(code: int, extra_fields: tuple) -> TextStyle:
    """Return the style a format code and its extra fields encode.

    An unknown code reads as body text, and a known code with unknown extra
    fields as the style that has that code and none.
    """
    candidates = [style for style in TextStyle if style.code == code]
    for style in candidates:
        if style.extra_fields == extra_fields:
            return style
    return next((style for style in candidates if not style.extra_fields), TextStyle.BODY)


def _read_style_entry(reader: TaggedBlockReader) -> tuple[CrdtId, TextStyle]:
    char_id = reader.data.read_crdt_id()
    reader.read_id(1)
    with reader.read_subblock(2) as entry:
        end = entry.offset + entry.size
        reader.data.read_uint8()  # _STYLE_ENTRY_MARKER
        code = reader.data.read_uint8()
        extra = []
        if reader.data.tell() < end and reader.data.check_tag(2, TagType.Byte1):
            extra.append((2, "byte", reader.read_byte(2)))
        if reader.data.tell() < end and reader.data.check_tag(3, TagType.Byte4):
            extra.append((3, "int", reader.read_int(3)))
    return char_id, _style_for(code, tuple(extra))


def _read_root_text_styles(reader: TaggedBlockReader) -> dict[CrdtId, TextStyle]:
    """Read the style map of the root text block ``reader`` is positioned in."""
    reader.read_id(1)
    with reader.read_subblock(2):
        with reader.read_subblock(1):
            with reader.read_subblock(1):
                for _ in range(reader.data.read_varuint()):
                    text_item_from_stream(reader)
        with reader.read_subblock(2):
            with reader.read_subblock(1):
                count = reader.data.read_varuint()
                styles = dict(_read_style_entry(reader) for _ in range(count))
    with reader.read_subblock(3):
        reader.data.read_float64()
        reader.data.read_float64()
    reader.read_float(4)
    return styles


def read_paragraph_styles(page_bytes: bytes) -> dict[CrdtId, TextStyle]:
    """Return the paragraph styles of a ``.rm`` page, keyed like ``Text.styles``.

    Empty when the page has no typed text.
    """
    reader = TaggedBlockReader(io.BytesIO(page_bytes))
    reader.read_header()
    while True:
        with reader.read_block() as block:
            if block is None:
                return {}
            if block.block_type == RootTextBlock.BLOCK_TYPE:
                return _read_root_text_styles(reader)
            reader.data.read_bytes(block.size)


def markdown_lines(text: Text, styles: Optional[dict] = None) -> list[str]:
    """Return a page's paragraphs with the line prefixes of their styles.

    ``styles`` comes from :func:`read_paragraph_styles`; without it every
    paragraph reads as body text. Numbered items are numbered from 1 in each
    run, the way the device displays them. The output parses back to the same
    styles with :func:`parse_paragraphs`.
    """
    styles = styles or {}
    lines = []
    number = 0
    for paragraph in TextDocument.from_scene_item(text).contents:
        style = styles.get(paragraph.start_id, TextStyle.BODY)
        number = number + 1 if style is TextStyle.NUMBERED else 0
        prefix = f"{number}. " if style is TextStyle.NUMBERED else _STYLE_PREFIXES.get(style, "")
        lines.append(prefix + str(paragraph))
    return lines
