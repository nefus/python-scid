"""Index entry encoding/decoding for SCID4 (.si4) and SCID5 (.si5) formats.

An IndexEntry holds all per-game metadata: offsets, name IDs, ratings,
dates, ECO, flags, material signature, etc.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO

from scid._filebuf import (
    decode_text,
    decode_uint32_le,
    encode_text,
    encode_uint32_le,
    pack5,
    read_uint8,
    read_uint16_be,
    read_uint24_be,
    read_uint32_be,
    unpack5,
    write_uint8,
    write_uint16_be,
    write_uint24_be,
    write_uint32_be,
)
from scid.constants import (
    RESULT_NONE,
    SCID4_INDEX_ENTRY_SIZE,
    SCID4_INDEX_HEADER_SIZE,
    SCID4_MAGIC_INDEX,
    SCID4_VERSION,
    SCID5_INDEX_ENTRY_SIZE,
    decode_count,
    encode_count,
)
from scid.dates import (
    decode_event_date_delta,
    encode_event_date_delta,
)


@dataclass
class IndexEntry:
    """Per-game metadata stored in the SCID index file."""

    # Game data location in .sg4/.sg5
    offset: int = 0
    game_data_size: int = 0

    # Name IDs (references into NameBase)
    white_id: int = 0
    black_id: int = 0
    event_id: int = 0
    site_id: int = 0
    round_id: int = 0

    # Ratings
    white_elo: int = 0
    black_elo: int = 0
    white_elo_type: int = 0  # 3-bit rating type
    black_elo_type: int = 0

    # Dates (20-bit packed)
    date: int = 0
    event_date: int = 0

    # Game info
    result: int = RESULT_NONE  # 2-bit
    eco_code: int = 0  # 16-bit
    num_half_moves: int = 0  # max 1023
    flags: int = 0  # 22-bit bitmask

    # Counts (approximate, 4-bit encoded)
    n_comments: int = 0
    n_variations: int = 0
    n_nags: int = 0

    # Material / search
    final_mat_sig: int = 0  # 24-bit
    stored_line_code: int = 0  # 8-bit
    home_pawn_data: bytes = field(default_factory=lambda: b"\x00" * 9)

    # Chess960
    chess960: bool = False


# ======================================================================
# SCID4 index format (.si4)
# ======================================================================


@dataclass
class IndexHeader:
    """SCID4 index file header."""

    version: int = SCID4_VERSION
    base_type: int = 0
    num_games: int = 0
    auto_load: int = 0
    description: str = ""
    custom_flags: list[str] = field(default_factory=lambda: [""] * 6)


def read_si4_header(f: BinaryIO) -> IndexHeader:
    """Read the 182-byte SCID4 index header."""
    magic = f.read(8)
    if magic != SCID4_MAGIC_INDEX:
        raise ValueError(f"Bad .si4 magic: {magic!r}")

    hdr = IndexHeader()
    hdr.version = read_uint16_be(f)
    hdr.base_type = read_uint32_be(f)
    hdr.num_games = read_uint24_be(f)
    hdr.auto_load = read_uint24_be(f)

    desc_bytes = f.read(108)
    hdr.description = decode_text(desc_bytes.split(b"\x00", 1)[0])

    for i in range(6):
        flag_bytes = f.read(9)
        hdr.custom_flags[i] = decode_text(flag_bytes.split(b"\x00", 1)[0])

    return hdr


def write_si4_header(f: BinaryIO, hdr: IndexHeader) -> None:
    """Write the 182-byte SCID4 index header."""
    f.write(SCID4_MAGIC_INDEX)
    write_uint16_be(f, hdr.version)
    write_uint32_be(f, hdr.base_type)
    write_uint24_be(f, hdr.num_games)
    write_uint24_be(f, hdr.auto_load)

    desc = encode_text(hdr.description, 107)
    f.write(desc + b"\x00" * (108 - len(desc)))

    for i in range(6):
        flag = encode_text(hdr.custom_flags[i], 8)
        f.write(flag + b"\x00" * (9 - len(flag)))


def decode_si4_entry(data: bytes) -> IndexEntry:
    """Decode a 47-byte SCID4 index entry."""
    ie = IndexEntry()

    # Bytes 0-3: offset (32-bit big-endian)
    ie.offset = struct.unpack_from(">I", data, 0)[0]

    # Bytes 4-5: lower 16 bits of game data length (big-endian)
    # Byte 6: bit7 = length bit 16, bits 5-0 = custom flags (upper 6 bits of 22-bit flags)
    len_low = struct.unpack_from(">H", data, 4)[0]
    len_flags = data[6]
    ie.game_data_size = ((len_flags & 0x80) << 9) | len_low  # 17-bit length
    custom_flags_6 = len_flags & 0x3F

    # Bytes 7-8: flags (lower 16 bits of 22-bit flags)
    flags_low = struct.unpack_from(">H", data, 7)[0]
    ie.flags = flags_low | (custom_flags_6 << 16)

    # Byte 9: WhiteID_high(4) | BlackID_high(4)
    # Bytes 10-11: WhiteID_low (16-bit big-endian)
    # Bytes 12-13: BlackID_low (16-bit big-endian)
    wb_high = data[9]
    white_low = struct.unpack_from(">H", data, 10)[0]
    black_low = struct.unpack_from(">H", data, 12)[0]
    ie.white_id = ((wb_high & 0xF0) << 12) | white_low
    ie.black_id = ((wb_high & 0x0F) << 16) | black_low

    # Byte 14: EventID_high(3) | SiteID_high(3) | RoundID_high(2)
    # Bytes 15-16: EventID_low (16-bit big-endian)
    # Bytes 17-18: SiteID_low (16-bit big-endian)
    # Bytes 19-20: RoundID_low (16-bit big-endian)
    esr_high = data[14]
    event_low = struct.unpack_from(">H", data, 15)[0]
    site_low = struct.unpack_from(">H", data, 17)[0]
    round_low = struct.unpack_from(">H", data, 19)[0]
    ie.event_id = ((esr_high & 0xE0) << 11) | event_low
    ie.site_id = ((esr_high & 0x1C) << 14) | site_low
    ie.round_id = ((esr_high & 0x03) << 16) | round_low

    # Bytes 21-22: VarCounts + result (16-bit big-endian)
    # bits 0-3: variations, 4-7: comments, 8-11: NAGs, 12-15: result
    var_counts = struct.unpack_from(">H", data, 21)[0]
    ie.n_variations = decode_count(var_counts & 0x0F)
    ie.n_comments = decode_count((var_counts >> 4) & 0x0F)
    ie.n_nags = decode_count((var_counts >> 8) & 0x0F)
    ie.result = (var_counts >> 12) & 0x0F

    # Bytes 23-24: ECO code (16-bit big-endian)
    ie.eco_code = struct.unpack_from(">H", data, 23)[0]

    # Bytes 25-28: date (20 bits, low) + event date delta (12 bits, high)
    date_edate = struct.unpack_from(">I", data, 25)[0]
    ie.date = date_edate & 0xFFFFF
    edate_raw = date_edate >> 20
    ie.event_date = decode_event_date_delta(ie.date, edate_raw)

    # Bytes 29-30: white elo (12 bits) + rating type (4 bits)
    w_elo_raw = struct.unpack_from(">H", data, 29)[0]
    ie.white_elo = w_elo_raw & 0xFFF
    ie.white_elo_type = (w_elo_raw >> 12) & 0x0F

    # Bytes 31-32: black elo (12 bits) + rating type (4 bits)
    b_elo_raw = struct.unpack_from(">H", data, 31)[0]
    ie.black_elo = b_elo_raw & 0xFFF
    ie.black_elo_type = (b_elo_raw >> 12) & 0x0F

    # Bytes 33-36: finalMatSig (24 bits, low) + storedLineCode (8 bits, high)
    mat_raw = struct.unpack_from(">I", data, 33)[0]
    ie.final_mat_sig = mat_raw & 0xFFFFFF
    ie.stored_line_code = (mat_raw >> 24) & 0xFF

    # Byte 37: NumHalfMoves low 8 bits
    # Byte 38: NumHalfMoves high 2 bits (bits 7-6) + HomePawnData[0] count (bits 5-0)
    num_hm_low = data[37]
    pawn_data_0 = data[38]
    ie.num_half_moves = ((pawn_data_0 & 0xC0) << 2) | num_hm_low

    # Bytes 38-46: home pawn data (9 bytes: count from byte 38 + 8 bytes from 39-46)
    hp = bytearray(9)
    hp[0] = pawn_data_0 & 0x3F
    hp[1:] = data[39:47]
    ie.home_pawn_data = bytes(hp)

    return ie


def encode_si4_entry(ie: IndexEntry) -> bytes:
    """Encode an IndexEntry into 47 bytes (SCID4 format)."""
    buf = bytearray(SCID4_INDEX_ENTRY_SIZE)

    # Bytes 0-3: offset
    struct.pack_into(">I", buf, 0, ie.offset & 0xFFFFFFFF)

    # Bytes 4-5: lower 16 bits of game data length (big-endian)
    struct.pack_into(">H", buf, 4, ie.game_data_size & 0xFFFF)

    # Byte 6: bit7 = length bit 16, bits 5-0 = custom flags (upper 6 of 22-bit flags)
    custom_flags_6 = (ie.flags >> 16) & 0x3F
    len_flags = ((ie.game_data_size >> 9) & 0x80) | custom_flags_6
    buf[6] = len_flags

    # Bytes 7-8: flags lower 16 bits
    struct.pack_into(">H", buf, 7, ie.flags & 0xFFFF)

    # Byte 9: WhiteID_high(4) | BlackID_high(4)
    wb_high = ((ie.white_id & 0x0F0000) >> 12) | ((ie.black_id & 0x0F0000) >> 16)
    buf[9] = wb_high
    # Bytes 10-11: WhiteID_low
    struct.pack_into(">H", buf, 10, ie.white_id & 0xFFFF)
    # Bytes 12-13: BlackID_low
    struct.pack_into(">H", buf, 12, ie.black_id & 0xFFFF)

    # Byte 14: EventID_high(3) | SiteID_high(3) | RoundID_high(2)
    esr_high = (
        ((ie.event_id & 0x070000) >> 11)
        | ((ie.site_id & 0x070000) >> 14)
        | ((ie.round_id & 0x030000) >> 16)
    )
    buf[14] = esr_high
    # Bytes 15-16: EventID_low
    struct.pack_into(">H", buf, 15, ie.event_id & 0xFFFF)
    # Bytes 17-18: SiteID_low
    struct.pack_into(">H", buf, 17, ie.site_id & 0xFFFF)
    # Bytes 19-20: RoundID_low
    struct.pack_into(">H", buf, 19, ie.round_id & 0xFFFF)

    # Bytes 21-22: VarCounts + result (16-bit big-endian)
    var_counts = (
        encode_count(ie.n_variations)
        | (encode_count(ie.n_comments) << 4)
        | (encode_count(ie.n_nags) << 8)
        | ((ie.result & 0x0F) << 12)
    )
    struct.pack_into(">H", buf, 21, var_counts)

    # Bytes 23-24: ECO
    struct.pack_into(">H", buf, 23, ie.eco_code)

    # Bytes 25-28: date (20 bits, low) + event date delta (12 bits, high)
    delta = encode_event_date_delta(ie.date, ie.event_date)
    date_edate = (delta << 20) | (ie.date & 0xFFFFF)
    struct.pack_into(">I", buf, 25, date_edate)

    # Bytes 29-30: white elo (12 bits) + rating type (4 bits)
    w_elo = (ie.white_elo & 0xFFF) | ((ie.white_elo_type & 0x0F) << 12)
    struct.pack_into(">H", buf, 29, w_elo)

    # Bytes 31-32: black elo (12 bits) + rating type (4 bits)
    b_elo = (ie.black_elo & 0xFFF) | ((ie.black_elo_type & 0x0F) << 12)
    struct.pack_into(">H", buf, 31, b_elo)

    # Bytes 33-36: finalMatSig (24 bits, low) + storedLineCode (8 bits, high)
    mat_raw = (ie.final_mat_sig & 0xFFFFFF) | (ie.stored_line_code << 24)
    struct.pack_into(">I", buf, 33, mat_raw)

    # Byte 37: NumHalfMoves low 8 bits
    buf[37] = ie.num_half_moves & 0xFF

    # Byte 38: NumHalfMoves high 2 bits (bits 7-6) + HomePawnData[0] count (bits 5-0)
    hp_count = ie.home_pawn_data[0] if ie.home_pawn_data else 0
    buf[38] = ((ie.num_half_moves >> 8) & 0x03) << 6 | (hp_count & 0x3F)

    # Bytes 39-46: home pawn data bytes 1-8
    hp = ie.home_pawn_data
    for i in range(min(8, len(hp) - 1)):
        buf[39 + i] = hp[1 + i] if 1 + i < len(hp) else 0

    return bytes(buf)


# ======================================================================
# SCID5 index format (.si5)
# ======================================================================


def decode_si5_entry(data: bytes) -> IndexEntry:
    """Decode a 56-byte SCID5 index entry (little-endian)."""
    ie = IndexEntry()

    w0 = decode_uint32_le(data, 0)
    n_comments_raw, ie.white_id = unpack5(w0, 4)
    ie.n_comments = decode_count(n_comments_raw)

    w1 = decode_uint32_le(data, 4)
    n_var_raw, ie.black_id = unpack5(w1, 4)
    ie.n_variations = decode_count(n_var_raw)

    w2 = decode_uint32_le(data, 8)
    n_nags_raw, ie.event_id = unpack5(w2, 4)
    ie.n_nags = decode_count(n_nags_raw)

    ie.site_id = decode_uint32_le(data, 12)

    w4 = decode_uint32_le(data, 16)
    chess960, ie.round_id = unpack5(w4, 1)
    ie.chess960 = bool(chess960)

    w5 = decode_uint32_le(data, 20)
    ie.white_elo, ie.date = unpack5(w5, 12)

    w6 = decode_uint32_le(data, 24)
    ie.black_elo, ie.event_date = unpack5(w6, 12)

    w7 = decode_uint32_le(data, 28)
    ie.num_half_moves, ie.flags = unpack5(w7, 10)

    w8 = decode_uint32_le(data, 32)
    ie.game_data_size, offset_high = unpack5(w8, 17)

    offset_low = decode_uint32_le(data, 36)
    ie.offset = (offset_high << 32) | offset_low

    w10 = decode_uint32_le(data, 40)
    ie.stored_line_code, ie.final_mat_sig = unpack5(w10, 8)

    w11 = decode_uint32_le(data, 44)
    compound, ie.eco_code = unpack5(w11, 16)
    hp_count = (compound >> 8) & 0xFF
    ie.white_elo_type = (compound >> 5) & 0x07
    ie.black_elo_type = (compound >> 2) & 0x07
    ie.result = compound & 0x03

    # Home pawn data: count from compound field + 8 bytes from offset 48
    hp = bytearray(9)
    hp[0] = hp_count
    hp[1:] = data[48:56]
    ie.home_pawn_data = bytes(hp)

    return ie


def encode_si5_entry(ie: IndexEntry) -> bytes:
    """Encode an IndexEntry into 56 bytes (SCID5 format, little-endian)."""
    buf = bytearray(SCID5_INDEX_ENTRY_SIZE)

    w0 = pack5(encode_count(ie.n_comments), 4, ie.white_id)
    struct.pack_into("<I", buf, 0, w0)

    w1 = pack5(encode_count(ie.n_variations), 4, ie.black_id)
    struct.pack_into("<I", buf, 4, w1)

    w2 = pack5(encode_count(ie.n_nags), 4, ie.event_id)
    struct.pack_into("<I", buf, 8, w2)

    struct.pack_into("<I", buf, 12, ie.site_id)

    w4 = pack5(1 if ie.chess960 else 0, 1, ie.round_id)
    struct.pack_into("<I", buf, 16, w4)

    w5 = pack5(ie.white_elo, 12, ie.date)
    struct.pack_into("<I", buf, 20, w5)

    w6 = pack5(ie.black_elo, 12, ie.event_date)
    struct.pack_into("<I", buf, 24, w6)

    w7 = pack5(ie.num_half_moves, 10, ie.flags)
    struct.pack_into("<I", buf, 28, w7)

    offset_high = (ie.offset >> 32) & 0x7FFF
    offset_low = ie.offset & 0xFFFFFFFF
    w8 = pack5(ie.game_data_size, 17, offset_high)
    struct.pack_into("<I", buf, 32, w8)
    struct.pack_into("<I", buf, 36, offset_low)

    w10 = pack5(ie.stored_line_code, 8, ie.final_mat_sig)
    struct.pack_into("<I", buf, 40, w10)

    hp_count = ie.home_pawn_data[0] if ie.home_pawn_data else 0
    compound = (
        ((hp_count & 0xFF) << 8)
        | ((ie.white_elo_type & 0x07) << 5)
        | ((ie.black_elo_type & 0x07) << 2)
        | (ie.result & 0x03)
    )
    w11 = pack5(compound, 16, ie.eco_code)
    struct.pack_into("<I", buf, 44, w11)

    # Home pawn data bytes 1-8
    hp = ie.home_pawn_data
    for i in range(8):
        buf[48 + i] = hp[1 + i] if 1 + i < len(hp) else 0

    return bytes(buf)


# ======================================================================
# Full index file reading/writing
# ======================================================================


def read_si4(f: BinaryIO) -> tuple[IndexHeader, list[IndexEntry]]:
    """Read a complete SCID4 index file (.si4)."""
    hdr = read_si4_header(f)
    entries = []
    for _ in range(hdr.num_games):
        data = f.read(SCID4_INDEX_ENTRY_SIZE)
        if len(data) < SCID4_INDEX_ENTRY_SIZE:
            break
        entries.append(decode_si4_entry(data))
    return hdr, entries


def write_si4(f: BinaryIO, hdr: IndexHeader, entries: list[IndexEntry]) -> None:
    """Write a complete SCID4 index file (.si4)."""
    hdr.num_games = len(entries)
    write_si4_header(f, hdr)
    for ie in entries:
        f.write(encode_si4_entry(ie))


def read_si5(f: BinaryIO) -> list[IndexEntry]:
    """Read a complete SCID5 index file (.si5)."""
    data = f.read()
    entries = []
    offset = 0
    while offset + SCID5_INDEX_ENTRY_SIZE <= len(data):
        entries.append(decode_si5_entry(data[offset : offset + SCID5_INDEX_ENTRY_SIZE]))
        offset += SCID5_INDEX_ENTRY_SIZE
    return entries


def write_si5(f: BinaryIO, entries: list[IndexEntry]) -> None:
    """Write a complete SCID5 index file (.si5)."""
    for ie in entries:
        f.write(encode_si5_entry(ie))
