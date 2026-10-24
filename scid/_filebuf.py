"""Low-level byte I/O helpers for reading/writing SCID binary formats."""

from __future__ import annotations

import struct
from typing import BinaryIO

# ---------------------------------------------------------------------------
# Big-endian readers (SCID4)
# ---------------------------------------------------------------------------


def read_uint8(f: BinaryIO) -> int:
    return f.read(1)[0]


def read_uint16_be(f: BinaryIO) -> int:
    return struct.unpack(">H", f.read(2))[0]


def read_uint24_be(f: BinaryIO) -> int:
    b = f.read(3)
    return (b[0] << 16) | (b[1] << 8) | b[2]


def read_uint32_be(f: BinaryIO) -> int:
    return struct.unpack(">I", f.read(4))[0]


# ---------------------------------------------------------------------------
# Little-endian readers (SCID5)
# ---------------------------------------------------------------------------


def read_uint16_le(f: BinaryIO) -> int:
    return struct.unpack("<H", f.read(2))[0]


def read_uint32_le(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]


def decode_uint32_le(data: bytes, offset: int = 0) -> int:
    return struct.unpack_from("<I", data, offset)[0]


# ---------------------------------------------------------------------------
# Big-endian writers (SCID4)
# ---------------------------------------------------------------------------


def write_uint8(f: BinaryIO, val: int) -> None:
    f.write(bytes([val & 0xFF]))


def write_uint16_be(f: BinaryIO, val: int) -> None:
    f.write(struct.pack(">H", val & 0xFFFF))


def write_uint24_be(f: BinaryIO, val: int) -> None:
    f.write(bytes([(val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF]))


def write_uint32_be(f: BinaryIO, val: int) -> None:
    f.write(struct.pack(">I", val & 0xFFFFFFFF))


# ---------------------------------------------------------------------------
# Little-endian writers (SCID5)
# ---------------------------------------------------------------------------


def write_uint16_le(f: BinaryIO, val: int) -> None:
    f.write(struct.pack("<H", val & 0xFFFF))


def write_uint32_le(f: BinaryIO, val: int) -> None:
    f.write(struct.pack("<I", val & 0xFFFFFFFF))


def encode_uint32_le(val: int) -> bytes:
    return struct.pack("<I", val & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Varint (LEB128) — used by SCID5 namebase
# ---------------------------------------------------------------------------


def read_varint(f: BinaryIO) -> int | None:
    """Read a LEB128 varint. Returns None on EOF."""
    result = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            return None if shift == 0 else result
        byte = b[0]
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 128:
            break
    return result


def write_varint(f: BinaryIO, val: int) -> None:
    """Write a LEB128 varint."""
    while val >= 128:
        f.write(bytes([(val & 0x7F) | 0x80]))
        val >>= 7
    f.write(bytes([val & 0x7F]))


def encode_varint(val: int) -> bytes:
    """Encode a LEB128 varint to bytes."""
    buf = bytearray()
    while val >= 128:
        buf.append((val & 0x7F) | 0x80)
        val >>= 7
    buf.append(val & 0x7F)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------


def decode_text(data: bytes | bytearray | memoryview) -> str:
    """Decode a stored string: UTF-8 (current SCID), else latin-1 (legacy DBs)."""
    raw = bytes(data)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def encode_text(text: str, max_len: int | None = None) -> bytes:
    """Encode a string as UTF-8, truncated to *max_len* bytes on a character boundary."""
    raw = text.encode("utf-8")
    if max_len is not None and len(raw) > max_len:
        raw = raw[:max_len]
        while raw:
            try:
                raw.decode("utf-8")
                break
            except UnicodeDecodeError:
                raw = raw[:-1]
    return raw


# ---------------------------------------------------------------------------
# Null-terminated string reading
# ---------------------------------------------------------------------------


def read_terminated_string(data: bytes | memoryview, offset: int) -> tuple[str, int]:
    """Read a null-terminated string from data starting at offset.

    Returns (string, new_offset_past_null).
    """
    end = (
        data.index(0, offset)
        if isinstance(data, (bytes, bytearray))
        else _find_null(data, offset)
    )
    s = decode_text(data[offset:end])
    return s, end + 1


def _find_null(data: memoryview, offset: int) -> int:
    for i in range(offset, len(data)):
        if data[i] == 0:
            return i
    return len(data)


# ---------------------------------------------------------------------------
# SCID5 pack/unpack helpers
# ---------------------------------------------------------------------------


def pack5(high_val: int, high_bits: int, low_val: int) -> int:
    """Pack two values into a 32-bit word (SCID5 style)."""
    return (high_val << (32 - high_bits)) | low_val


def unpack5(word: int, high_bits: int) -> tuple[int, int]:
    """Unpack a 32-bit word into (high_val, low_val) (SCID5 style)."""
    low_bits = 32 - high_bits
    high_val = (word >> low_bits) & ((1 << high_bits) - 1)
    low_val = word & ((1 << low_bits) - 1)
    return high_val, low_val
