"""20-bit date encoding used by SCID index entries."""

from __future__ import annotations


def encode_date(year: int = 0, month: int = 0, day: int = 0) -> int:
    """Encode a date into 20 bits: year(11)|month(4)|day(5).

    A value of 0 for any component means "unknown".
    """
    return ((year & 0x7FF) << 9) | ((month & 0x0F) << 5) | (day & 0x1F)


def decode_date(raw: int) -> tuple[int, int, int]:
    """Decode a 20-bit date into (year, month, day)."""
    year = (raw >> 9) & 0x7FF
    month = (raw >> 5) & 0x0F
    day = raw & 0x1F
    return year, month, day


def date_to_pgn(raw: int) -> str:
    """Convert a 20-bit date to PGN date string like '2023.07.15'."""
    year, month, day = decode_date(raw)
    y = f"{year:04d}" if year else "????"
    m = f"{month:02d}" if month else "??"
    d = f"{day:02d}" if day else "??"
    return f"{y}.{m}.{d}"


def date_from_pgn(s: str) -> int:
    """Parse a PGN date string like '2023.07.15' into 20-bit encoding."""
    parts = s.split(".")
    try:
        year = int(parts[0]) if parts[0] != "????" else 0
    except (ValueError, IndexError):
        year = 0
    try:
        month = int(parts[1]) if len(parts) > 1 and parts[1] != "??" else 0
    except (ValueError, IndexError):
        month = 0
    try:
        day = int(parts[2]) if len(parts) > 2 and parts[2] != "??" else 0
    except (ValueError, IndexError):
        day = 0
    return encode_date(year, month, day)


# ---------------------------------------------------------------------------
# SCID4 EventDate delta encoding (12 bits relative to game date)
# ---------------------------------------------------------------------------


def encode_event_date_delta(game_date: int, event_date: int) -> int:
    """Encode event date as a 12-bit delta from game date (SCID4 only).

    The delta encodes year offset relative to game year, month, and day.
    A year field of 0 means "no event date".
    """
    if event_date == 0:
        return 0
    gy, gm, gd = decode_date(game_date)
    ey, em, ed = decode_date(event_date)
    eyear_enc = (ey + 4 - gy) & 0x07
    if eyear_enc == 0:
        return 0  # Cannot encode, would mean "no event date"
    # 12 bits: yearEnc (3 bits) | month (4 bits) | day (5 bits)
    return (eyear_enc << 9) | (em << 5) | ed


def decode_event_date_delta(game_date: int, delta: int) -> int:
    """Decode a 12-bit event date delta relative to game date (SCID4 only)."""
    eyear_enc = (delta >> 9) & 0x07
    if eyear_enc == 0:
        return 0
    gy, _, _ = decode_date(game_date)
    ey = eyear_enc + gy - 4
    if ey < 0:
        ey = 0
    month = (delta >> 5) & 0x0F
    day = delta & 0x1F
    return encode_date(ey, month, day)
