"""ECO code encoding/decoding (16-bit ecoT)."""

from __future__ import annotations


def eco_encode(s: str) -> int:
    """Encode an ECO code string (e.g. 'B12', 'B12a3') to 16-bit value.

    Returns 0 for empty/invalid input.
    """
    if not s or len(s) < 3:
        return 0
    letter = s[0].upper()
    if letter < "A" or letter > "E":
        return 0
    try:
        d1 = int(s[1])
        d2 = int(s[2])
    except (ValueError, IndexError):
        return 0

    eco = (ord(letter) - ord("A")) * 13100 + d1 * 1310 + d2 * 131

    if len(s) >= 4 and "a" <= s[3] <= "z":
        sub = (ord(s[3]) - ord("a")) * 5
        if len(s) >= 5 and "1" <= s[4] <= "4":
            sub += int(s[4])
        eco += 1 + sub

    return eco + 1


def eco_decode(eco: int) -> str:
    """Decode a 16-bit ECO value to a string. Returns '' for 0 (none)."""
    if eco == 0:
        return ""
    eco -= 1
    basic = eco // 131
    letter = chr(basic // 100 + ord("A"))
    d1 = (basic % 100) // 10
    d2 = basic % 10
    result = f"{letter}{d1}{d2}"

    sub = eco % 131
    if sub > 0:
        sub -= 1
        result += chr(sub // 5 + ord("a"))
        r = sub % 5
        if r > 0:
            result += str(r)

    return result
