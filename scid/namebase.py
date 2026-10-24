"""NameBase: read/write player, event, site, round names for SCID4 and SCID5."""

from __future__ import annotations

from typing import BinaryIO

from scid._filebuf import (
    decode_text,
    encode_text,
    read_uint8,
    read_uint16_be,
    read_uint24_be,
    read_uint32_be,
    read_varint,
    write_uint8,
    write_uint16_be,
    write_uint24_be,
    write_uint32_be,
    write_varint,
)
from scid.constants import (
    NAME_EVENT,
    NAME_INFO,
    NAME_PLAYER,
    NAME_ROUND,
    NAME_SITE,
    SCID4_MAGIC_NAMEBASE,
)

NUM_NAME_TYPES = 4  # PLAYER, EVENT, SITE, ROUND


class NameBase:
    """In-memory name store for SCID databases.

    Stores four lists of strings (player, event, site, round), each indexed
    by integer ID.  IDs are assigned sequentially for SCID5 (append-only)
    or may be non-sequential for SCID4 (explicit IDs in file).
    """

    def __init__(self) -> None:
        # names_[name_type] -> list of (id -> name_string)
        # For SCID4 the list may have gaps (None entries).
        self.names: list[list[str | None]] = [[] for _ in range(NUM_NAME_TYPES)]
        # Reverse lookup: name_type -> {name_string -> id}
        self._index: list[dict[str, int]] = [{} for _ in range(NUM_NAME_TYPES)]
        # SCID5 DB metadata
        self.db_info: dict[str, str] = {}

    def get_name(self, name_type: int, name_id: int) -> str:
        """Look up a name by type and ID. Returns '' for missing entries."""
        names = self.names[name_type]
        if name_id < len(names) and names[name_id] is not None:
            return names[name_id]  # type: ignore[return-value]
        return ""

    def get_id(self, name_type: int, name: str) -> int | None:
        """Look up an ID by name type and string. Returns None if not found."""
        return self._index[name_type].get(name)

    def add_name(self, name_type: int, name: str) -> int:
        """Add a name (or return existing ID). Returns the ID."""
        existing = self._index[name_type].get(name)
        if existing is not None:
            return existing
        name_id = len(self.names[name_type])
        self.names[name_type].append(name)
        self._index[name_type][name] = name_id
        return name_id

    def insert(self, name_type: int, name_id: int, name: str) -> None:
        """Insert a name at a specific ID (used by SCID4 reader)."""
        names = self.names[name_type]
        while len(names) <= name_id:
            names.append(None)
        names[name_id] = name
        self._index[name_type][name] = name_id

    def count(self, name_type: int) -> int:
        """Return number of non-None names for a type."""
        return sum(1 for n in self.names[name_type] if n is not None)

    # ------------------------------------------------------------------
    # SCID4 (.sn4) reading
    # ------------------------------------------------------------------

    @classmethod
    def read_sn4(cls, f: BinaryIO) -> NameBase:
        """Read a SCID4 namebase (.sn4) file."""
        nb = cls()

        # Header
        magic = f.read(8)
        if magic != SCID4_MAGIC_NAMEBASE:
            raise ValueError(f"Bad .sn4 magic: {magic!r}")

        # Obsolete timestamp (4 bytes)
        _timestamp = read_uint32_be(f)

        # Counts per name type (3 bytes each)
        num_names = [read_uint24_be(f) for _ in range(NUM_NAME_TYPES)]

        # Max frequencies per type (3 bytes each, obsolete but needed for decoding)
        max_freq = [read_uint24_be(f) for _ in range(NUM_NAME_TYPES)]

        # Read name records for each type
        for nt in range(NUM_NAME_TYPES):
            _read_sn4_names(f, nb, nt, num_names[nt], max_freq[nt])

        return nb

    # ------------------------------------------------------------------
    # SCID4 (.sn4) writing
    # ------------------------------------------------------------------

    def write_sn4(self, f: BinaryIO, freq: list[list[int]] | None = None) -> None:
        """Write a SCID4 namebase (.sn4) file.

        Args:
            freq: Optional per-type frequency lists (obsolete, written as zeros
                  if not provided).
        """
        f.write(SCID4_MAGIC_NAMEBASE)
        write_uint32_be(f, 0)  # timestamp

        # Collect actual names per type as encoded bytes (SCID4 stores at
        # most 255 bytes per name), sorted bytewise like SCID's unsigned
        # string comparison.
        sorted_names: list[list[tuple[int, bytes]]] = []
        for nt in range(NUM_NAME_TYPES):
            entries = []
            for i, name in enumerate(self.names[nt]):
                if name is not None:
                    entries.append((i, encode_text(name, 255)))
            entries.sort(key=lambda x: x[1])
            sorted_names.append(entries)

        # Write counts
        for nt in range(NUM_NAME_TYPES):
            write_uint24_be(f, len(sorted_names[nt]))

        # Compute max frequencies
        if freq is None:
            freq = [[0] * len(self.names[nt]) for nt in range(NUM_NAME_TYPES)]
        max_freqs = []
        for nt in range(NUM_NAME_TYPES):
            mf = max(freq[nt]) if freq[nt] else 0
            max_freqs.append(mf)
            write_uint24_be(f, mf)

        # Write name records
        for nt in range(NUM_NAME_TYPES):
            _write_sn4_names(
                f, sorted_names[nt], len(sorted_names[nt]), max_freqs[nt], freq[nt]
            )

    # ------------------------------------------------------------------
    # SCID5 (.sn5) reading
    # ------------------------------------------------------------------

    @classmethod
    def read_sn5(cls, f: BinaryIO) -> NameBase:
        """Read a SCID5 namebase (.sn5) file."""
        nb = cls()

        while True:
            val = read_varint(f)
            if val is None:
                break

            name_type = val & 0x07
            length = val >> 3
            data = f.read(length)
            if len(data) < length:
                break

            name = decode_text(data)

            if name_type < NUM_NAME_TYPES:
                # Sequential ID assignment
                nb.names[name_type].append(name)
                nb._index[name_type][name] = len(nb.names[name_type]) - 1
            elif name_type == NAME_INFO:
                # DB metadata: key is prefix, value is rest
                for key in (
                    "type",
                    "description",
                    "autoload",
                    "flag1",
                    "flag2",
                    "flag3",
                    "flag4",
                    "flag5",
                    "flag6",
                ):
                    if name.startswith(key):
                        nb.db_info[key] = name[len(key) :]
                        break

        return nb

    # ------------------------------------------------------------------
    # SCID5 (.sn5) writing
    # ------------------------------------------------------------------

    def write_sn5(self, f: BinaryIO) -> None:
        """Write a SCID5 namebase (.sn5) file."""
        # Write all names in order
        for nt in range(NUM_NAME_TYPES):
            for name in self.names[nt]:
                if name is None:
                    name = ""
                data = encode_text(name)
                val = (len(data) << 3) | nt
                write_varint(f, val)
                f.write(data)

        # Write DB_INFO entries
        for key, value in self.db_info.items():
            data = encode_text(key + value)
            val = (len(data) << 3) | NAME_INFO
            write_varint(f, val)
            f.write(data)


# ======================================================================
# Internal helpers for SCID4 front-coded namebase
# ======================================================================


def _read_sn4_names(
    f: BinaryIO, nb: NameBase, nt: int, num_names: int, max_freq: int
) -> None:
    """Read front-coded name records for one name type from .sn4 file.

    Lengths and shared prefixes count bytes, so decoding happens per name
    after the front coding has been undone.
    """
    prev_name = b""

    # Determine field sizes
    id_size = 3 if num_names >= 65536 else 2

    if max_freq >= 65536:
        freq_size = 3
    elif max_freq >= 256:
        freq_size = 2
    else:
        freq_size = 1

    for i in range(num_names):
        # Read ID
        if id_size == 3:
            name_id = read_uint24_be(f)
        else:
            name_id = read_uint16_be(f)

        # Read frequency (obsolete, discard)
        if freq_size == 3:
            read_uint24_be(f)
        elif freq_size == 2:
            read_uint16_be(f)
        else:
            read_uint8(f)

        # Read name (front-coded)
        length = read_uint8(f)

        if i > 0:
            prefix = read_uint8(f)
        else:
            prefix = 0

        suffix_len = length - prefix
        if suffix_len < 0:
            raise ValueError(f"Corrupt .sn4: prefix {prefix} > length {length}")

        name = prev_name[:prefix] + f.read(suffix_len)
        prev_name = name

        nb.insert(nt, name_id, decode_text(name))


def _write_sn4_names(
    f: BinaryIO,
    sorted_entries: list[tuple[int, bytes]],
    num_names: int,
    max_freq: int,
    freqs: list[int],
) -> None:
    """Write front-coded name records for one name type to .sn4 file."""
    id_size = 3 if num_names >= 65536 else 2

    if max_freq >= 65536:
        freq_size = 3
    elif max_freq >= 256:
        freq_size = 2
    else:
        freq_size = 1

    prev_name = b""

    max_id = (1 << (8 * id_size)) - 1
    for i, (name_id, name) in enumerate(sorted_entries):
        # Write ID.  The ID width depends on the number of names (as in
        # SCID's reader), so an ID beyond it cannot be stored.
        if name_id > max_id:
            raise ValueError(
                f"name ID {name_id} does not fit the {id_size}-byte .sn4 ID "
                f"field for {num_names} names"
            )
        if id_size == 3:
            write_uint24_be(f, name_id)
        else:
            write_uint16_be(f, name_id)

        # Write frequency
        fr = freqs[name_id] if name_id < len(freqs) else 0
        if freq_size == 3:
            write_uint24_be(f, fr)
        elif freq_size == 2:
            write_uint16_be(f, fr)
        else:
            write_uint8(f, fr)

        # Write name (front-coded, already truncated to 255 bytes)
        write_uint8(f, len(name))

        if i > 0:
            prefix = _str_prefix(name, prev_name)
            write_uint8(f, prefix)
        else:
            prefix = 0

        f.write(name[prefix:])

        prev_name = name


def _str_prefix(s1: bytes, s2: bytes) -> int:
    """Return the length of the common prefix of two byte strings."""
    n = min(len(s1), len(s2))
    for i in range(n):
        if s1[i] != s2[i]:
            return i
    return n
