"""Shared fixtures for python-scid tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Path to test databases
GAMES_DIR = Path.home() / "games" / "chess"


def _db_exists(name: str, version: int) -> bool:
    ext = f".si{version}"
    return (GAMES_DIR / f"{name}{ext}").exists()


# --- SCID4 databases ---


@pytest.fixture
def db_666_path() -> Path:
    """Path to 666.si4 (12 games, small)."""
    p = GAMES_DIR / "666"
    if not _db_exists("666", 4):
        pytest.skip("666 test database not found")
    return p


@pytest.fixture
def db_scandi_path() -> Path:
    p = GAMES_DIR / "scandi"
    if not _db_exists("scandi", 4):
        pytest.skip("scandi test database not found")
    return p


@pytest.fixture
def db_repertoire_path() -> Path:
    p = GAMES_DIR / "repertoire"
    if not _db_exists("repertoire", 4):
        pytest.skip("repertoire test database not found")
    return p


@pytest.fixture
def db_koblenz_path() -> Path:
    p = GAMES_DIR / "koblenz lehrbuch der schachtaktik"
    if not _db_exists("koblenz lehrbuch der schachtaktik", 4):
        pytest.skip("koblenz test database not found")
    return p


@pytest.fixture
def db_freechess_path() -> Path:
    p = GAMES_DIR / "freechess"
    if not _db_exists("freechess", 4):
        pytest.skip("freechess test database not found")
    return p


# --- SCID5 databases ---


@pytest.fixture
def db_kiop23_path() -> Path:
    p = GAMES_DIR / "kiop23"
    if not _db_exists("kiop23", 5):
        pytest.skip("kiop23 test database not found")
    return p


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Temporary path for creating test databases."""
    return tmp_path / "testdb"
