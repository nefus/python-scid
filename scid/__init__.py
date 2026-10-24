"""python-scid: Read, write and search SCID chess databases."""

__version__ = "0.1.0"

from scid.database import ScidDatabase
from scid.search import (
    HeaderSearchCriteria,
    MaterialSearchCriteria,
    MaterialSearchResult,
    PositionSearchResult,
    TreeNode,
    TreeResult,
)
from scid.quality import (
    GameQuality,
    MoveQuality,
    analyze_database_quality,
    analyze_game_quality,
)

__all__ = [
    "ScidDatabase",
    "HeaderSearchCriteria",
    "MaterialSearchCriteria",
    "MaterialSearchResult",
    "PositionSearchResult",
    "TreeNode",
    "TreeResult",
    "GameQuality",
    "MoveQuality",
    "analyze_database_quality",
    "analyze_game_quality",
]
