"""python-scid: Read, write and search SCID chess databases."""

__all__: list[str] = []
__version__ = "0.1.0"

try:
    from scid.database import ScidDatabase  # noqa: F401
    from scid.search import (  # noqa: F401
        HeaderSearchCriteria,
        MaterialSearchCriteria,
        MaterialSearchResult,
        PositionSearchResult,
        TreeNode,
        TreeResult,
    )
    from scid.quality import (  # noqa: F401
        GameQuality,
        MoveQuality,
        analyze_database_quality,
        analyze_game_quality,
    )

    __all__ += [
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
except ImportError:
    pass
