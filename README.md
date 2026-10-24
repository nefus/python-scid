# python-scid

Pure Python library for reading, writing, and searching
[SCID](http://scid.sourceforge.net/) chess databases.
Works with [python-chess](https://python-chess.readthedocs.io/) — games are
returned as `chess.pgn.Game` objects, positions as `chess.Board`.

Supports both **SCID4** (`.si4`/`.sn4`/`.sg4`) and **SCID5**
(`.si5`/`.sn5`/`.sg5`) formats.

## Installation

```
pip install python-scid
```

Requires Python 3.10+ and python-chess >= 1.10.

## Quick start

```python
from scid import ScidDatabase

with ScidDatabase.open("/path/to/mydb") as db:
    print(f"{len(db)} games")

    # Read a game (returns chess.pgn.Game)
    game = db[0]
    print(game.headers["White"], "vs", game.headers["Black"])
    print(game.mainline_moves())

    # Fast header-only access (no move decoding)
    h = db.headers(0)
    print(h["White"], h["WhiteElo"])
```

## Opening a database

Pass the base path **without** the file extension. The format (SCID4 or
SCID5) is auto-detected from which files exist on disk.

```python
db = ScidDatabase.open("~/games/chess/mydb")         # read-only
db = ScidDatabase.open("~/games/chess/mydb", "rw")   # read-write
```

## Creating a database

```python
db = ScidDatabase.create("/tmp/newdb", version=5)   # SCID5 (default)
db = ScidDatabase.create("/tmp/newdb", version=4)   # SCID4
```

## Reading games

`ScidDatabase` supports `len()`, indexing, slicing, and iteration.
Each game is returned as a `chess.pgn.Game`.

```python
with ScidDatabase.open("mydb") as db:
    # Single game
    game = db[0]

    # Slice
    first_ten = db[:10]

    # Iterate all games
    for game in db:
        board = game.end().board()
        print(board)
```

### Fast header access

`db.headers(i)` returns a `dict[str, str]` with the Seven Tag Roster
plus Elo, ECO, and EventDate — without decoding the move data. This is
roughly 300x faster than `db[i].headers`.

```python
for i in range(len(db)):
    h = db.headers(i)
    if int(h.get("WhiteElo", "0")) > 2700:
        print(h["White"], h["WhiteElo"])
```

## Writing games

```python
import chess.pgn

db = ScidDatabase.create("/tmp/mydb")

game = chess.pgn.Game()
game.headers["White"] = "Carlsen"
game.headers["Black"] = "Caruana"
game.headers["Result"] = "1-0"
node = game.add_variation(chess.Move.from_uci("e2e4"))
node = node.add_variation(chess.Move.from_uci("e7e5"))

idx = db.add_game(game)           # returns the new game index
db.replace_game(idx, game)        # overwrite a game
db.delete_game(idx)               # mark as deleted
db.compact()                      # remove deleted games and reclaim space
db.save()                         # write to disk
```

Opened with `mode="rw"` as a context manager, the database is saved on a
normal exit of the `with` block, but not when it is left by an exception.

## PGN import / export

```python
# Import PGN into a new SCID database (UTF-8 by default)
db = ScidDatabase.from_pgn("games.pgn", "/tmp/imported", version=5)
db = ScidDatabase.from_pgn("legacy.pgn", "/tmp/legacy", encoding="latin-1")

# Export a SCID database to PGN
with ScidDatabase.open("mydb") as db:
    db.to_pgn("output.pgn")
```

## Searching

### Header search

Filter games by player names, dates, Elo, ECO codes, results, and more.
Operates on the index only — no game decoding required.

```python
from scid import ScidDatabase, HeaderSearchCriteria

with ScidDatabase.open("mydb") as db:
    # Keyword arguments (simple)
    hits = db.search_header(white="Kasparov", date_min="1990.01.01")

    # Criteria object (full control)
    c = HeaderSearchCriteria(
        player="Carlsen",            # either color
        eco_min="B90", eco_max="B99",
        white_elo_min=2700,
        results={"1-0", "0-1"},      # exclude draws
    )
    hits = db.search_header(c)

    # hits is a list of game indices
    for i in hits:
        print(db.headers(i)["White"], "vs", db.headers(i)["Black"])
```

**Available criteria** (all optional, combined with AND):

| Field | Type | Description |
|-------|------|-------------|
| `white` | `str` | White player name (substring, case-insensitive) |
| `black` | `str` | Black player name |
| `player` | `str` | Either player |
| `event` | `str` | Event name |
| `site` | `str` | Site name |
| `round` | `str` | Round |
| `date_min`, `date_max` | `str` | Date range (`"YYYY.MM.DD"`) |
| `event_date_min`, `event_date_max` | `str` | Event date range |
| `eco_min`, `eco_max` | `str` | ECO code range (`"B00"` to `"B99"`) |
| `white_elo_min`, `white_elo_max` | `int` | White Elo range |
| `black_elo_min`, `black_elo_max` | `int` | Black Elo range |
| `elo_min`, `elo_max` | `int` | Either player's Elo |
| `length_min`, `length_max` | `int` | Game length in half-moves |
| `results` | `set[str]` | Allowed results (`"1-0"`, `"0-1"`, `"1/2-1/2"`, `"*"`) |
| `game_num_min`, `game_num_max` | `int` | Game index range (0-based) |

Quoting a name (`'"Carlsen, Magnus"'`) forces an exact match instead of
substring.

### Position search

Find all games that pass through a given board position. Uses SCID's
three-level pre-filtering (stored lines, home pawn signature, material
signature) before falling back to full game replay.

```python
import chess

with ScidDatabase.open("mydb") as db:
    # Search for the Italian Game position (1.e4 e5 2.Nf3 Nc6 3.Bc4)
    board = chess.Board()
    for move in ["e4", "e5", "Nf3", "Nc6", "Bc4"]:
        board.push_san(move)

    results = db.search_position(board)
    for r in results:
        print(f"Game {r.game_index} reaches position at ply {r.ply}")
```

### Material search

Find games where piece counts fall within specified ranges at any point.

```python
from scid import MaterialSearchCriteria

with ScidDatabase.open("mydb") as db:
    # Rook endgames: 1 rook each, no queens, no minor pieces, no pawns
    c = MaterialSearchCriteria(
        wq=(0, 0), wr=(1, 1), wb=(0, 0), wn=(0, 0), wp=(0, 0),
        bq=(0, 0), br=(1, 1), bb=(0, 0), bn=(0, 0), bp=(0, 0),
    )
    results = db.search_material(c)
    for r in results:
        print(f"Game {r.game_index} at ply {r.ply}")
```

### Tree search (opening statistics)

Compute an opening tree from a position — aggregating move frequencies,
win/draw/loss counts, average Elo, FIDE performance rating, and more
across all games in the database.

```python
import chess

with ScidDatabase.open("mydb") as db:
    # Opening tree from the starting position
    tree = db.search_tree()

    print(f"{tree.games_matched} games")
    for node in tree.nodes:
        print(
            f"  {node.san:6s}  {node.count:6d}  "
            f"score={node.score:.1%}  "
            f"Elo={node.avg_elo:.0f}  "
            f"draw={node.draw_pct:.0f}%"
        )

    # Tree from a specific position
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    tree = db.search_tree(board)
```

**`TreeNode` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `move` | `chess.Move` | The move played |
| `san` | `str` | Standard algebraic notation |
| `count` | `int` | Number of games |
| `white_wins` | `int` | Games won by White |
| `draws` | `int` | Drawn games |
| `black_wins` | `int` | Games won by Black |
| `score` | `float` | Score from White's perspective (0.0–1.0) |
| `avg_elo` | `float` | Average Elo of the side making the move |
| `performance` | `float` | FIDE performance rating (0 if < 10 rated games) |
| `avg_year` | `float` | Average game year |
| `draw_pct` | `float` | Draw percentage (0–100) |

**`TreeResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `position` | `chess.Board` | The searched position |
| `nodes` | `list[TreeNode]` | Per-move stats, sorted by frequency |
| `total` | `TreeNode \| None` | Aggregated totals across all moves |
| `games_searched` | `int` | Total games considered |
| `games_matched` | `int` | Games containing the position |

### Game quality (Stockfish)

`analyze_game_quality()` compares each main-line move with Stockfish's best
move and returns a provisional 0–100 move-quality index. It requires a
separately installed UCI engine; the index is not a player rating and its
scaling has not yet been calibrated against a representative game set.

```python
import chess.engine
from scid import ScidDatabase, analyze_game_quality

with chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish") as engine:
    with ScidDatabase.open("mydb") as db:
        quality = analyze_game_quality(
            db[0], engine, limit=chess.engine.Limit(time=0.1)
        )

        print(quality.index)  # None for a game without moves
        for move in quality.moves:
            print(move.ply, move.played_san, move.best_san,
                  move.expected_score_loss)
```

The default search limit is depth 12. ``GameQuality`` also exposes the mean
expected-score loss and number of analyzed moves. Each move's loss is
calculated in expected points from the mover's perspective.

For batch analysis, ``analyze_database_quality()`` starts one Stockfish
process per worker and returns results keyed by game index. By default it
uses half the logical CPU count, with one Stockfish thread per worker. The
database must remain open and unmodified until analysis finishes. Set
``pgn_path`` to write an annotated PGN checkpoint. It stores tags such as
``ScidQualityIndex`` and per-move details; rerunning with the same database,
engine, and search limit resumes completed games. The SCID database itself is
left unchanged.

```python
from pathlib import Path
from scid import ScidDatabase, analyze_database_quality

with ScidDatabase.open(Path.home() / "games/chess/twic/scid/twic") as db:
    qualities = analyze_database_quality(
        db,
        engine_command="/usr/local/bin/stockfish",
        game_indices=range(1000),
        workers=8,
        engine_threads=1,
        pgn_path=Path.home() / "games/chess/twic/scid/twic-quality.pgn",
        progress_callback=lambda done, total: (
            print(f"{done}/{total}")
            if done % 100 == 0 or done == total
            else None
        ),
    )

print(qualities[0].index)
```

## Chaining searches

Search results (game index lists) can be passed between searches to
narrow results progressively:

```python
with ScidDatabase.open("mydb") as db:
    # Find Kasparov's games as White with the Sicilian
    hits = db.search_header(white="Kasparov", eco_min="B20", eco_max="B99")

    # Of those, find games reaching a specific position
    board = chess.Board()
    for m in ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3"]:
        board.push_san(m)
    pos_hits = db.search_position(board, game_indices=hits)

    # Opening tree over just those games
    tree = db.search_tree(board, game_indices=hits)
```

## Performance

Benchmarks on a 193K-game FICS database (Intel i7):

| Operation | Time | Rate |
|-----------|------|------|
| Open database | 0.6s | — |
| Read all headers | 0.4s | 521K headers/sec |
| Header search (all games) | 0.035s | — |
| Position search (Italian Game) | 18.5s | 84% pre-filtered |
| Tree search (starting position) | 5.3s | — |
| Full game decode | — | ~1,500 games/sec |

## Format support

| Feature | SCID4 | SCID5 |
|---------|-------|-------|
| Read | Yes | Yes |
| Write | Yes | Yes |
| Index | 47-byte big-endian | 56-byte little-endian |
| Namebase | Front-coded, sorted | Varint, append-only |
| Game data | Identical binary format | Identical binary format |

## License

MIT
