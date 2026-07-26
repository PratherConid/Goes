"""Read self-play trajectory files (ai/checkpoints/**/*_traj.json) and print a recorded game."""

import argparse
import json
import re
from pathlib import Path

# Matches goes_train's per-game self-play log line, e.g.:
#   "  game 40/64  plies=354  stones=[21,109,]  territories=[0,0,]  winners=[2,]"
# (see train.cpp's self-play loop) - stones/territories are stone-type-indexed lists with a
# trailing comma after every element (including the last), so the captured groups are stripped of
# empty entries by _parse_int_list below rather than split naively.
_GAME_LINE_RE = re.compile(
    r"game\s+\d+/\d+\s+plies=(\d+)\s+stones=\[([\d,]*)\]\s+territories=\[([\d,]*)\]\s+winners=\[([\d,]*)\]"
)

def _parse_int_list(s):
    return [int(x) for x in s.split(",") if x]

def parse_game_lines(text):
    """Parse goes_train's stdout for per-game self-play summary lines into a list of
    {"plies": int, "stones": [int, ...], "territories": [int, ...], "winners": [int, ...]} dicts,
    one per matching line, in the order they appear. `text` may be the full captured stdout (a
    string) or an iterable of lines; non-matching lines are skipped."""
    lines = text.splitlines() if isinstance(text, str) else text
    games = []
    for line in lines:
        m = _GAME_LINE_RE.search(line)
        if not m:
            continue
        games.append({
            "plies": int(m.group(1)),
            "stones": _parse_int_list(m.group(2)),
            "territories": _parse_int_list(m.group(3)),
            "winners": _parse_int_list(m.group(4)),
        })
    return games

def running_average(games, ws):
    """Compute the trailing running average (window size `ws`) of each game's "stones"/
    "territories" lists (as returned by parse_game_lines) across the sequence, returning a list of
    {"stones": [float, ...], "territories": [float, ...]} dicts, one per entry in `games`, in the
    same order. Entry i's window covers games[max(0, i-ws+1):i+1] - a shrinking window for the
    first ws-1 entries, rather than requiring a full window before any output is produced.

    Runs in O(len(games)) total: maintains running per-stone-type sums and adds/subtracts one
    game's worth of values as the window slides, rather than re-summing the whole window at every
    step. `games` is already a materialized list, so the element leaving the window (games[i-ws])
    is looked up by direct indexing - no separate queue needed to remember it."""
    result = []
    stone_sums = None
    territory_sums = None
    for i, g in enumerate(games):
        if stone_sums is None:
            stone_sums = [0.0] * len(g["stones"])
            territory_sums = [0.0] * len(g["territories"])
        stone_sums = [a + b for a, b in zip(stone_sums, g["stones"])]
        territory_sums = [a + b for a, b in zip(territory_sums, g["territories"])]
        if i >= ws:
            old = games[i - ws]
            stone_sums = [a - b for a, b in zip(stone_sums, old["stones"])]
            territory_sums = [a - b for a, b in zip(territory_sums, old["territories"])]
        n = min(i + 1, ws)
        result.append({
            "stones": [s / n for s in stone_sums],
            "territories": [t / n for t in territory_sums],
        })
    return result

def print_game(path, idx):
    with open(path, "r") as f:
        content = json.load(f)
    if not isinstance(content, list):
        print("print_game :: content of trajectory file is not a list")
    if idx >= len(content):
        print(f"print_game :: Index {idx} out of bound {len(content)}")
    print(json.dumps(content[idx]))

def entry_print_game():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", default="ai/checkpoints",
        help="Path to a *_traj.json file, or a directory to search for one "
             "(defaults to the most recently modified trajectory file found "
             "under it). Default: ai/checkpoints")
    parser.add_argument("game_index", type=int, help="Index of the game to print")
    args = parser.parse_args()

    path = Path(args.path)
    print_game(path, args.game_index)

def entry_plot_running_avg():
    parser = argparse.ArgumentParser(
        description="Plot the running average of per-game stones/territories from a captured "
                     "goes_train stdout log.")
    parser.add_argument("filename", help="Path to a text file containing goes_train's captured stdout")
    parser.add_argument("--window-size", "-w", type=int, default=10,
                         help="Running average window size (default: 10)")
    parser.add_argument("--output", "-o", default=None,
                         help="Save the plot to this path instead of showing it interactively")
    args = parser.parse_args()

    with open(args.filename, "r") as f:
        text = f.read()
    games = parse_game_lines(text)
    if not games:
        print(f"entry_plot_running_avg :: no matching game lines found in {args.filename}")
        return
    avgs = running_average(games, args.window_size)

    # Imported here rather than at module level so parse_game_lines/running_average/print_game
    # stay usable without matplotlib installed - only this plotting entry point needs it.
    import matplotlib.pyplot as plt

    x = list(range(len(avgs)))
    fig, ax = plt.subplots()
    for k in range(len(avgs[0]["stones"])):
        ax.plot(x, [a["stones"][k] for a in avgs], label=f"stones[{k}]")
    for k in range(len(avgs[0]["territories"])):
        ax.plot(x, [a["territories"][k] for a in avgs], label=f"territories[{k}]", linestyle="--")

    ax.set_xlabel("game index")
    ax.set_ylabel(f"running average (window={args.window_size})")
    ax.set_title(f"Running average of stones/territories - {Path(args.filename).name}")
    ax.legend()

    if args.output:
        fig.savefig(args.output)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    entry_plot_running_avg()
