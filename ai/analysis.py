"""Read self-play trajectory files (ai/checkpoints/**/*_traj.json) and print a recorded game."""

import argparse
import json
from pathlib import Path

def print_game(path, idx):
    with open(path, "r") as f:
        content = json.load(f)
    if not isinstance(content, list):
        print("print_game :: content of trajectory file is not a list")
    if idx >= len(content):
        print(f"print_game :: Index {idx} out of bound {len(content)}")
    print(json.dumps(content[idx]))

def main():
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

if __name__ == "__main__":
    main()
