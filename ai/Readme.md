## AI Training Pipeline

Self-play training pipeline for Goes using Monte Carlo Tree Search (MCTS), in the style of AlphaZero.

Three model architectures are used depending on board topology:
- **CNN**: a plain residual conv stack (no pooling) for boards with a 2D integer embedding — `rect`, `rectd`, `tri`, `twsq`, `gtsq`. The default for these boards.
- **UNet**: an alternative for the same board types (U-Net encoder/decoder with pooling), selectable via `--net-arch unet`.
- **MessagePassingGNN**: for higher-dimensional boards whose nodes cannot be laid out on a 2D grid — `cub`, `hcub`.

## Differences from the TypeScript engine

`shared/boardConfig.ts` stores node positions (`pos`) and a bounding box (`boardDimension`) for the canvas renderer. The C++ `BoardConfig` now includes `pos` (ported from the TypeScript factory functions) but omits `boardDimension`, which is only needed by the renderer.

`BoardState` otherwise has full feature parity with `shared/boardState.ts`: multi-stone-per-turn offering (`TurnInfo.stones`), protected/friendly stones, per-player and global stone placement limits, `maxPlies`, resignation (including `withdraw_move()` re-stamping a resignation-caused game-over onto the new last move), `komi` (per-player scoring handicap, applied in `compute_winners()` and folded into both MCTS reward formulas — see **Reward**, below), and `koRule` (both `positional` and `situational` superko are implemented — `HistoryManager`'s `(ply_mod, board)` key collapses `ply_mod` to a constant under `positional`, since two boards are then "the same" purely by content, mirroring `compareState()` skipping the ply-mod comparison in `shared/boardState.ts`). `stoneToPlayerMap` is stone → *set* of players (each mapped player gets a stone's full point value, not split), matching the TypeScript side.

Not ported (no C++ consumer needs them): `GameConfig.players`/`FinishedGame`-style replay reconstruction — self-play never replays a finished game, and the inference server replays a live move list instead.

## Directory Structure

```
ai/
├── CMakeLists.txt
├── third_party/
│   ├── httplib.h                       single-header HTTP server (cpp-httplib)
│   └── nlohmann/json.hpp               single-header JSON library
├── include/ and src/                   (headers in include/, implementations in src/)
│   ├── game/
│   │   ├── board_config.{h,cpp}        Adjacency-only port of shared/boardConfig.ts
│   │   └── board_state.{h,cpp}         Port of shared/boardState.ts
│   ├── model/
│   │   ├── features.{h,cpp}            BoardState → per-node feature tensor
│   │   ├── gnn.{h,cpp}                 MessagePassingGNN (policy + ownership heads, for cub/hcub)
│   │   ├── unet.{h,cpp}                U-Net (policy + ownership heads, for 2D boards)
│   │   ├── cnn.{h,cpp}                 Plain residual CNN (policy + ownership heads, for 2D boards)
│   │   ├── evaluator.h                 Type-erased Evaluator used by MCTS and self-play
│   │   └── any_model.h                 AnyModel variant + make_evaluator() factory
│   ├── mcts/
│   │   └── mcts.{h,cpp}                AlphaZero-style MCTS
│   ├── training/
│   │   ├── self_play.{h,cpp}           Generate training records from self-play games
│   │   └── replay_buffer.{h,cpp}       Circular buffer of game records
│   └── util/
│       └── sha256.{h,cpp}              Checkpoint-directory hashing (see Checkpoint Directories and Matching)
├── src/
│   ├── train.cpp                       Main training entry point → binary: goes_train
│   └── server.cpp                      HTTP inference server: POST /move → binary: goes_server
└── checkpoints/                        Saved model weights (created at runtime)
```

## Prerequisites

1. **MSVC 2022** (or GCC/Clang on Linux) with C++17 support
2. **CMake ≥ 3.18**
3. **OpenMP** — bundled with MSVC and GCC; on macOS install via `brew install libomp`. Optional but recommended: enables parallel MCTS select across game trees.
4. **LibTorch** — download the pre-built package from https://pytorch.org/get-started/locally/
   - Select: your OS (IMPORTANT!), C++ / LibTorch, CUDA version or CPU
   - Extract to a local directory, e.g. `D:\libtorch` on windows and `~/include/libtorch` on linux
5. **Third-party headers** (single-file, copy into `third_party/`):
   - [httplib.h](https://github.com/yhirose/cpp-httplib/releases) → `third_party/httplib.h`
   - [json.hpp](https://github.com/nlohmann/json/releases) → `third_party/nlohmann/json.hpp`

## Build

**Note:** Make sure that the absolute paths do not contain any unicode characters

**Linux:**

Install CUDA runtime libraries required by LibTorch (Ubuntu 22.04):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# Replace "cuda-12" with your major CUDA version (check with: nvcc --version).
# For CUDA 11.x use "cuda-11"; for CUDA 13.x use "cuda-13"; package names for other versions may differ.
sudo apt-get install libcudnn9-cuda-12 libnccl2 libnccl-dev libcusparselt0 libcusparselt-dev libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
```

Then build

```bash
cmake -S ai -B ai/build \
      -DCMAKE_PREFIX_PATH="~/include/libtorch" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build ai/build --config Release
```

**Windows:**
```powershell
cmake -S ai -B ai\build -DCMAKE_PREFIX_PATH="D:/libtorch" -DCMAKE_BUILD_TYPE=Release
cmake --build ai\build --config Release
```

**Note:**
1. The build adds `-Wl,--disable-new-dtags` on Linux so that `RPATH` (rather than `RUNPATH`) is embedded in the binaries. This is necessary because `libcupti` is a transitive dependency and `RUNPATH` is not searched transitively by the dynamic linker.
2. On Windows with Visual Studio generator, DLLs are automatically copied next to the binaries by the CMakeLists.

## Training

Run from the project root:

**Linux:**

```bash
export OMP_NUM_THREADS=8
ai/build/goes_train --game-config public/game_presets/9x9_go.json --verbosity 1 --linear-move-bound 1.5 2.5
```

For higher-end training devices,

```bash
export OMP_NUM_THREADS=8

ai/build/goes_train --game-config public/game_presets/9x9_go.json --verbosity 1 --linear-move-bound 1.5 2.5 --gamegen-batch-size 256 --iterations 65536 --save-every 8 --self-play-games 64 --num-simulations 512 --buffer-size 8192 --cnn-hidden-dim 128

ai/build/goes_train --game-config public/game_presets/9x9_go.json --verbosity 1 --linear-move-bound 1.5 2.5 --gamegen-batch-size 256 --iterations 65536 --save-every 8 --self-play-games 64 --num-simulations 512 --buffer-size 8192 --net-arch gnn --gnn-hidden-dim 256

ai/build/goes_train --game-config public/game_presets/7x7x2_twsq_go.json --verbosity 1 --linear-move-bound 1.5 2.5 --gamegen-batch-size 256 --iterations 65536 --save-every 8 --self-play-games 64 --num-simulations 512 --buffer-size 8192 --cnn-hidden-dim 128

ai/build/goes_train --game-config public/game_presets/13x13_two_ply_go.json --verbosity 1 --linear-move-bound 1.5 2.5 --gamegen-batch-size 256 --iterations 65536 --save-every 8 --self-play-games 64 --num-simulations 512 --buffer-size 16384 --cnn-hidden-dim 128
```

**Windows:**

```powershell
$env:OMP_NUM_THREADS = "8"
ai\build\Release\goes_train --game-config public/game_presets/9x9_go.json --verbosity 1 --linear-move-bound 1.5 2.5
```

Checkpoints are saved as `ai/checkpoints/<hash>/<arch>_XXXXXX.pt`, where `<hash>` is a fresh
64-character SHA-256 of the model config, game config, and a timestamp - a new, unique directory
every run, never reused implicitly. To continue a specific run later, pass `--resume <hash>` (the
hash a run created is printed to stdout, and is also just the directory name under
`ai/checkpoints/`) - this validates that the current `--game-config`/architecture flags match the
resumed directory's saved config exactly before loading its latest weights and replay-buffer
history. See **Checkpoint Directories and Matching**, below, for the full scheme.

## Profiling Training (Linux)

Build with `RelWithDebInfo` to keep optimizations while embedding debug symbols for readable stack traces:

```bash
cmake -S ai -B ai/build \
      -DCMAKE_PREFIX_PATH="~/include/libtorch" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build ai/build --config RelWithDebInfo
```

**`perf` (low overhead, recommended):**

```bash
# May need a while to collect data after the program finishes
perf record -g --call-graph dwarf \
    ai/build/goes_train --game-config public/game_presets/9x9_go.json --verbosity 2 --linear-move-bound 0.5 1.5 --iterations 3 --self-play-games 10 --gamegen-batch-size 5 --num-simulations 10

perf report --stdio | head -200   # flat report to stdout
perf report                        # interactive TUI with drill-down
```

If `perf` refuses to run due to permissions:

```bash
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

**`callgrind` (no kernel permissions needed, ~20× slower):**

```bash
valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
    ai/build/goes_train --game-config public/game_presets/9x9_go.json --verbosity 2 --linear-move-bound 0.5 1.5 --iterations 3 --self-play-games 4 --gamegen-batch-size 2 --num-simulations 5

callgrind_annotate --auto=yes callgrind.out | less
```

Callgrind reports exact instruction counts per source line, which is useful for pinpointing hot paths inside tight loops.

**Limiting run time:** pass `--iterations N` to cap the run, or wrap the binary with `timeout`:

```bash
timeout 120s perf record -g --call-graph dwarf ai/build/goes_train --game-config public/game_presets/9x9_go.json --iterations 999
```

## Inference Server

**Linux**

```bash
ai/build/goes_server --checkpoint-dir ai/checkpoints --port 8765
```

**Windows**

```bash
ai\build\Release\goes_server --checkpoint-dir ai\checkpoints --port 8765
```

**Endpoints:**

- `POST /move` — runs MCTS from the given position and returns the chosen move.
  - **Request fields:**
    - `config`: object with the game configuration (matches `shared/types.ts`'s `GameConfig.toJSON()` wire shape):
      - `boardType`: `"rect"` | `"rectd"` | `"cub"` | `"hcub"` | `"tri"` | `"twsq"` | `"gtsq"`
      - `boardArgs`: integer dimensions matching the board type (e.g. `[9, 9]` for a 9×9 rect board)
      - `numStones`, `numPlayers`, `forcedPassOnly`
      - `turnList`: array of `{player, stones, protected, friendly}` (see `shared/types.ts`'s `TurnInfo`) — `stones`/`protected`/`friendly` are `numStones`-length 0/1 arrays
      - `stoneToPlayerMap`: `{stone: player[]}` — stone color → the set of players it scores for (a stone mapped to several players credits each one its full point value, not split)
      - `playerStonePlaceLimit` _(optional, default all-unlimited)_: `(int|null)[][]`, `[stone-1][player-1]`
      - `globalStonePlaceLimit` _(optional, default all-unlimited)_: `(int|null)[]`, `[stone-1]`
      - `maxPlies` _(optional, default `null`)_: `int|null`
      - `allowSuicide` _(optional, default `false`)_: whether a move leaving the mover's own group with zero liberties is legal (self-captures that group immediately); part of `weak_equal()`'s checkpoint-matching fields like the other config fields
      - `scoreRule` _(optional, default `"area"`)_: `"stone"` | `"territoryonly"` | `"area"` | `"territory"` — see `BoardState::compute_points()`; part of `weak_equal()`'s checkpoint-matching fields like the other config fields. `"territory"` (real-world Japanese-style scoring) scores territory only in `compute_points()` (same as `"territoryonly"`) plus each player's `captureCount` (stones captured so far), the latter folded in separately at the player-aggregation layer (`compute_winners()`, and both reward formulas below) the same way `komi` is - see **Reward**, below
    - `moves`: full move history as an array. Each entry is either a legacy `int|null` (board index, or `null` for pass — the stone is auto-picked if the turn offers exactly one) or an object `{"pos": int|null, "stone": int|null}` (required once a turn offers more than one stone) — see `shared/types.ts`'s `ReplayMove`.
    - `board`: current stone array (length N) — used to verify the replayed state matches the client
    - `session_id`: opaque string returned by a previous response; omit, `""`, or `null` for a new session
    - `num_simulations` _(optional)_: overrides the server default
    - `temperature` _(optional)_: overrides the server default; `0` = argmax visit count, `>0` = sample from the visit distribution
  - **Response:** `{move, stone, policy, value, session_id}`
    - `move`: board index (0-based) or `null` for pass
    - `stone`: chosen stone color (1-indexed) or `null` for pass
    - `policy`: MCTS visit distribution over all `numStones*N+1` actions (stone-major flattened — see **Policy Head and Action Space**, below)
    - `value`: estimated value for the current player
    - `session_id`: include in the next request for this game to reuse the cached board state
  - The server caches a live `BoardState` per `session_id`. On each request it finds the longest common prefix of the stored and incoming move lists (comparing both `pos` and `stone`) and withdraws/advances the cached state when cheaper than a full replay, then verifies the result against the `board` field.
  - Checkpoint directories are opaque hash names, not derivable from `config` - the server scans `ai/checkpoints`'s subdirectories once (first request only; later-created directories need a server restart to be picked up) and matches `config` against each directory's saved config via `weak_equal()` (see **Checkpoint Directories and Matching**, below). The matched directory's model is then loaded lazily on first use and cached.
- `GET /health` — returns `{status, loaded_tags, device}`.

The `GOES_CHECKPOINT_DIR` environment variable overrides the checkpoint directory, and `GOES_NUM_SIMS` overrides the default number of MCTS simulations.

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--game-config PATH` | _(required)_ | Path to a GameConfig JSON file (`shared/types.ts`'s `GameConfig.toJSON()` wire shape - see `public/game_presets/`) - board type/dims, stone/player counts, turn list, scoring rule, komi, ko rule, suicide, etc. all come from here; `forcedPassOnly` must be `false` (the model isn't history-aware yet) |
| `--iterations N` | `200` | Training iterations |
| `--self-play-games N` | `10` | Games to complete before each training step |
| `--gamegen-batch-size N` | `25` | Games generated in parallel (pool size) |
| `--num-simulations N` | `200` | MCTS sims per move |
| `--linear-move-bound K1 K2` | _(none)_ | End games after Uniform(K1,K2)×N plies, resampled each time a game spawns |
| `--train-fraction F` | `0.1` | Train on `F × current buffer size` randomly selected game states per iteration, rounded up w.r.t. batch size |
| `--batch-size N` | `128` | Training batch size |
| `--gnn-hidden-dim N` | `128` | GNN hidden dimension (cub/hcub boards) |
| `--unet-hidden-dim N` | `16` | UNet hidden dimension (2D boards) |
| `--cnn-hidden-dim N` | `64` | CNN hidden dimension (2D boards) |
| `--cnn-conv-size N` | `5` | CNN convolution kernel size (2D boards) - must be odd and > 1 |
| `--num-layers N` | `9` | GNN message-passing layers |
| `--save-every N` | `10` | Save a checkpoint every N iterations |
| `--checkpoint-dir PATH` | `ai/checkpoints` | Checkpoint directory |
| `--resume TAG` | _(none)_ | Continue an existing hash-named checkpoint directory instead of starting a fresh one - see **Checkpoint Directories and Matching** |
| `--verbosity N` | `1` | 0=silent, 1=per-game, ≥2=per-ply |

## Input Features

The model's input is a self-describing JSON descriptor with shape `{"blocks": [[name, ...args], ...], "totalDims": N}`, built once per fresh training run by `compute_input_descr(const GameConfig&, int N)` (`ai/include/training/self_play.h`) and persisted into the checkpoint's `_config.json` as `ModelConfig::input_descr` - never recomputed on `--resume` or by the server (`server.cpp`'s `load_model()`), both of which only ever read whatever descriptor is already stored, so a checkpoint stays self-describing even if the block formulas below change later.

`board_to_features()` (`ai/src/model/features.cpp`) consumes this descriptor "event loop style": it walks `descr["blocks"]` in order and dispatches each entry by name to that block's fill logic, using the entry's own args (not a fresh `GameConfig` lookup) to know its width. Recognized block names:

- `stoneOccupancy` `[num_stones]` - `num_stones+1` channels: `is_stone[k]` one-hot (`k=0..num_stones-1`) + `is_empty`.
- `legalPlace` `[num_stones]` - `num_stones` channels: `is_legal[stone]`, one channel per stone color (a turn can offer several, with different legality via protected/friendly).
- `liberty` `[bits]` - `bits` channels: each node's raw liberty count, clamp-scale encoded (see below) - channel `i` is `max(0, 1 - liberty/2^i)`.
- `groupSize` `[]` - 1 channel: `group_size / N`.
- `plyMod` `[turn_list_len]` - `turn_list_len` channels: one-hot at `(ply_count % turn_list_len)`, broadcast to all nodes.
- `playerStoneBudget` `[bits_grid]` - `bits_grid` is a dense `num_stones x num_players` list of channel counts (stone-major); a 0 entry means "no limit configured for this pair, contributes zero channels." Each nonzero entry contributes that many broadcast channels: the remaining placement count for that `(stone, player)` pair, clamp-scale encoded like `liberty` above.
- `globalStoneBudget` `[bits]` - `bits` is a dense `num_stones`-length list of channel counts, same 0-means-absent/clamp-scale/broadcast convention as `playerStoneBudget`, but per-stone (summed across all players).

The `liberty`/`playerStoneBudget`/`globalStoneBudget` blocks use a soft, continuous stand-in for raw binary bit-encoding: `clamp_scale(value, bit_index) = max(0, 1 - value/2^bit_index)`, one channel per `bit_index` in `[0, bits)` - 1 at `value=0`, ramping linearly down to 0 at `value=2^bit_index` and clamped at 0 beyond, so a small change in `value` moves every channel by a small, continuous amount rather than flipping a bit discontinuously.

## Policy Head and Action Space

A turn can offer several stone colors at once (`TurnInfo.stones`), so the action space is per-`(stone, position)` rather than per-position: width `numStones*N + 1` (the `+1` is pass), flattened **stone-major** — `flat_index = (stone-1)*N + pos` for a placement, `flat_index = numStones*N` for pass. Every model's policy head, `board_to_features()`'s `legal_mask`, and MCTS's node arrays (`prior`/`visit_count`/`total_value`) all share this exact layout, so a flat action index decodes the same way everywhere: `stone = action / N + 1; pos = action % N` (unless `action == numStones*N`, which means pass).

Each architecture produces `numStones` per-position/per-node place-logit channels plus one pass-field channel (pooled to a single scalar exactly as before - via a learned affine reduction for CNN/UNet, an attention-weighted sum for GNN); the place channels are reshaped to `(B, numStones*N)` and concatenated with the pooled pass logit before a single softmax over the whole `(B, numStones*N+1)` vector - one probability distribution over "place this stone at this position, or pass," not a separate distribution per stone or per position.

Worked example: `numStones=3`, `N=9` (3×3 board) → width `3*9+1=28`. Stone 2 at position 5 → index `(2-1)*9+5=14`. Stone 1 at position 0 → index `0`. Pass → index `27`.

Stones a turn doesn't offer are simply illegal everywhere that ply (masked out via `legal_mask`, like any other illegal action) rather than shrinking the action space - the tensor width never changes turn-to-turn, only turn-to-turn *legality* does.

## Checkpoint Directories and Matching

Checkpoint directories are opaque: `ai/checkpoints/<hash>/`, where `<hash>` is a 64-character
SHA-256 (`sha256_hex()`, `ai/include/util/sha256.h`/`ai/src/util/sha256.cpp`) of the concatenation
of the model config's JSON (`ModelConfig::to_json()`), the game config's JSON
(`GameConfig::to_json()`), and a timestamp. The timestamp means every training run gets its own
fresh directory - starting `goes_train` twice with identical flags never reuses or collides with
an earlier run's directory.

**Starting fresh** (no `--resume`): `train.cpp`'s `handle_checkpoint_dir()` computes the hash,
errors if that exact directory somehow already exists (a collision), creates it, and prints the
hash to stdout so it can be passed to `--resume` later.

**Resuming** (`--resume TAG`): `handle_checkpoint_dir()` errors if `ai/checkpoints/TAG` doesn't
exist or has no checkpoint config to compare against, then requires the current run's game config
and model config to `strong_equal()`-match (`self_play.h`/`model_config.h`) the directory's saved
ones **exactly** - every field, not just a subset - before loading its latest weights and
replay-buffer history (`resume()`). This is deliberately strict: resuming into a directory whose
config doesn't match byte-for-byte would silently train an architecture-or-rules mismatch.

**Server-side matching**: since a `/move` request's config can no longer be turned into a
directory name directly, `server.cpp`'s `find_checkpoint_dir()` scans every subdirectory of
`ai/checkpoints` once (cached in `ServerState::checkpoint_index` - directories created after that
first scan need a server restart to be picked up), reads each one's saved `GameConfig`, and picks
the first whose saved config `weak_equal()`-matches the request. `weak_equal()` compares only the
fields that actually affect which trained model is appropriate for a given ruleset - board
type/args, `numStones`, `numPlayers`, `turnList`, `stoneToPlayerMap`, `forcedPassOnly`,
`allowSuicide`, `scoreRule`, `komi`, `koRule` (the same field set the old, now-removed,
human-readable `model_tag()` used to encode into its directory name) - so `playerStonePlaceLimit`/
`globalStonePlaceLimit`/`maxPlies` are intentionally allowed to differ between the request and the
matched checkpoint, same as before this scheme existed.

## How It Works

1. **Self-play**: the current model plays games against itself. Each move is chosen by running MCTS simulations guided by the model's policy and ownership estimates.
2. **Record collection**: each ply stores `(features, MCTS visit distribution)`; once a game ends, its final board's stone/territory ownership is recorded once, stone-type indexed with no player mapping (see Ownership Heads, below), and shared across every ply of that game as the ownership heads' training target.
3. **Training**: mini-batches are sampled from a replay buffer. The model is trained to predict the MCTS visit distribution (policy head, cross-entropy scaled by `1/log(numStones*N)` so the loss magnitude stays comparable across both board sizes and stone counts) and the final per-location stone/territory ownership (see Ownership Heads, below).
4. **Iteration**: the updated model is used for the next round of self-play.

## Reward

The value target for every ply is a sum of a rank-based component and a point-fraction component, both in (−1, 1), computed over **points** — stones, territory, or stones + territory, depending on the game's `scoreRule` (set via `--game-config`; see `BoardState::compute_points()` / Ownership Heads, below):

```
rank_reward  = (# opp with fewer points − # opp with more points) × 2 / (2P−1)   [0 if P=1]
point_reward = (point_fraction − 1/P) / (2P−1)
reward       = rank_reward + point_reward
```

where `point_fraction = (player's points) / (total points)`, or 0 when there are no points at all, and `P` is the number of players. The `[-1, 1]` interval is divided into `2P−1` equal sub-intervals; the rank reward places each player at the midpoint of the sub-interval corresponding to their rank, and the point reward adds a continuous signal within one half-interval width. The combined reward stays within `(−1, 1)` for all player counts. Because the point term is linear in `point_fraction`, the per-player rewards are exactly zero-sum whenever there are points at all (`Σ (point_fraction − 1/P) = 0` and the rank term is antisymmetric); the only non-zero-sum case is a fully empty/pointless terminal board.

This exact formula is used in two places: directly, via `compute_player_rewards()`, for genuine terminal game-over states inside MCTS - fed `BoardState::compute_points(state.score_rule, state.score())`; and indirectly, via `estimate_player_rewards()` (see Ownership Heads, below), for non-terminal leaves evaluated by the model - fed the same rule applied to the model's own predicted stone/territory estimates. Either way, MCTS backup has per-player rewards for every simulation — not just for the player to move at the leaf. Backup does not negate: each node looks up its own player's reward from the map, so Q-values at every node reflect that node's player's outcome directly.

Both formulas also fold in each player's `komi` before computing the rank/fraction reward - komi is added to every player's point total before the shared denominator (`total`/`sum_total`) is computed, so the point-fraction term stays zero-sum across players, and the search objective stays consistent with `BoardState::compute_winners()`'s own komi-adjusted winner determination. When `scoreRule` is `"territory"`, each player's `captureCount` (`BoardState::capture_count()`) is folded in the same way, right alongside komi - captures so far are a fully-known quantity at any ply (not a prediction), so this needs no special handling for non-terminal leaves either.

Since `scoreRule` now directly determines the reward signal self-play trains on, it's a game-dynamics config like `forcedPassOnly`: it's part of `weak_equal()`'s checkpoint-matching fields (a `stone`/`area`/`territoryonly`/`territory`-trained model is not interchangeable with another rule's), and games under different rules get separate checkpoint folders and separate `/move` sessions.

## Ownership Heads: Per-Location Stone/Territory Estimates

Rather than predicting the game outcome directly, the model predicts, for every board location, two independent distributions over "which stone type ends up occupying/holding this point when the game ends" - together called the **ownership** output, and stone-type indexed (1..num_stones) to match `ScoreData`/`BoardState::board` everywhere else in the codebase:

- **Stone estimate**: softmax over `num_stones + 1` channels (channel 0 = empty/no stone) - the model's belief about which stone type (if any) occupies this point at game end.
- **Territory estimate**: same channel layout (channel 0 = dame/occupied/no territory) - the model's belief about which stone type's territory this point is at game end.

Both come out of the same per-node features the policy head reads, via two independent `Linear→ReLU→Linear→softmax` heads (`stone_head`, `territory_head`), packaged as one `(2, N, num_stones+1)` ownership tensor (index 0 = stone estimate, index 1 = territory estimate).

**Training targets**: for each finished game, the final board (`BoardState::board`) and `ScoreData::territory_owner` (from the flood-fill territory computation - see `BoardState::count_score()`) are used *directly* as `stone_owner`/`territory_owner` (length-N, stone-type-indexed) ground truth - no player mapping. These are stored once per game in `GameRecord` and shared across every ply, the same way a per-game scalar target used to be broadcast in the old design.

**Loss** (`ai/src/train.cpp`):
- *Stone loss* / *Territory loss*: per-location squared error for each estimate - the predicted `(N, num_stones+1)` distribution against the true one-hot ownership (`stone_owner`/`territory_owner`, one-hot expanded via `torch::one_hot`), summed over the `num_stones+1` channels at each location (not averaged - averaging over channels as `torch::mse_loss` would do by default shrinks the loss too much to carry useful gradient signal), then averaged over locations and the batch: `(stone_estimate - one_hot(stone_owner))^2 .sum(channels) .mean(locations, batch)` (and the same for territory).
- *Point loss*: raw per-stone-type point total (no rank adjustment, no player aggregation), analogous to the scalar value loss from before the ownership refactor but supervising points instead of the rank-adjusted reward - the difference between `estimate_stone_points(ownership, score_rule)` (the model's own expected total per stone type) and the same function applied to a one-hot ground-truth "ownership" tensor built from `stone_owner`/`territory_owner`, scaled by `num_stones / N` (board size) before squaring - the raw difference can be as large as `N`, which would otherwise dwarf the per-location Stone/Territory losses - then summed over stone types and averaged over the batch: `((predicted_points - actual_points) / N * num_stones)^2 .sum(stone types) .mean(batch)`.

**MCTS and the `/move` endpoint still need a single per-player scalar** for backup/UCB and for the reported `value`. `estimate_player_rewards()` (`ai/include/model/evaluator.h`) calls `estimate_stone_points()` (the same function the point loss above uses) to get the model's expected per-stone-type point total under the state's `score_rule` - combined exactly like `compute_points()` does (stones only, territory only, or their sum) - aggregates that to per-player totals via `stone_to_player_map` (a small membership-matrix matmul, mirroring `compute_player_rewards()`'s own stone→player aggregation), then applies the exact same rank+fraction formula as `compute_player_rewards()` (see Reward, above), just fed continuous expected totals instead of integer ground-truth counts. MCTS always batches leaves from one game's config at a time, so a single rule/map per batched call is correct.

## MCTS Backup: Hybrid Averaging + Proven Values

`MCTS::backup()` uses the standard AlphaZero running average (`total_value/visit_count`) for every edge, with one override: if a child is `proven` — its `reward_estimate` came from an actual game-over state via `compute_player_rewards()`, not a GNN guess — and its value is currently the best available at that node, the node adopts that child's exact reward vector as its own instead of averaging it in, and is itself marked `proven`. This only ever propagates *exact* terminal outcomes upward; an unproven, few-visit GNN estimate never overrides anything, so ordinary positions still rely purely on averaging (the statistical robustness that makes MCTS work well for large-branching-factor games like this one — a pure minimax backup applied everywhere was considered and rejected for that reason, since it would let a single noisy, barely-sampled estimate dictate a node's value among dozens of siblings).

`BoardState::game_over()` covers `maxPlies` truncation directly via a live check (`ply_count() >= max_plies`, alongside a live resigned-players check and `MoveInfo.all_passed`, which only covers the consecutive-passes cause - mirroring `shared/boardState.ts`), so a state that hit its own `maxPlies` is `proven` on exactly the same footing as any other game-over state — once `game_over()` is true, `make_move()` always refuses further moves, so the recorded score genuinely is final rather than a heuristic snapshot of a game that could still continue.