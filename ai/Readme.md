## AI Training Pipeline

Self-play training pipeline for Goes using Monte Carlo Tree Search (MCTS) - KataGo's training and search algorithm, ported to Goes' more general rules engine (see **How It Works**).

Four model architectures are available:
- **CNN**: a plain residual conv stack (no pooling) for boards with a 2D integer embedding — `rect`, `rectd`, `tri`, `trihex`, `hex`, `hexdel`, `snubsq`, `twsq`, `gtsq`. The default for these boards.
- **UNet**: an alternative for the same board types (U-Net encoder/decoder with pooling), selectable via `--net-arch unet`.
- **MessagePassingGNN**: for higher-dimensional boards whose nodes cannot be laid out on a 2D grid — `cublat`, `hcub` — and for any `boardDescr` program whose resulting board doesn't have `emb_dim == 2` (e.g. one using `triangleForm`/`quadForm`/`globalCentralize`/`quadOctarize`, which always force `emb_dim = 0` - see **Differences from the TypeScript engine**, below). CNN/UNet's featurizer actually derives its grid shape from the input descriptor's `sideLens` (the training board's own side lengths, or `--side-lens`), not from `boardDescr` itself, so a modified 2D board that happens to keep `emb_dim == 2` (e.g. via `rectify`/`truncate`/`edgeSplit`/`mergeClose`/`scale`) wouldn't crash it either way - `effective_arch()` checks `bc.emb_dim` directly, with no separate allowlist of board kinds.
- **Transformer**: history-aware (attends over every previously reached board state, not just the current one), topology-agnostic, selectable via `--net-arch transformer` on any board type. This is the **only** architecture that supports `forcedPassOnly: true` — see **Transformer Architecture**, below.

## Differences from the TypeScript engine

`shared/boardConfig.ts` stores real-valued node positions (`pos`) and a bounding box (`boardDimension`) for the canvas renderer. The C++ `BoardConfig` carries the same positions in its `Embedding` (`emb.pos`, ported from the TypeScript factory functions) but as **exact unsigned integers only**, and omits `boardDimension`, which is only needed by the renderer; the `Embedding` also records the embedding box's side lengths (`emb.side_lens`) in place of the TypeScript `embDim`. Where the TypeScript positions are irrational, or need real-number averaging, the board drops to `emb_dim = 0` with an empty `pos[]` per node. Adjacency stays exact and complete either way, and only CNN/UNet read coordinates (they require `emb_dim == 2`):
- **Boards with `emb_dim = 0`:** `regpolyB`, `starB`, `apB`, `dodecaB`, `icosaB`, `reg120CellB`, `reg600CellB`, the flake/`mengerB` fractals and `diamondCubicB`. `tetraB`/`octaB` are not among them: they delegate to `simplex_board`/`orthoplex_board`, whose exact-integer embeddings stand in for the TypeScript coordinates.
- **Modifiers that force `emb_dim = 0`:** the `form` family (`triangleForm`/`quadForm`/`quadDiagForm`/`quadKnightForm`/`quadBishopForm`/`form`), the `localReplace` family (`simpCentralize`/`triCentralize`/`quadCentralize`, their `*Centering` variants, and `quadOctarize`) and `globalCentralize`. The rest (`rectify`, `truncate`, `edgeSplit`, `mergeClose`, `scale`, `nis`/`eis`) keep the board's `emb_dim`, so a 2D board stays CNN/UNet-eligible after them. `rectify`/`truncate`/`mergeClose` throw on an `emb_dim = 0` board, since they need real edge directions and distances.
- **`scale`** is a no-op (returns the board unchanged): `emb.pos` must stay exact integers, and C++ never renders positions.

`BoardState` otherwise has full feature parity with `shared/boardState.ts`: multi-stone-per-turn offering (`TurnInfo.stones`), protected/friendly stones, per-player and global stone placement limits, `maxPlies`, resignation (permanent — never undone by `withdraw_move()`, and `game_over()` re-checks the resigned set live rather than stamping it onto a move), `komi` (per-player scoring handicap, applied in `compute_winners()` and folded into the per-player totals the search utility compares — see **Search Utility**, below), and `koRule` (both `positional` and `situational` superko are implemented — `HistoryManager`'s `(ply_mod, board)` key collapses `ply_mod` to a constant under `positional`, since two boards are then "the same" purely by content, mirroring `compareState()` skipping the ply-mod comparison in `shared/boardState.ts`). `stoneToPlayerMap` is stone → *set* of players (each mapped player gets a stone's full point value, not split), matching the TypeScript side.

Not ported (no C++ consumer needs it): `GameConfig.players` (the `PlayerInfo` slot assignments) — `parse_game_cfg()` ignores it and `GameConfig::to_json()` omits it, since self-play and the inference server search every player with one model.

## Directory Structure

```
ai/
├── CMakeLists.txt
├── third_party/
│   ├── httplib.h                       single-header HTTP server (cpp-httplib)
│   └── nlohmann/json.hpp               single-header JSON library
├── include/ and src/                   (headers in include/, implementations in src/)
│   ├── game/
│   │   ├── board_config.{h,cpp}        Port of shared/boardConfig.ts (adjacency + integer embedding)
│   │   ├── board_state.{h,cpp}         Port of shared/boardState.ts
│   │   ├── cleg_base.{h,cpp}           Port of shared/clegBase.ts (cleg AST/type/value declarations)
│   │   ├── cleg_parser.{h,cpp}         Port of shared/clegParser.ts (source text → AST)
│   │   ├── cleg_check.{h,cpp}          Port of shared/clegCheck.ts (static type checker)
│   │   ├── cleg_eval.{h,cpp}           Port of shared/clegEval.ts (builtins + evaluator → BoardConfig)
│   │   ├── cleg_eval_internal.h        Declarations shared only by cleg_eval.cpp and cleg_check.cpp (BUILTIN_FUNCTIONS, EGR_TYPE)
│   │   ├── selector.{h,cpp}            Port of shared/selector.ts (node/edge/simplex/quad selectors)
│   │   ├── fractal.{h,cpp}             Port of shared/fractal.ts (flake/Menger glue tables)
│   │   ├── geometry.{h,cpp}            Port of shared/geometry.ts (LP-based convex hull, for rectify)
│   │   └── topology.{h,cpp}            Port of shared/topology.ts (merge_boards, find_simplices/quads)
│   ├── model/
│   │   ├── features.{h,cpp}            BoardState → per-node feature tensor and legal-action mask (see Input Features)
│   │   ├── gnn.{h,cpp}                 MessagePassingGNN (policy + ownership + rank heads, for cublat/hcub)
│   │   ├── unet.{h,cpp}                U-Net (policy + ownership + rank heads, for 2D boards)
│   │   ├── cnn.{h,cpp}                 Plain residual CNN (policy + ownership + rank heads, for 2D boards)
│   │   ├── transformer.{h,cpp}         History-aware Transformer (policy + ownership + rank heads, any board; only arch supporting forcedPassOnly)
│   │   ├── rank_head.{h,cpp}           Pairwise-outcome (rank) head shared by every architecture
│   │   ├── model_config.{h,cpp}        ModelConfig and its per-architecture subclasses, JSON (de)serialization
│   │   ├── model_output.h              ModelOutput (the policy/ownership/rank tensors every model returns) and ModelInput (one position's prepared input)
│   │   ├── evaluator.h                 Type-erased Evaluator: per-position prepare() + batched run()
│   │   └── any_model.h                 AnyModel variant, build_model() and make_evaluator()
│   ├── mcts/
│   │   ├── mcts.{h,cpp}                Single-root MCTS, one simulation at a time, with KataGo's search features (see Search)
│   │   ├── nn_server.{h,cpp}           KataGo's NNEvaluator-style batching queue shared by all self-play games (see Self-Play)
│   │   └── utility.{h,cpp}             The search utility: KataGo's win/loss + score utility for N players
│   ├── training/
│   │   ├── self_play.{h,cpp}           Game config parsing; the self-play runner, game records and trajectory replay (see Self-Play)
│   │   └── replay_buffer.{h,cpp}       The training window over recorded rows (see Training Loop)
│   └── util/
│       └── sha256.{h,cpp}              Checkpoint-directory hashing (see Checkpoint Directories and Matching)
├── src/
│   ├── train.cpp                       Main training entry point → binary: goes_train
│   └── server.cpp                      HTTP inference server: POST /move → binary: goes_server
├── analysis.py                         Offline reader for dumped _traj.json trajectories / training logs
└── checkpoints/                        Saved model weights (created at runtime)
```

## Prerequisites

1. **MSVC 2022** (or GCC/Clang on Linux) with C++20 support
2. **CMake ≥ 3.18**
3. **LibTorch** — download the pre-built package from https://pytorch.org/get-started/locally/
   - Select: your OS (IMPORTANT!), C++ / LibTorch, CUDA version or CPU
   - Extract to a local directory, e.g. `D:\libtorch` on windows and `~/include/libtorch` on linux
4. **Third-party headers** (single-file, copy into `third_party/`):
   - [httplib.h](https://github.com/yhirose/cpp-httplib/releases) → `third_party/httplib.h`
   - [json.hpp](https://github.com/nlohmann/json/releases) → `third_party/nlohmann/json.hpp`

## Build

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
1. Make sure that the absolute paths do not contain any unicode characters.
2. The build adds `-Wl,--disable-new-dtags` on Linux so that `RPATH` (rather than `RUNPATH`) is embedded in the binaries. This is necessary because `libcupti` is a transitive dependency and `RUNPATH` is not searched transitively by the dynamic linker.
3. On Windows with Visual Studio generator, DLLs are automatically copied next to the binaries by the CMakeLists.
4. Pass `-DSANITIZE=ON` (Linux only) to enable AddressSanitizer/UndefinedBehaviorSanitizer (`-fsanitize=address,undefined`) - catches memory errors (use-after-free, out-of-bounds access, etc.) at the exact point they occur, with a full stack trace, rather than however much later they might otherwise surface as a crash. Useful when chasing a segfault; adds meaningful runtime overhead, so leave it off for normal training runs. However, before debugging segfault, make sure the cause of segfault is not the machine running out of memory.

## Training

Run from the project root:

**Linux:**

```bash
ai/build/goes_train --game-config public/game_presets/9x9_go.json --linear-move-bound 1.5 2.5 --min-rows 10000
```

The defaults are KataGo's - `synchronous_loop.sh`'s 500 games per iteration and `selfplay1.cfg`'s
searches (600-visit full searches and 100-visit cheap ones) - except that training starts once 2048
rows exist (KataGo waits for 100k); these commands start training at 10k rows (`--min-rows 10000`).
For larger runs:

```bash
ai/build/goes_train --game-config public/game_presets/9x9_go.json --linear-move-bound 1.5 2.5 --gamegen-batch-size 256 --iterations 1000 --cnn-hidden-dim 128 --min-rows 10000

ai/build/goes_train --game-config public/game_presets/9x9_go.json --linear-move-bound 1.5 2.5 --gamegen-batch-size 256 --iterations 1000 --net-arch gnn --gnn-hidden-dim 256 --min-rows 10000

ai/build/goes_train --game-config public/game_presets/7x7x2_twsq_go.json --linear-move-bound 1.5 2.5 --gamegen-batch-size 256 --iterations 1000 --cnn-hidden-dim 128 --min-rows 10000

ai/build/goes_train --game-config public/game_presets/3x3_go_fpo.json --gamegen-batch-size 256 --iterations 1000 --net-arch transformer --transformer-hidden-dim 128 --min-rows 10000

ai/build/goes_train --game-config public/game_presets/13x13_two_ply_go.json --linear-move-bound 1.5 2.5 --gamegen-batch-size 256 --iterations 1000 --cnn-hidden-dim 128 --min-rows 10000
```

**Windows:**

```powershell
ai\build\Release\goes_train --game-config public/game_presets/9x9_go.json --linear-move-bound 1.5 2.5 --min-rows 10000
```

A run's files go to `<checkpoint-dir>/<hash>/` (`--checkpoint-dir`, default `ai/checkpoints`),
where `<hash>` is a fresh 64-character SHA-256 of the model config, game config, and a timestamp -
a new, unique directory every run, never reused implicitly:

- `<arch>_config.json` - the game and model config (what the server matches requests against)
- `<arch>_XXXXXX.pt` - the model exported at the end of iteration `XXXXXX`, kept only every
  `--saves-every` iterations (default 8) and at iteration `--iterations` (what the server uses; the
  newest one wins). A run stopped earlier - e.g. killed mid-run - keeps its newest weights only in
  `trainer/`, so its newest `.pt` can be up to `--saves-every - 1` iterations older
- `<arch>_XXXXXX_traj.json` - that iteration's self-play games (see **Self-Play**)
- `trainer/` - what `--resume` needs: the trained model, its weight average, the optimizer and the
  loop's counters (`state.json`)

To continue a run later, pass `--resume <hash>` (printed at startup; it is also the directory name) -
this validates that the current `--game-config`/architecture flags match the saved config exactly,
restores the trainer, and refills the training window from the newest recorded games. See
**Checkpoint Directories and Matching**, below.

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
    ai/build/goes_train --game-config public/game_presets/9x9_go.json --verbosity 2 --linear-move-bound 0.5 1.5 --iterations 3 --games-per-iteration 10 --gamegen-batch-size 5 --full-visits 10 --cheap-visits 4 --min-rows 50

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
    ai/build/goes_train --game-config public/game_presets/9x9_go.json --verbosity 2 --linear-move-bound 0.5 1.5 --iterations 3 --games-per-iteration 4 --gamegen-batch-size 2 --full-visits 5 --cheap-visits 2 --min-rows 50

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

- `POST /move` — runs a search from the given position (with KataGo's play-mode search settings - see **Search**) and returns the chosen move.
  - **Request fields:**
    - `config`: a `GameConfig`, in `shared/gameConfig.ts`'s `toJSON()` wire shape (see that class for the fields). `boardSeed` is required, `players` is ignored, and which fields select the checkpoint is described in **Checkpoint Directories and Matching**, below.
    - `moves`: full move history as an array of `{"pos": int|null, "stone": int|null}` objects (`shared/types.ts`'s `ReplayMove`) — `pos` is the board index, or `null` for pass; `stone` may be `null` only when the turn offers exactly one stone, which is then auto-picked.
    - `resigns` _(optional, default none)_: `[[ply, [player, ...]], ...]` — who resigned at each ply, the same shape as `shared/gameConfig.ts`'s `FinishedGame.toJSON()`'s `resigns` field. Replayed interleaved with `moves` (see `replay_tail()`, `server.cpp`), since a resigned player may always pass even under `forcedPassOnly`, but the resignation must be applied at the ply it actually happened.
    - `board`: current stone array (length N) — used to verify the replayed state matches the client
    - `session_id`: opaque string returned by a previous response; omit, `""`, or `null` for a new session
    - `num_simulations` _(optional)_: overrides the server default; must be `>= 1` (the request is rejected otherwise)
    - `temperature` _(optional)_: overrides the server default; `0` = the search's best move (its LCB-adjusted play-selection values' argmax), `>0` = sample from those values at that temperature
  - **Response:** `{move, stone, policy, value, session_id}`
    - `move`: board index (0-based) or `null` for pass
    - `stone`: chosen stone color (1-indexed) or `null` for pass
    - `policy`: the search's policy over all `numStones*N+1` actions (its pruned play-selection values, normalized; stone-major flattened — see **Policy Head and Action Space**, below)
    - `value`: the search's average utility for the player to move (see **Search Utility**)
    - `session_id`: include in the next request for this game to reuse the cached board state
  - The server caches a live `BoardState` per `session_id`. On each request it finds the longest common prefix of the stored and incoming move lists (comparing both `pos` and `stone`) and withdraws/advances the cached state when cheaper than a full replay, then verifies the result against the `board` field.
  - Checkpoint directories are opaque hash names, not derivable from `config` - the server scans the `--checkpoint-dir` directory's subdirectories once (first request only; later-created directories need a server restart to be picked up) and matches `config` against each directory's saved config via `weak_equal()` (see **Checkpoint Directories and Matching**, below). The matched directory's model is then loaded lazily on first use and cached.
- `GET /health` — returns `{status, loaded_tags, device}`.

`goes_server` accepts `--checkpoint-dir PATH` (default `ai/checkpoints`), `--port N` (default `8765`), `--sims N` (default number of MCTS simulations, default `200`, must be `>= 1`), `--temperature F` (default `0.0` = the search's best move) and `--cpu` (force CPU even if CUDA is available). The `GOES_CHECKPOINT_DIR`, `GOES_NUM_SIMS` and `GOES_TEMPERATURE` environment variables override the checkpoint directory, default simulation count and default temperature respectively (they are applied after the flags, so they win).

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--game-config PATH` | _(required)_ | Path to a GameConfig JSON file (`shared/gameConfig.ts`'s `GameConfig.toJSON()` wire shape - see `public/game_presets/`) - board description (cleg source text, `boardDescr`), stone/player counts, turn list, scoring rule, komi, ko rule, suicide, etc. all come from here; `forcedPassOnly: true` requires `--net-arch transformer` (the other three architectures aren't history-aware) |
| `--iterations N` | `200` | Self-play / train / export cycles |
| `--saves-every N` | `8` | Keep the exported model as `<arch>_XXXXXX.pt` only every N iterations and at iteration `--iterations`; `trainer/` and every `_traj.json` are still written each iteration |
| `--games-per-iteration N` | `500` | Self-play games finished per cycle |
| `--gamegen-batch-size N` | `128` | Games played concurrently, one thread each (KataGo's `numGameThreads`) - see **Self-Play** |
| `--nn-server-threads N` | `1` | Threads evaluating the games' network requests (KataGo's `numNNServerThreadsPerModel`) |
| `--nn-max-batch N` | `128` | Most network requests evaluated as one batch (KataGo's `nnMaxBatchSize`) |
| `--selfplay-gpus LIST` | _(training device)_ | Comma-separated CUDA devices for self-play's network evaluation, e.g. `0,1`: network server thread `i` uses the `(i mod count)`-th, each GPU holding its own copy of the model (KataGo's `cudaDeviceToUseThread<N>`). `--nn-server-threads` must be at least the number of devices. Training stays on the training device |
| `--full-visits N` | `600` | Visits of a full search - the searches that become training data |
| `--cheap-visits N` | `100` | Visits of a cheap search - played, but not trained on |
| `--cheap-prob F` | `0.75` | Probability that a move gets a cheap search |
| `--c-puct F` | `1.1` | PUCT exploration constant |
| `--linear-move-bound K1 K2` | _(none)_ | End games after Uniform(K1,K2)×N plies, resampled each time a game spawns (combined with the game config's `maxPlies` by taking the tighter bound) |
| `--batch-size N` | `128` | Training batch size |
| `--lr F` | `0.001` | Adam learning rate after warmup |
| `--l2 F` | `0.0001` | Adam weight decay |
| `--no-lr-warmup` | _(off)_ | Skip KataGo's learning-rate warmup over the first 2M training samples |
| `--min-rows N` | `2048` | Rows needed before training starts (see **Training Loop**; KataGo's is 100000) |
| `--taper-window-scale F` | `50000` | Training-window power-law scale |
| `--taper-window-exponent F` | `0.65` | Training-window power-law exponent |
| `--expand-window-per-row F` | `0.4` | Training-window growth per new row |
| `--max-train-per-data F` | `4` | Training samples allowed per new row (KataGo's is 8; Goes has no symmetry augmentation, so every reuse of a row is an identical input) |
| `--max-train-samples-per-iteration N` | `500000` | Cap on training samples per cycle |
| `--swa-scale F` | `8` | Window of the exported weight average, in snapshots |
| `--swa-period-samples N` | `80000` | Training samples between weight-average snapshots; `0` exports the raw model |
| `--net-arch NAME` | `auto` | `auto`\|`cnn`\|`unet`\|`gnn`\|`transformer` - `auto` picks `cnn` for a board with `emb_dim == 2`, else `gnn` (see `effective_arch()`) |
| `--cpu` | _(off)_ | Force CPU even if CUDA is available |
| `--gnn-hidden-dim N` | `128` | GNN hidden dimension (cublat/hcub and other non-2D boards) |
| `--unet-hidden-dim N` | `16` | UNet hidden dimension (2D boards) |
| `--cnn-hidden-dim N` | `64` | CNN hidden dimension (2D boards) |
| `--cnn-conv-size N` | `3` | CNN convolution kernel size (2D boards) - must be odd and > 1 |
| `--transformer-hidden-dim N` | `128` | Transformer hidden dimension (any board) |
| `--num-layers N` | `9` | GNN message-passing layers |
| `--num-attn-layers N` | `8` | Transformer cross-attention layer count (separate flag from `--num-layers` - not shared with GNN) |
| `--side-lens L1,L2,...` | _(the board's own)_ | A standard box `sideLens` for boards of several sizes (see **Input Features**), which every network's input and output are zero-extended to, so it must contain the board's own side lengths. Any lengths for a board without an embedding, where they only set `--n-bound`'s default |
| `--n-bound N` | _(product of the side lengths)_ | The node bound `nBound` (see **Input Features**), at least the board's node count. This or `--side-lens` is required for a board without an embedding |
| `--checkpoint-dir PATH` | `ai/checkpoints` | Checkpoint directory |
| `--resume TAG` | _(none)_ | Continue an existing hash-named checkpoint directory - see **Checkpoint Directories and Matching**. `--side-lens`/`--n-bound` must be repeated, since they're part of the stored model config |
| `--retrain TAG` | _(none)_ | Train a fresh model on an existing directory's recorded games (its `_traj.json` files, replayed in order into a *new* directory, training and exporting after each file as the original run did), then continue with self-play for any remaining `--iterations`. Only the game config must match - it never loads `TAG`'s weights. Mutually exclusive with `--resume` |
| `--verbosity N` | `1` | 0=silent, 1=per-game and per-iteration, ≥2=also each iteration's network batch counts and evaluation time |

## Input Features

The model's input is a self-describing JSON descriptor with shape `{"blocks": [[name, ...args], ...], "totalDims": F, "sideLens": side_lens, "nBound": n_bound}`, built once per fresh training run by `compute_input_descr(const GameConfig&, const std::vector<unsigned>& side_lens, int n_bound)` (`ai/include/training/self_play.h`) and persisted into the checkpoint's `_config.json` as `ModelConfig::input_descr`. The server (`server.cpp`'s `load_model()`) never recomputes it - it only reads the stored descriptor, so a checkpoint stays self-describing even if the block formulas below change later. `--resume` does recompute it, as part of the model config it requires to match the stored one exactly (see **Checkpoint Directories and Matching**). Used identically by all four architectures for their CURRENT-ply-only input.

`board_to_features()` (`ai/src/model/features.cpp`) consumes this descriptor "event loop style": it walks `descr["blocks"]` in order and dispatches each entry by name to that block's fill logic, using the entry's own args (not a fresh `GameConfig` lookup) to know its width. Board size enters only through two top-level fields. `sideLens` is a standard box: the training board's embedding side lengths (`Embedding::side_lens`, which every board construction function derives from its arguments and input boards rather than from the nodes it keeps), or `--side-lens`, one covering every size a randomly sized board can take. Every network's input and output live in the slot space these two fields fix (`NodeSlots`, `compute_node_slots()` in `model/features.h`): a node goes to its position's cell of the box - on a board without an embedding, to slot `i` of `nBound`. Each architecture's `prepare_input()` zero-extends a position to it (`slot_input()`), the replay buffer zero-extends each sampled training row the same way, and `output_rows()` reads each row's policy and ownership back at its own board's nodes - so the weights depend only on these two fields, one batch can mix boards, and any board a model runs on, including each server request's, must fit (`fits_in()`, and at most `nBound` nodes). The GNN pads each row's adjacency to the slots too, computed once per board and cached by the model (`MessagePassingGNNImpl::adj_norms_of()`); its neighbor-count embedding table has `nBound + 1` rows. `nBound` is an upper bound on the board's node count, which sizes the features below: `num_cells` - the product of `sideLens` - unless `--n-bound` overrides it. A board without an embedding needs one of the two flags. Recognized block names:

- `stoneOccupancy` `[num_stones]` - `num_stones+1` channels: `is_stone[k]` one-hot (`k=0..num_stones-1`) + `is_empty`.
- `legalPlace` `[num_stones]` - `num_stones` channels: `is_legal[stone]`, one channel per stone color (a turn can offer several, with different legality via protected/friendly).
- `liberty` `[bits]` - `bits` channels (enough to binary-encode `nBound`): each node's raw liberty count, clamp-scale encoded (see below) - channel `i` is `max(0, 1 - liberty/2^i)`.
- `groupSize` `[]` - 1 channel: `group_size / nBound`.
- `plyMod` `[turn_list_len]` - `turn_list_len` channels: one-hot at `(ply_count % turn_list_len)`, broadcast to all nodes.
- `consectivePassOneHot` `[turn_list_len+1]` - `turn_list_len+1` channels: one-hot at the current consecutive-pass count (`MoveInfo::consecutive_passes`, which ranges over `[0, turn_list_len]` - the game ends once it would exceed `turn_list_len`), broadcast to all nodes.
- `playerStoneBudget` `[bits_grid]` - `bits_grid` is a dense `num_stones x num_players` list of channel counts (stone-major); a 0 entry means "no limit configured for this pair, contributes zero channels." Each nonzero entry contributes that many broadcast channels: the remaining placement count for that `(stone, player)` pair, clamp-scale encoded like `liberty` above.
- `globalStoneBudget` `[bits]` - `bits` is a dense `num_stones`-length list of channel counts, same 0-means-absent/clamp-scale/broadcast convention as `playerStoneBudget`, but per-stone (summed across all players).

The `liberty`/`playerStoneBudget`/`globalStoneBudget` blocks use a soft, continuous stand-in for raw binary bit-encoding: `clamp_scale(value, bit_index) = max(0, 1 - value/2^bit_index)`, one channel per `bit_index` in `[0, bits)` - 1 at `value=0`, ramping linearly down to 0 at `value=2^bit_index` and clamped at 0 beyond, so a small change in `value` moves every channel by a small, continuous amount rather than flipping a bit discontinuously.

**Transformer's two descriptors**: the Transformer is the only architecture that uses two independent descriptors instead of one. Its CURRENT ply uses the exact same `compute_input_descr()` output as CNN/UNet/GNN, unmodified - a single current state is always safe to describe fully, since there's no risk of it recurring at a different point in time with different bits. Every PAST ply instead uses a much smaller, separate 2-block descriptor (`plyMod`, `stoneOccupancy` only) built directly in `train.cpp` and stored as `TransformerConfig::history_descr` - `legalPlace` (and every other block) is excluded here because it depends on transient per-turn context (protected/friendly flags, which stones a turn happens to offer) rather than the board's physical content, so the same physical position recurring later in a game could get different `legalPlace` bits - breaking the "one board position → one embedding" property the transformer's cross-state comparison relies on. `liberty`/`consectivePassOneHot` are also omitted from the history descriptor, since they add cost (`liberty` needs a `group_liberty()` traversal per historical ply) without being essential to the history signal.

## Policy Head and Action Space

A turn can offer several stone colors at once (`TurnInfo.stones`), so the action space is per-`(stone, position)` rather than per-position: width `numStones*N + 1` (the `+1` is pass), flattened **stone-major** — `flat_index = (stone-1)*N + pos` for a placement, `flat_index = numStones*N` for pass. Every model's policy head, `board_to_features()`'s `legal_mask`, and MCTS's action indices all share this exact layout, so a flat action index decodes the same way everywhere: `stone = action / N + 1; pos = action % N` (unless `action == numStones*N`, which means pass).

Each architecture produces `numStones` per-position/per-node place-logit channels plus one pass-field channel, pooled to a single scalar via a learned attention-weighted sum over nodes (identical mechanism across all four - see `*PolicyHeadImpl` in `cnn.cpp`/`unet.cpp`/`gnn.cpp`/`transformer.cpp`); the place channels are reshaped to `(B, numStones*N)` and concatenated with the pooled pass logit before a single softmax over the whole `(B, numStones*N+1)` vector - one probability distribution over "place this stone at this position, or pass," not a separate distribution per stone or per position.

Worked example: `numStones=3`, `N=9` (3×3 board) → width `3*9+1=28`. Stone 2 at position 5 → index `(2-1)*9+5=14`. Stone 1 at position 0 → index `0`. Pass → index `27`.

Stones a turn doesn't offer are simply illegal everywhere that ply (masked out via `legal_mask`, like any other illegal action) rather than shrinking the action space - the tensor width never changes turn-to-turn, only turn-to-turn *legality* does.

## Transformer Architecture

Unlike CNN/UNet/GNN, which evaluate only the current `BoardState`, the transformer attends over every board state reached so far in the game - this is what lets it support `forcedPassOnly: true`, where legality (superko in particular) depends on the *set* of previously-reached states, not just the current one. `BoardState` already content-interns its full history via `HistoryManager` (`board_at(ply)`, `consecutive_passes_at(ply)`), so `Evaluator`/MCTS need no transformer-specific interface: `TransformerImpl::prepare_input()` (`ai/src/model/transformer.cpp`, the transformer's `Evaluator::prepare()`) builds a position's current-ply features, legal mask and every past ply's history features from a single `const BoardState&`, on the CPU, in the thread that requests the evaluation; `evaluate_inputs()` (its `Evaluator::run()`) zero-pads the batch's histories to the longest one and runs one forward pass.

Pipeline, applied per sampled ply during training (and per MCTS leaf during self-play):

1. **Encoders** (two - see **Transformer's two descriptors** in Input Features, above): the *current* state's full-descriptor `(N, F)` features are zero-extended to the `S` slots of `NodeSlots` (see **Input Features** - `S` is the product of `sideLens`, or `nBound` without an embedding) and flattened to `(S·F,)`, then `Linear(S·F, D) → ReLU → Linear(D, D)` (`encoder_in`/`encoder_out`). Every *past* state's narrower `(N, F_hist)` history-descriptor features, zero-extended the same way, go through a separate, identically-shaped MLP with its own weights (`hist_encoder_in`/`hist_encoder_out`, shared across every past ply), landing in the same `D`-dim space. Both deliberately destroy board topology information (no adjacency/shape assumption) in exchange for being usable on any board type.
2. **History set**: every *past* state's `(D,)` embedding forms an unordered set, used as-is (no self-attention refinement pass). No positional/recency encoding is ever added, so the set stays permutation-equivariant (order-symmetric) by construction. A learned "history sentinel" token is prepended (never masked) so attention always has at least one valid key, even for a genesis state with zero past plies.
3. **Cross-attention**: the *current* state's `(D,)` embedding is the query (it never joins the history set itself), attending directly over the (unrefined) history set as keys/values, stacked `--num-attn-layers` times - only the query evolves layer-to-layer, the key/value set stays fixed.
4. **Decoder**: the literal inverse of the encoder - a monolithic MLP `Linear(D, D) → ReLU → Linear(D, S·D)`, reshaped to `(S, D)` and read back at the board's `N` slots as `(N, D)`. No per-node operation anywhere in the decode path either, keeping the whole pipeline topology-agnostic end to end.
5. **Heads**: the resulting `(N, D)` tensor feeds the same shared `stone_head`/`territory_head`/policy-head shapes CNN/UNet/GNN use, unmodified.

**Cost tradeoff**: unlike the other three architectures' O(1)-per-leaf evaluation, this is O(ply_count) per leaf (it reconstructs every historical ply's features from scratch) - noticeably more expensive per MCTS simulation for long games, though each past-ply reconstruction skips the liberty computation entirely (the history descriptor never needs it - see `board_to_features_at_ply()`'s perf guard, `features.cpp`); only the current ply's full-descriptor reconstruction pays for it. Worth tuning `--full-visits`/`--cheap-visits`/`--gamegen-batch-size` down accordingly for a transformer run. Cross-attention's query has length 1, so its own cost is O(ply_count) per layer, not self-attention's O(ply_count²) - see **Notes**, below, for why the history set gets no self-attention.

## Checkpoint Directories and Matching

Checkpoint directories are opaque: `<checkpoint-dir>/<hash>/` (`--checkpoint-dir`, default
`ai/checkpoints`), where `<hash>` is a 64-character SHA-256 (`sha256_hex()`,
`ai/include/util/sha256.h`/`ai/src/util/sha256.cpp`) of the concatenation
of the model config's JSON (`ModelConfig::to_json()`), the game config's JSON
(`GameConfig::to_json()`), and a timestamp. The timestamp means every training run gets its own
fresh directory - starting `goes_train` twice with identical flags never reuses or collides with
an earlier run's directory.

**Starting fresh** (no `--resume`): `train.cpp`'s `fresh_checkpoint_dir()` computes the hash,
errors if that exact directory somehow already exists (a collision), creates it, and prints the
hash to stdout so it can be passed to `--resume` later.

**Resuming** (`--resume TAG`): `train.cpp` errors if `<checkpoint-dir>/TAG` has no trainer state or
no checkpoint config to compare against, then requires the current run's game config
and model config to `strong_equal()`-match (`self_play.h`/`model_config.h`) the directory's saved
ones **exactly** - every field, not just a subset (`validate_dir_config()`) - before restoring
`trainer/` and refilling the training window from the newest recorded games. This is deliberately strict: resuming into a directory whose
config doesn't match byte-for-byte would silently train an architecture-or-rules mismatch.

**Server-side matching**: a directory name can't be derived from a `/move` request's config, so
`server.cpp`'s `find_checkpoint_dir()` scans every subdirectory of the `--checkpoint-dir` directory once (cached in
`ServerState::checkpoint_index` - directories created after that first scan need a server restart
to be picked up), reads each one's saved `GameConfig`, and picks the first whose saved config
`weak_equal()`-matches the request. `weak_equal()` compares only the fields that affect which
trained model is appropriate for a given ruleset - `boardDescr`, `numStones`, `numPlayers`,
`turnList`, `stoneToPlayerMap`, `forcedPassOnly`, `allowSuicide`, `scoreRule`, `komi`, `koRule`,
and which `playerStonePlaceLimit`/`globalStonePlaceLimit` entries are set (they decide the model's
budget input channels - see **Input Features**) - so `boardSeed`, `maxPlies` and the place limits'
values may differ between the request and the matched checkpoint. (`boardSeed` is left out because
training builds its board from a fresh random seed each run.)

The matched directory's model is loaded on its first request and cached: the newest
`<arch>_XXXXXX.pt` of the architecture named by the directory's `<arch>_config.json` (a run directory
holds one architecture). Only the exports of every `--saves-every`-th iteration and of iteration
`--iterations` are kept as `.pt` files, so this can lag the newest trained weights in `trainer/`,
which the server never reads.

## How It Works

Training follows KataGo's synchronous self-play loop (`python/selfplay/synchronous_loop.sh` in KataGo), adapted to Goes' general rules - any board graph, any number of stone types, players and turn-list entries. Each iteration:

1. **Self-play** (see **Self-Play**): `--games-per-iteration` games are played with the newest exported model - every player searched by that one model, with no gating. Before the first model exists, a uniform evaluator stands in for it.
2. **Record collection**: each finished game becomes a `GameRecord` of training rows - its fully searched positions, each with the search's policy target, the game's final stone/territory ownership (see **Ownership Heads**) and a blended pair-outcome target (see **Value Targets**).
3. **Training** (see **Training Loop**): mini-batches are drawn from a window over the newest rows, for as many samples as the new data allows.
4. **Export**: the weight average (or, before it exists, the trained model) becomes the next iteration's self-play model, and is kept as `<arch>_XXXXXX.pt` every `--saves-every` iterations and at iteration `--iterations`.

KataGo features left out because they don't fit Goes' rules engine or would change the model architecture: board-symmetry augmentation (Goes boards are arbitrary graphs), komi/handicap/board-size randomization and fair-komi (komi is part of a game's config, and not a model input), draw/no-result utilities, asymmetric playouts and lead estimation (two-player Go specifics), pass-alive endings and passing hacks (Benson's algorithm is Go-specific), subtree value-bias correction (keyed on 5×5 grid patterns), and the opponent-policy, soft-policy and score-distribution heads (new heads). Graph search, the neural-net cache, tree reuse and multi-threaded virtual loss are also not ported: KataGo's self-play clears its tree for every full search and runs one search thread per game anyway.

## Search

`MCTS::search()` (`ai/src/mcts/mcts.cpp`) searches one root, one leaf at a time, getting each new leaf's network outputs from an `NNQuery` - in self-play the shared `NNServer` (see **Self-Play**), in the server a direct one-position call. Each node keeps search averages of every player's utility (see **Search Utility**), its square (for LCB), the win term, and the rank head's pair-outcome probabilities, its own evaluation counted as its first visit. A terminal position is scored exactly instead of by the network. Every root's tree is searched with one model.

The settings come from KataGo (`SearchParams::self_play()` = its self-play config, `cpp/configs/training/selfplay1.cfg`; `SearchParams::server()` = the values it plays with outside self-play, `cpp/program/setup.cpp`'s defaults - only the parameter values are taken, not KataGo's engine protocol):

- **PUCT**: a child's value is `Q + c · sqrt(N + 0.01) · P / (1 + n)`, with `c = cpuct + cpuct_log · log((N + 500) / 500)` (self-play: `--c-puct` 1.1, no log growth; server: 1.0 + 0.45 log growth).
- **First-play urgency**: an unvisited child is valued at the parent's value - its search average blended toward its own evaluation the less of its prior mass has been visited (weight `min(1, visited_mass²)`) - minus `0.2 · sqrt(visited_mass)` (at the root: 0 in a full self-play search, 0.1 in the server; a cheap self-play search keeps 0.2).
- **Root policy**: raised to `1/T` with `T` decaying from 1.25 to 1.1 over the game (full self-play searches), then mixed 25/75 with Dirichlet noise over the legal moves whose concentrations sum to 10.83 - half spread uniformly, half by how far each move's log-policy sits above the mean.
- **Forced playouts**: in full self-play searches, a root child below `sqrt(2 · P · N)` visits is always explored next.
- **Score centre**: after a 10-visit pre-search, the dynamic score centre is re-read from the root's own search average (see **Search Utility**).
- **Play-selection values** (the policy target, and what the move is chosen from): each visited root child's visit count, with every child but the most stably explored one cut back to the visits PUCT would have given it had it known the final values (this removes forced playouts), the child with the best lower confidence bound (5 stdevs, with a small max-variance prior) raised enough to outrank every child it beats on LCB, and values below 1 visit (at most max/64) dropped. Self-play uses LCB only for the policy target, the server also for the move.
- **Move choice**: sampled from the play-selection values raised to `1/T`. In self-play, `T` decays from 0.75 to 0.15 with a half-life of 19 plies scaled by `19/sqrt(N)` (`interpolate_early()`); the server uses the request's `temperature` (0 = argmax).

## Self-Play

`SelfPlayRunner` (`ai/src/training/self_play.cpp`) follows KataGo's self-play structure: `--gamegen-batch-size` threads each play one game at a time, start a new one as soon as it ends, and search sequentially. Every search's network requests go to one `NNServer` (`ai/src/mcts/nn_server.cpp`, KataGo's `NNEvaluator` queue). A search builds its position's network input itself on the CPU - features, legal mask and, for the Transformer, every past ply's history features (`Evaluator::prepare()`, the counterpart of KataGo's `fillRow`) - then submits it and blocks. Each of `--nn-server-threads` server threads takes whatever is queued, up to `--nn-max-batch`, without waiting for a full batch, copies the inputs into one batch and onto the device (one copy per input tensor), runs the network and splits the outputs into per-position rows. With `--selfplay-gpus`, the server threads are spread over several GPUs, each GPU running its own copy of the model on whatever batch its threads take from the shared queue (KataGo's per-thread `cudaDeviceToUseThread<N>`); training stays on the training device. Batches form from however many searches are waiting at once, and the network runs while the other searches do their CPU work. Once an iteration's `--games-per-iteration` games have finished, each thread stops at its next ply boundary. Unfinished games carry over into the next iteration, continuing with the newly exported model, and keep their full move history - which the Transformer reads from move 0. Defaults (`SelfPlayParams`) are KataGo's:

- **Opening**: a new game's first `floor(Exp(1) · 0.04 · N)` moves are sampled straight from the network's policy - not searched, not trained on - for opening variety.
- **Cheap and full searches**: each later move gets a cheap search (`--cheap-visits`, no noise, no forced playouts, root policy temperature 1, root FPU reduction 0.2 as at non-root nodes) with probability `--cheap-prob`, otherwise a full one (`--full-visits`). Only full searches start with training weight 1; cheap ones start at 0.
- **Reduced visits**: once some player's search win term has stayed above 0.9 for 3 consecutive searched plies, a full search's visits move toward 100 and its weight toward 0.1, by the square of how far past 0.9 it is.
- **Surprise weighting** (at game end): the game's total weight is redistributed so 50% follows each ply's policy surprise (KL of the search target against the searched prior - cheap plies count only beyond 1.5× the game's average surprise) and 10% follows its value surprise (KL of the network's pair outcomes against the result smoothed over the shortest TD horizon, `c = 0.016` - see **Value Targets**), then every weight is rounded stochastically to an integer row count.

A game's trajectory (`<arch>_XXXXXX_traj.json`, one array of games per iteration) records `{"model": m, "plies": [...]}`, where `m` is the iteration of the model that finished it (-1 = uniform evaluator), an opening ply is `{"move": a}` and a searched ply is `{"move": a, "search": [pair outcomes], "weight": rows}` plus `"policy"` when `rows > 0`. That is everything `replay_trajectory()` needs to rebuild the game's record exactly, by replaying its moves.

## Value Targets

The rank head's target for a searched ply blends the final outcome with the search's own estimates at later plies - KataGo's TD value targets. With `V_k` the search-averaged pair-outcome probabilities at searched ply `k` (of `T`) and `Z` the final outcome (one-hot per pair), each horizon `f = 1/(1 + N·c)`, `c ∈ {0.176, 0.056, 0.016}` (about 64, 21 and 7 moves on 19x19), gives

```
TD_f(t) = Σ_{k=t}^{T-1} f(1-f)^(k-t) V_k  +  (1-f)^(T-t) Z
```

and the row's target is `(Z + TD_0.176 + TD_0.056 + TD_0.016) / 4`. KataGo trains its main value head on `Z` and a separate head per horizon, all with equal loss weight; cross-entropy is linear in its target, so a single head trained on the average receives exactly that combined loss. Because the target differs from ply to ply, a game's rows don't all share one label the network can memorize.

## Training Loop

`ai/src/train.cpp`, KataGo's `shuffle.py` + `train.py` + export steps, in memory:

- **Rows and the window**: a `GameRecord` stores each training row once with its row count; `ReplayBuffer` gives every row a global index and samples uniformly (without repeats within a batch) from the newest `window` rows. The window follows KataGo's power law - nothing until `--min-rows` rows exist (rows from the uniform evaluator count only up to `--min-rows`), then `window = min_rows + 0.4 · ((x^0.65 − s^0.65) / (0.65 · s^−0.35))` with `x` the usable rows past `--min-rows` plus the scale `s` (`--taper-window-scale`) - so it keeps growing, ever more slowly. Games that fall out of the window are dropped.
- **How much to train**: a sample bucket gains `--max-train-per-data` (4) samples per new row - the first time, one sample per existing row - capped at `--max-train-samples-per-iteration`; training takes batches while the bucket holds one.
- **Optimizer**: Adam (`--lr`, `--l2`), gradient-norm clipping at 1, with KataGo's learning-rate warmup (×1/20 rising to ×1 over the first 2M samples).
- **Weight average**: every `--swa-period-samples` samples, `average += (model − average) / --swa-scale` (KataGo's SWA); the average is what gets exported.
- **Losses**: policy cross-entropy against the play-selection target (scaled by `1/log(numStones·N+1)`), the ownership and point losses (see **Ownership Heads**), and the rank head's cross-entropy against its blended target (scaled by `1/log 3`), all with weight 1.

## Search Utility

The quantity MCTS maximizes (`ai/include/mcts/utility.h`) is KataGo's utility - win/loss plus a bounded score term - generalized to any number of players by summing over the player pairs the rank head scores. For player `p`, opponents `q`, `P` players, board size `N`:

```
win_p     = 1/(P-1) · Σ_q [ P(p ahead of q) − P(q ahead of p) ]
static_p  = 1/(P-1) · Σ_q (2/π) atan( d_pq / (2·sqrt(N)) )
dynamic_p = 1/(P-1) · Σ_q (2/π) atan( (d_pq − c_pq) / (scale·sqrt(N)) )
utility_p = win_p + static_factor · static_p + dynamic_factor · dynamic_p
```

`d_pq` is p's final total minus q's (komi-adjusted, and under `"territory"` capture-adjusted - `compute_player_totals()`), and `c_pq` is the search's dynamic score centre: the root's expected difference pulled 25% toward 0, but never more than `scale·sqrt(N)` away from it. Self-play uses `static_factor` 0, `dynamic_factor` 0.4, `scale` 0.5; the server 0.1, 0.3 and 0.75 with a 20% pull. For two players this is exactly KataGo's `(W − L) + static + dynamic` score utility. Every term is antisymmetric in `(p, q)`, so utilities always sum to 0.

A terminal position uses its exact outcomes and differences (`exact_pair_eval()`). A network leaf takes `P(p ahead of q)` from the rank head (see **Rank Head**) and `d_pq` from the ownership heads' expected totals (`estimate_player_totals()`, see **Ownership Heads**). KataGo averages the atan over its score head's predicted spread; Goes has no score-spread head, so a leaf uses its expected difference alone.

The score term is what keeps the search playing once the result is decided: per point near the current score it is worth `0.4 · (2/π) / (0.5·sqrt(N))` - about 0.06 on 9x9 - and its centre follows the game, so the winning side still prefers a bigger win and the losing side keeps fighting.

Since `scoreRule` determines the totals the utility compares, it is a game-dynamics config like `forcedPassOnly`: it is part of `weak_equal()`'s checkpoint-matching fields, and games under different rules get separate checkpoint directories and `/move` sessions.

## Ownership Heads: Per-Location Stone/Territory Estimates

Rather than predicting the game outcome directly, the model predicts, for every board location, two independent distributions over "which stone type ends up occupying/holding this point when the game ends" - together called the **ownership** output, and stone-type indexed (1..num_stones) to match `ScoreData`/`BoardState::board` everywhere else in the codebase:

- **Stone estimate**: softmax over `num_stones + 1` channels (channel 0 = empty/no stone) - the model's belief about which stone type (if any) occupies this point at game end.
- **Territory estimate**: same channel layout (channel 0 = dame/occupied/no territory) - the model's belief about which stone type's territory this point is at game end.

Both come out of the same per-node features the policy head reads, via two independent `Linear→ReLU→Linear→softmax` heads (`stone_head`, `territory_head`), packaged as one `(2, N, num_stones+1)` ownership tensor (index 0 = stone estimate, index 1 = territory estimate).

**Training targets**: for each finished game, the final board (`BoardState::board`) and `ScoreData::territory_owner` (from the flood-fill territory computation - see `BoardState::count_score()`) are used *directly* as `stone_owner`/`territory_owner` (length-N, stone-type-indexed) ground truth - no player mapping. These are stored once per game in `GameRecord` and shared by all of its rows.

**Loss** (`ai/src/train.cpp`):
- *Stone loss* / *Territory loss*: per-location squared error for each estimate - the predicted `(N, num_stones+1)` distribution against the true one-hot ownership (`stone_owner`/`territory_owner`, one-hot expanded via `torch::one_hot`), summed over the `num_stones+1` channels at each location (not averaged - averaging over channels as `torch::mse_loss` would do by default shrinks the loss too much to carry useful gradient signal), then averaged over locations and the batch: `(stone_estimate - one_hot(stone_owner))^2 .sum(channels) .mean(locations, batch)` (and the same for territory).
- *Point loss*: raw per-stone-type point total (no player aggregation) - the difference between `estimate_stone_points(ownership, score_rule)` (the model's own expected total per stone type) and the same function applied to a one-hot ground-truth "ownership" tensor built from `stone_owner`/`territory_owner`, scaled by `num_stones / N` (board size) before squaring - the raw difference can be as large as `N`, which would otherwise dwarf the per-location Stone/Territory losses - then summed over stone types and averaged over the batch: `((predicted_points - actual_points) / N * num_stones)^2 .sum(stone types) .mean(batch)`.

**The search reads these heads through `estimate_player_totals()`** (`ai/include/model/evaluator.h`): `estimate_stone_points()` (the same function the point loss above uses) gives the expected per-stone-type points under the state's `score_rule` - combined exactly like `compute_points()` does (stones only, territory only, or their sum) - which are credited to players via `stone_to_player_map` (a small membership-matrix matmul, mirroring `compute_player_totals()`) and have komi and, under `"territory"`, captures so far added. Differences of these expected totals are the score term's `d_pq` (see **Search Utility**); the win term comes from the rank head instead (see **Rank Head**) - the ownership heads can't supply it. A network batch only ever holds positions from one game config, so a single rule/map per batched call is correct.

## Rank Head: Pairwise Outcome Probabilities

Every architecture has a third head, `RankHead` (`ai/include/model/rank_head.h`), that predicts the game's final ranking directly. For every unordered player pair `(p, q)` with `p < q` - in the order `(1,2), (1,3), ..., (1,P), (2,3), ..., (P−1,P)` (`rank_pairs()`) - it outputs three probabilities: **p finishes ahead of q**, **they tie**, **q finishes ahead of p**. A 2-player game has one pair, i.e. ordinary win/draw/loss.

**Architecture**: the trunk's final per-node hidden states (the same `(B, N, hidden)` tensor the policy and ownership heads read) are mean- and max-pooled over the board, concatenated, and passed through `Linear(2·hidden → hidden) → ReLU → Linear(hidden → 3·pairs)`, with a softmax over each pair's 3 outcomes. It is returned as `ModelOutput::rank`, shape `(B, num_rank_pairs(P), 3)` (`ai/include/model/model_output.h`).

**Training target**: the blended pair-outcome distribution of **Value Targets**, built from the game's exact final outcome (`compute_player_totals()` of the final position, a player no stone credits ranked on komi alone) and the search's pair-outcome averages along the game. **Loss** (`compute_losses()`, `ai/src/train.cpp`): cross-entropy against that target, averaged over pairs and the batch and scaled by `1/log 3`, printed as `rank=` in each iteration's loss line.

**How MCTS uses it**: the rank head gives the search utility's win term - `win_p = 1/(P−1) · Σ_q (P(p ahead of q) − P(q ahead of p))`, the exact expectation of the pairwise comparisons a finished game's win term counts (see **Search Utility**).

**Why the win term can't be derived from the ownership head**: what the search needs is the *expected* outcome, `E[sign(Δ)] = P(win) − P(loss)`, where `Δ` is a pair's final score difference. The ownership heads only give per-location marginal distributions, and so the *expected* difference `E[Δ]`. Neither is enough:
- **The sign of an expectation is not the expectation of the sign.** Plugging `E[Δ]` into the win term gives `sign(E[Δ])`, which is ±1 the moment the expected leader changes. A position expected to finish +0.5 points ahead with a wide spread of outcomes is close to a coin flip, but `sign(E[Δ])` scores it exactly like a certain win.
- **Per-location marginals don't determine `P(win)`.** That needs the distribution of `Δ`, and the locations are strongly correlated - a group lives or dies as a whole, and one capture flips many points at once. Marginals carry no correlation information, and assuming independence badly underestimates the spread, so a win probability built from them is overconfident.

Measured on a 9x9 CNN run that used `sign(E[Δ])` as the leaf value, both failure modes showed up directly. At ply 10, the predicted winner differed between a position's children in 83% of positions, so child values swung by the full rank step on what was nearly a coin flip (the value picked the actual winner only 52% of the time before ply 10); the search treated those swings as certainties, and its chosen move changed in 36% of positions when given 8× more simulations. By ply 70, every child agreed on the winner in 93% of positions, and child values differed by a median of 0.002 - far below MCTS's exploration bonus, so the search could no longer rank moves at all. A head trained with cross-entropy (a proper scoring rule) learns calibrated probabilities instead: near 0 net margin when the result is uncertain, and changing smoothly as moves actually change the odds. The blended targets keep it from memorizing each game's single result, and the score term keeps moves distinguishable once `P(win)` saturates.

## Notes

- The Transformer reads its history set only through cross-attention (see **Transformer Architecture**, above), whose length-1 query costs `O(T)` per layer for `T` past plies. Self-attention over the set (query=key=value over the full `(T, D)` sequence) would produce an `O(T²)` attention-weight tensor per layer, making peak GPU memory grow *quadratically* with history length - on `forcedPassOnly` boards with no move bound, a single unusually long game entering self-play or the training window could exhaust memory. Measured: with a self-attention stack over the history set, peak memory fit `peak_MB ≈ 338 + 0.0216·Tmax²` within ~9% across a 3x range of iteration counts, while the cross-attention-only architecture uses ~3.6-4x less peak memory at matched history length and batch size, growing linearly.
- With komi 0, both passing at the start is an exact draw. A KataGo 9x9 run at komi 0 (KataGo's own engine, Goes' 9x9 rules) learned to pass early - 30-46% of its self-play games were draws - during its first ~3,000 trained-net games, and grew out of it with more data (no draws after ~25k games). A Goes run at komi 0 may go through the same early phase.

`BoardState::game_over()` covers `maxPlies` truncation directly via a live check (`ply_count() >= max_plies`, alongside a live resigned-players check and `MoveInfo.all_passed`, which only covers the consecutive-passes cause - mirroring `shared/boardState.ts`), so MCTS treats a state that hit its own `maxPlies` as terminal like any other game-over state, scoring it exactly (`exact_pair_eval()`) rather than with the model — once `game_over()` is true, `make_move()` always refuses further moves, so the recorded score genuinely is final rather than a heuristic snapshot of a game that could still continue.
