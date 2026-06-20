## AI Training Pipeline

Self-play training pipeline for Goes using a Graph Neural Network (GNN) guided by Monte Carlo Tree Search (MCTS), in the style of AlphaZero.

Because Goes supports arbitrary board topologies (rectangular, triangular, cubical, hypercubical, twisted-square), a standard CNN cannot generalize across them. The GNN operates directly on the board's adjacency graph and works for all topologies without modification.

## Differences from the TypeScript engine

`shared/boardConfig.ts` stores node positions (`pos`) and a bounding box (`boardDimension`) for the canvas renderer. The Python `BoardConfig` omits both — only the adjacency matrix and node count are needed for gameplay and training.

As a result, `board_config.py` is also simpler: `_make` takes only `adj`, all factory functions skip position computation entirely, and `_tilted_disconnected_square_board` drops its `gap` parameter (which was purely visual).

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
│   │   └── gnn.{h,cpp}                 MessagePassingGNN (policy + value heads)
│   ├── mcts/
│   │   └── mcts.{h,cpp}                AlphaZero-style MCTS
│   └── training/
│       ├── self_play.{h,cpp}           Generate training records from self-play games
│       └── replay_buffer.{h,cpp}       Circular buffer of game records
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
   - Select: C++ / LibTorch / your OS / CUDA version or CPU
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
# For CUDA 11.x use "cuda-11"; package names for other versions may differ.
sudo apt-get install libcudnn9-cuda-12 libnccl2 libcusparselt0 libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
```

Then build

```bash
cmake -S ai -B ai/build \
      -DCMAKE_PREFIX_PATH="~/include/libtorch" \
      -DCMAKE_BUILD_TYPE=Release
```

If error message shows that certain shared libraries of `libcupti` and/or `nvshem` cannot be found, you need to set `-DCUPTI_DIR` and/or `-DNVSHMEM_DIR` to the directories where those libraries were installed. To find them:
```bash
find /usr /usr/local -name "libcupti*" 2>/dev/null   # use the directory containing this file
find /usr -name "libnvshmem*" 2>/dev/null            # use the directory containing this file
```

```bash
cmake -S ai -B ai/build \
      -DCMAKE_PREFIX_PATH="~/include/libtorch" \
      -DCUPTI_DIR="/usr/local/cuda/extras/CUPTI/lib64" \
      -DNVSHMEM_DIR="/usr/lib/x86_64-linux-gnu/nvshmem/12" \
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
export OMP_NUM_THREADS=8  # number of CPU threads for parallel MCTS select; defaults to all cores
ai/build/goes_train --no-forced-pass-only --board rect 9 9 --verbosity 1 --linear-move-bound 1.5
```

For higher-end training devices,

```bash
export OMP_NUM_THREADS=8  # number of CPU threads for parallel MCTS select; defaults to all cores
ai/build/goes_train --no-forced-pass-only --board rect 9 9 --verbosity 1 --linear-move-bound 1.5 --gamegen-batch-size 256 --iterations 65536 --save-every 16 --self-play-games 64
```

**Windows:**

```powershell
$env:OMP_NUM_THREADS = "8"  # number of CPU threads for parallel MCTS select; defaults to all cores
ai\build\Release\goes_train --no-forced-pass-only --board rect 9 9 --verbosity 1 --linear-move-bound 1.5
```

Checkpoints are saved as `ai/checkpoints/ckpt_XXXXXX.pt` and loaded automatically on resume.

## Profiling Training (Linux)

Build with `RelWithDebInfo` to keep optimizations while embedding debug symbols for readable stack traces:

```bash
cmake -S ai -B ai/build \
      -DCMAKE_PREFIX_PATH="~/include/libtorch" \
      -DCUPTI_DIR="/usr/local/cuda/extras/CUPTI/lib64" \
      -DNVSHMEM_DIR="/usr/lib/x86_64-linux-gnu/nvshmem/12" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build ai/build --config RelWithDebInfo
```

**`perf` (low overhead, recommended):**

```bash
# May need a while to collect data after the program finishes
perf record -g --call-graph dwarf \
    ai/build/goes_train --no-forced-pass-only --board rect 9 9 --verbosity 2 --linear-move-bound 1 --iterations 3 --self-play-games 10 --gamegen-batch-size 5 --num-simulations 10

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
    ai/build/goes_train --no-forced-pass-only --board rect 9 9 --verbosity 2 --linear-move-bound 1.5 --iterations 3 --self-play-games 4 --gamegen-batch-size 2 --num-simulations 5

callgrind_annotate --auto=yes callgrind.out | less
```

Callgrind reports exact instruction counts per source line, which is useful for pinpointing hot paths inside tight loops.

**Limiting run time:** pass `--iterations N` to cap the run, or wrap the binary with `timeout`:

```bash
timeout 120s perf record -g --call-graph dwarf ai/build/goes_train --iterations 999
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
    - `board_type`: `"rect"` | `"rectd"` | `"cub"` | `"hcub"` | `"tri"` | `"twsq"` | `"gtsq"`
    - `board_args`: integer dimensions (e.g. `[9, 9]` for a 9×9 rect board)
    - `num_stones`, `num_players`, `turn_stone_list`, `stone_to_player_map`, `forced_pass_only`
    - `moves`: full move history as an array — each entry is a board index (0-based) or `null` for pass
    - `board`: current stone array (length N) — used to verify the replayed state matches the client
    - `session_id`: opaque string returned by a previous response; omit, `""`, or `null` for a new session
    - `num_simulations` _(optional)_: overrides the server default
  - **Response:** `{move, policy, value, session_id}`
    - `move`: board index (0-based) or `null` for pass
    - `policy`: MCTS visit distribution over all N+1 actions
    - `value`: estimated value for the current player
    - `session_id`: include in the next request for this game to reuse the cached board state
  - The server caches a live `BoardState` per `session_id`. On each request it finds the longest common prefix of the stored and incoming move lists and retracts/advances the cached state when cheaper than a full replay, then verifies the result against the `board` field.
  - The model for each game config is loaded lazily on first request and cached.
- `GET /health` — returns `{status, loaded_tags, device}`.

The `GOES_CHECKPOINT_DIR` environment variable overrides the checkpoint directory, and `GOES_NUM_SIMS` overrides the default number of MCTS simulations.

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--board <type> [args]` | `rect 9 9` | Board type (rect/rectd/cub/hcub/tri/twsq/gtsq) |
| `--num-stones N` | `2` | Number of stone types |
| `--num-players N` | `2` | Number of players |
| `--iterations N` | `200` | Training iterations |
| `--self-play-games N` | `10` | Games to complete before each training step |
| `--gamegen-batch-size N` | `25` | Games generated in parallel (pool size) |
| `--num-simulations N` | `200` | MCTS sims per move |
| `--linear-move-bound F` | _(none)_ | End games after k×N plies |
| `--train-steps N` | `64` | Gradient steps per iteration |
| `--batch-size N` | `128` | Training batch size |
| `--hidden-dim N` | `128` | GNN hidden dimension |
| `--num-layers N` | `9` | GNN message-passing layers |
| `--save-every N` | `10` | Save a checkpoint every N iterations |
| `--checkpoint-dir PATH` | `ai/checkpoints` | Checkpoint directory |
| `--verbosity N` | `1` | 0=silent, 1=per-game, ≥2=per-ply |

## How It Works

1. **Self-play**: the current model plays games against itself. Each move is chosen by running MCTS simulations guided by the GNN's policy and value estimates.
2. **Record collection**: each ply stores `(features, MCTS visit distribution, game outcome)`.
3. **Training**: mini-batches are sampled from a replay buffer. The GNN is trained to predict the MCTS visit distribution (policy head, cross-entropy) and the game outcome (value head, MSE).
4. **Iteration**: the updated model is used for the next round of self-play.

## Reward

The value target for every ply is a sum of a rank-based component and a stone-fraction component, both in (−1, 1):

```
rank_reward  = (# opp with fewer stones − # opp with more stones) × 2 / (2P−1)   [0 if P=1]
stone_reward = tanh(stone_fraction − 1/P) / (2P−1)
reward       = rank_reward + stone_reward
```

where `stone_fraction = (player's stones) / (total stones)`, or 0 when the board is empty, and `P` is the number of players. The `[-1, 1]` interval is divided into `2P−1` equal sub-intervals; the rank reward places each player at the midpoint of the sub-interval corresponding to their rank, and the stone reward adds a continuous signal within one half-interval width. For two players the combined reward provably fits in `(−1, 1)` (the Tanh bound); for more players the range is smaller. This design keeps rewards within the GNN value head's Tanh output range at all player counts.

The same formula is computed for terminal nodes inside MCTS via `compute_player_rewards()`. The GNN value head outputs one value per player (player-ID order 1..P, with Tanh), so MCTS backup has per-player rewards for every simulation — not just for the player to move at the leaf. Backup does not negate: each node looks up its own player's reward from the map, so Q-values at every node reflect that node's player's outcome directly.