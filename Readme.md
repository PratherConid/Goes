## Introduction
This project implements several go variants, where the board topology, turn order, scoring, and placement rules can all be modified. Games are configured via `GameConfig` (`shared/types.ts`); the supported options, in the order they appear there, are:
* **Board topology**: `boardType`/`boardArgs` select the board's shape and dimensions - rectangular (`rect`), rectangular with periodic diagonal connections (`rectd`), cubical/hypercubical (`cub`/`hcub`), triangular (`tri`), or squares rotated 45° and tiled, either edge-connected (`twsq`) or corner-glued (`gtsq`).
* **Stone types and players**: `numStones` and `numPlayers` set how many distinct stone colors and players are in the game - a stone color need not map 1:1 to a player (see stone-to-player map, below).
* **Turn list**: `turnList` is the ordered, repeating sequence of turns. Each entry says which player moves, which stone color(s) they may choose among that turn (more than one may be offered at once), which colors are protected that turn (can never be captured, even at zero liberties), and which are friendly that turn (don't count as blocking anyone's liberties).
* **Player stone placement limit**: `playerStonePlaceLimit` caps how many times each player may ever place each stone color over the course of the game.
* **Global stone placement limit**: `globalStonePlaceLimit` caps how many times each stone color may ever be placed in total, summed across all players.
* **Stone-to-player map**: `stoneToPlayerMap` determines which player(s) each stone color scores points for - a color can score for several players at once (each gets its full point value, not split), or for none.
* **Forced pass only**: If enabled, a player is only allowed to pass when there are no legal moves on the board. This fundamentally changes the concept of life and death - for example, it forces players to put stones into the eyes of their own groups, allowing their opponent to capture their group. The game ends once every player in the turn list has passed consecutively. We use the SuperKo rule to ensure termination, which disallows repeating game states.
* **Score rule**: `scoreRule` selects how points are counted at game end - stones on the board only, territory only, both (Chinese-style/area scoring, the default), or territory plus each player's captured-stone count (Japanese-style).
* **Komi**: `komi` is a fixed per-player point handicap added before determining the winner.
* **Ko rule**: `koRule` selects the superko variant - `situational` (a repeated board position is only illegal when it's also the same player's turn as the earlier occurrence) or `positional` (any repeated position is illegal, regardless of whose turn it is).
* **Allow suicide**: `allowSuicide` controls whether a move that would leave the mover's own group at zero liberties is legal - if enabled, the move is legal and immediately self-captures that group instead of being rejected.
* **Max plies**: `maxPlies` sets a hard cap on the total number of plies before the game automatically ends, regardless of whether it would otherwise continue.
----
Here are the supported gameplay features
* **Online Registration**: `register <name> <password>` creates an account and logs in as it. Passwords are never stored in plain text - only a per-account random salt and the scrypt hash of the password are persisted. `login <name> <password>` authenticates an existing account, and is rejected if that account is already logged in from another connection; `flogin` instead takes over, forcibly disconnecting the other connection (which is notified before being closed).
* **Online Game**: Before creating a game, slots can be pre-assigned with `sol <slot>` (yourself) or `soe <slot> [sim] [t]` (a server-side AI engine) - see Player Modes, below; unassigned slots stay open for others. `newo` creates the game and prints its ID; other players join with `joino <ID>`. Every client watching a game, whether playing or spectating, receives the same broadcast of moves, resignations, and start/pending events; `swl`/`swo`/`swf` switch the active view between any local, online, or finished game the client is tracking without losing the others, and a dropped connection can rejoin an in-progress game and catch up on its state.
* **AI Engine**: A slot assigned `soe` is played automatically by the C++ MCTS engine (see `ai/Readme.md`) instead of a human. The server spawns one AI engine process per active game that needs one, on demand, and proxies its moves over HTTP - advancing the engine's moves back-to-back until a human slot's turn comes up or the game ends, at which point the process is released. `soe`'s optional `[sim] [t]` arguments override that slot's MCTS simulation count and sampling temperature for the rest of the game.
* **Resignation**: Players can resign in online games. If all players but one have resigned, the player that's left wins. If some players have resigned but more than two players remain, the game continues as if the resigned players are left, and the resigned players' moves are filled in with pass moves.
----
When typing commands in the command input bar, make sure the input method is set to English.

## Project Structure

```
Goes/
├── shared/     Pure TypeScript game logic (no browser or Node dependencies)
├── src/        Browser client (canvas renderer, UI)
├── server/     Node.js backend: WebSocket API (online games + AI proxy) and static file serving
└── ai/         C++ self-play training pipeline (GNN + MCTS) — see ai/Readme.md
```

## Running Locally

* Prerequisites: Node.js 18+, npm
* Start the client dev server (hot-reload, no backend needed):
  ```
  npm install
  npm run dev
  ```
  Open `http://localhost:5173` in browser
* The client talks to the main server over a single WebSocket (`/ws`) for both the
  AI engine and online multiplayer. To enable those in the dev client, run the main
  server in a separate terminal (build the engine first — see `ai/Readme.md`):
  ```
  cd server
  npm install
  npm run dev
  ```
  The main server listens on port 3000 and **spawns one AI engine process per game on demand**,
  proxying AI requests to it over HTTP. The Vite dev server proxies the `/ws`
  WebSocket to `localhost:3000`. (`npm run ai` / `npm run ai-win` still launch the
  engine standalone for manual testing, but the dev client reaches it through the
  main server.)
* Start the full backend server (serves the built client and the WebSocket):
  ```
  npm run build
  cd server
  npm install
  npm run dev
  ```
  Open `http://localhost:3000` in browser. The backend spawns AI engine processes on demand as games are created.

## Deploying as a Web Service

* Build the client bundle:
  ```
  npm install
  npm run build
  ```
  This produces a `dist/` folder.
* Install server dependencies and start the server:
  ```
  cd server
  npm install
  npm start
  ```
  The server listens on port 3000 (override with the `PORT` environment variable) and serves `dist/` as static files.

## Player Modes

Each slot in a game can be assigned one of several player modes. The mode is configured before starting a game using the `sol` and `soe` commands.

| Mode | Who issues move | Display | Description |
|------|-----------------|---------|-------------|
| `local` | this client's user | `L` | A human player at this client. Moves are submitted by clicking the board. |
| `server` | a remote client | `S` | A human player on a different client (online games). |
| `serverEngine` | the server AI | `E` | A server-side AI engine plays this slot automatically. Moves advance without any client input. |
| `client` | this client's user | `-` | Internal label the server assigns to `local` slots after game creation. Not visible to users. |

`N` indicates an unassigned (pending) slot.

### Online game setup commands

Before running `newo` (new online game), use these commands to pre-assign slots:

- `sol <slot>` — assign slot to yourself (local human player)
- `soe <slot> [simulations] [temperature]` — assign slot to a server-side AI engine; `simulations` and `temperature` default to the current engine settings

Slots not assigned via `sol`/`soe` remain open for other players to join with `joino`.

If no `sol`/`soe` commands are issued, the creator joins as a pure observer and all slots wait for remote players.

## Notable Supported Go Variants

### Regular Go

During each turn, black plays first, and white plays second. A player can pass at any time. A non-pass move is legal if and only if it does not kill the player's own group, and the game state after the move is not identical to a previous game state.

### NoPass Go

During each turn, black plays first, and white plays second. A player can only pass if there are no legal non-pass moves, and the game ends when both players pass. A non-pass move is legal if and only if it does not kill the player's own group, and the game state after the move is not identical to a previous game state.

## Gameplay
### NoPass Go on 3x3 Board
* Basic endgames:
  * White has 8 stones on the board, and the remaining space is the location of the first move by black
  * Player A has 7 stones on the board, the two remaining spaces are empty and not connected, and player A has no legal moves due to game state collision
  * Player A has 7 stones on the board, player B has 1 stone on the board, and both players have no legal moves due to game state collision