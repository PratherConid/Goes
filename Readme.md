## Introduction
This project implements several go variants, where the board topology, turn order, scoring, and placement rules can all be modified. Games are configured via `GameConfig` (`shared/gameConfig.ts`); the supported options, in the order they appear there, are:
* **Board topology**: `boardDescr` is a program in *cleg*, the project's small board-construction language (`shared/clegBase.ts`, parsed by `shared/clegParser.ts` and evaluated by `shared/clegEval.ts`) - a board is built by a program rather than picked from a fixed list. Builtin constructors give the base shapes - e.g. `rectB(w, h)` rectangular, `rectdB(w, h, m)` rectangular with periodic diagonal connections, `cublatB(w, h, d)`/`hcubB(meshdim, [...])` cubical/hypercubical, `triB(w)` triangular, and `twsqB(w, h, g)`/`gtsqB(w, h, g)` for squares rotated 45° and tiled, either edge-connected or corner-glued - and further builtins subdivide, glue, centralize, truncate, randomly thin out, or take products of them. Ready-made programs live in `public/board_presets/*.cleg`, and the `board` command opens the current one in an editor popup.
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
* **Online Game**: Before creating a game, slots can be pre-assigned with `sol <slot>` (yourself), `soe <slot> [sim] [t]` (a server-side AI engine), or `soi <slot> <name>` (an invitation to a specific account) - see Player Modes, below; unassigned slots stay open for others. `newo` creates the game and prints its ID; other players join with `joino <ID>`. An invited player gets a popup and must accept before the game starts - if any invitee refuses, the game is cancelled and everyone involved is told. Every client watching a game, whether playing or spectating, receives the same broadcast of moves, chat messages, resignations, withdrawals, and start/pending events; `swl`/`swo`/`swf` switch the active view between any local, online, or finished game the client is tracking without losing the others, and a dropped connection can rejoin an in-progress game and catch up on its state.
* **AI Engine**: A slot assigned `soe` is played automatically by the C++ MCTS engine (see `ai/Readme.md`) instead of a human. The server spawns one AI engine process per active game that needs one, on demand, and proxies its moves over HTTP - advancing the engine's moves back-to-back until a human slot's turn comes up or the game ends, at which point the process is released. `soe`'s optional `[sim] [t]` arguments override that slot's MCTS simulation count and sampling temperature for the rest of the game.
* **Resignation**: Players can resign in online games. If all players but one have resigned, the player that's left wins. If some players have resigned but more than one player remains, the game continues as if the resigned players are left, and the resigned players' moves are filled in with pass moves.
* **Withdrawal**: A player can propose rewinding an online game - back to their own last move, or to an explicitly chosen ply. Every other non-resigned human player has to agree; the game is locked against moves and resignations while the vote is open, and a single refusal cancels the proposal and leaves the game untouched. Once everyone agrees, the rewind is broadcast to all observers.
----
When typing commands in the command input bar, make sure the input method is set to English.

## Project Structure

```
Goes/
├── shared/     Pure TypeScript game logic (no browser or Node dependencies)
├── src/        Browser client (canvas renderer, UI)
├── server/     Node.js backend: WebSocket API (online games + AI proxy) and static file serving
├── test/       Automated test suite (node:test via tsx) — `npm test`
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
  WebSocket to `localhost:3000`. (`npm run ai` / `npm run ai-win` launch the engine
  standalone for manual testing, but the dev client reaches it through the main
  server.)
* Start the full backend server (serves the built client and the WebSocket):
  ```
  npm run build
  cd server
  npm install
  npm run dev
  ```
  Open `http://localhost:3000` in browser. The backend spawns an AI engine process the first time a
  given game actually needs one, and releases it when that game ends.

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
  The server listens on port 3000 and serves `dist/` as static files. The port and the data
  directory are positional arguments to `server/src/index.ts` (`index.ts <port> <dataDir> <autoStart>`),
  passed by `server/package.json`'s `start`/`dev` scripts - edit them there to change either.

## Player Modes

Each slot in a game holds a `PlayerInfo` (`shared/types.ts`) whose `type` is one of the five modes
below. A slot is set up before the game starts, with the `sol`/`soe`/`soi` commands for an online
game (`addl`/`adde`/`addi` in random-order mode, see `tfpro`), and the server is the authority for
what each slot's mode ends up as once the game exists.

| Mode | Who issues moves | Display | Description |
|------|------------------|---------|-------------|
| `local` | this client's user | `⌂` | A human player at this client. Moves are submitted by clicking the board. Used by local games, and by `sol` while an online game is still being set up - the server rewrites such a slot to `client` (under your account name) when it creates the game, so a live online game never contains one. |
| `client` | a remote or local client's user | the account name | A human participant in an online game, identified by their account name. The server assigns this to your own `sol` slots, to whoever fills an open slot via `joino`, and to an invited player who accepts. |
| `serverEngine` | the server AI | `⚙` | A server-side AI engine plays this slot automatically, moving as soon as its turn comes up, with no client input. |
| `pendingInvitedOnline` | nobody yet | the invited name | An online slot reserved by `soi` for one specific account until they accept (it becomes `client`) or refuse (the game is cancelled). The slot is not claimable via `joino`, and a game holding one never starts. |
| `localEngine` | this client's AI calls | `⚙` | Like `serverEngine`, but in a *local* game: this client drives the engine itself, and auto-advances the slot's turn. Produced when a config with `serverEngine` slots is started as a local game. |

A slot with nothing assigned yet displays as just its player number.

### Online game setup commands

Before running `newo` (new online game), use these commands to pre-assign slots:

- `sol <slot>` — assign slot to yourself (human player at this client)
- `soe <slot> [simulations] [temperature]` — assign slot to a server-side AI engine; `simulations` and `temperature` default to the current engine settings
- `soi <slot> <name>` — invite the account `<name>` to that slot; they must be registered and currently online, or `newo` is rejected outright

Slots not assigned this way remain open for other players to join with `joino`.

If none of these commands are issued, the creator joins as a pure observer and all slots wait for remote players.

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