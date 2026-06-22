## Introduction
This project implements several go variants, where the board topology, turn rule, and pass rule are modified.
* If the "forced pass only" option is enabled, a player is only allowed to pass when there are no legal moves on the board. This fundamentally changes the concept of life and death. For example, the modified rule will force players to put stones into the eyes of their own groups, allowing their opponent to capture their group. The game ends when both players pass. We use the SuperKo rule to ensure termination, which disallows repeating game configurations.
* The turn rule can be changed by setting the "turn list", which is the list of players that places the stones in one turn. The length of the turn list is the number of plys in a turn.
----
When typing commands in the command input bar, make sure the input method is set to English.

## Project Structure

```
Goes/
├── shared/     Pure TypeScript game logic (no browser or Node dependencies)
├── src/        Browser client (canvas renderer, UI)
├── server/     Node.js/Express backend (API stubs, static file serving)
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
* To enable the AI engine in the dev client, also run the AI server in a separate terminal (after building it — see `ai/Readme.md`):
  ```
  npm run ai        # Linux/macOS
  npm run ai-win    # Windows
  ```
  The AI server listens on port 8765. The Vite dev server proxies `/api/ai` requests directly to it.
* Start the full backend server (serves the built client and proxies AI requests):
  ```
  npm run build
  cd server
  npm install
  npm run dev
  ```
  Open `http://localhost:3000` in browser. The backend automatically starts the AI engine on startup.

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