// Finished-online-game backing store: an append-only JSON-Lines file of
// {id, finishedGame, observers} records, loaded by loadGameRecordStore() both as
// a username -> finished game IDs index and as the raw record list
// OnlineGameManager reconstructs its finishedGames map from.
//
// Stateless module: every function takes the state it needs as an argument.
// Loading (loadGameRecordStore) is the caller's job (see wsServer.ts's
// attachWebSocket, the one place server/src wires real arguments into these stores).

import * as fs from 'node:fs';
import * as path from 'node:path';
import { FinishedGame } from '@shared/types.js';

export interface LoadedRecord {
    id: string;
    finishedGame: FinishedGame;
    observers: Set<string>;
}

export interface GameRecordStoreState {
    filePath: string;
    // username -> set of finished game IDs they observed.
    finishedGamesByUser: Map<string, Set<string>>;
    // Raw parsed records, for OnlineGameManager to rebuild finishedGames from at startup.
    loadedRecords: LoadedRecord[];
}

function addToIndex(finishedGamesByUser: Map<string, Set<string>>, id: string, observers: Iterable<string>): void {
    for (const name of observers) {
        let ids = finishedGamesByUser.get(name);
        if (!ids) { ids = new Set(); finishedGamesByUser.set(name, ids); }
        ids.add(id);
    }
}

// Reads+parses the game record file under `dataDir` into a fresh GameRecordStoreState.
export function loadGameRecordStore(dataDir: string): GameRecordStoreState {
    const filePath = path.join(dataDir, 'gameRecords');
    const finishedGamesByUser = new Map<string, Set<string>>();
    const loadedRecords: LoadedRecord[] = [];
    fs.mkdirSync(dataDir, { recursive: true });
    if (fs.existsSync(filePath)) {
        const lines = fs.readFileSync(filePath, 'utf8').split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            try {
                const raw = JSON.parse(trimmed);
                if (typeof raw.id !== 'string' || !raw.finishedGame || !Array.isArray(raw.observers)) {
                    console.warn('[gameRecordStore] malformed game record line (missing fields):', trimmed);
                    continue;
                }
                const observers = new Set<string>(raw.observers);
                loadedRecords.push({ id: raw.id, finishedGame: FinishedGame.fromJSON(raw.finishedGame), observers });
                addToIndex(finishedGamesByUser, raw.id, observers);
            } catch {
                console.warn('[gameRecordStore] malformed game record line (bad JSON):', trimmed);
            }
        }
    }
    return { filePath, finishedGamesByUser, loadedRecords };
}

// Appends one JSON-line record of a finished game, then updates the in-memory
// index - only after the write succeeds, so a failed append never leaves a
// phantom in-memory-only entry.
export async function recordFinishedGame(state: GameRecordStoreState, id: string, finishedGame: FinishedGame, observers: Set<string>): Promise<void> {
    const record = { id, finishedGame: finishedGame.toJSON(), observers: [...observers] };
    await fs.promises.appendFile(state.filePath, JSON.stringify(record) + '\n');
    addToIndex(state.finishedGamesByUser, id, observers);
}

// Returns the IDs of finished games `userName` observed, or [] if none.
export function getFinishedGames(state: GameRecordStoreState, userName: string): string[] {
    return [...(state.finishedGamesByUser.get(userName) ?? [])];
}
