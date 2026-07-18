// Registration/login backing store: an append-only JSON-Lines file of
// { name, salt, hash } records, loaded into memory by loadUserStore(). Passwords
// are never stored or logged in readable form - only a per-user salt and the
// scrypt hash of (password, salt).
//
// Stateless module: every function takes the state it needs as an argument.
// Loading (loadUserStore) is the caller's job (see wsServer.ts's attachWebSocket,
// the one place server/src wires real arguments into these stores).

import { scrypt, randomBytes, timingSafeEqual } from 'node:crypto';
import { promisify } from 'node:util';
import * as fs from 'node:fs';
import * as path from 'node:path';

const scryptAsync = promisify(scrypt) as (password: string, salt: string, keylen: number) => Promise<Buffer>;

export interface Credentials { salt: string; hash: string; }

export interface UserStoreState {
    filePath: string;
    // null = registration in-flight (reserved, not yet hashed/persisted) - see registerUser().
    users: Map<string, Credentials | null>;
}

// Reads+parses the registration file under `dataDir` into a fresh UserStoreState.
export function loadUserStore(dataDir: string): UserStoreState {
    const filePath = path.join(dataDir, 'registration');
    const users = new Map<string, Credentials | null>();
    fs.mkdirSync(dataDir, { recursive: true });
    if (fs.existsSync(filePath)) {
        const lines = fs.readFileSync(filePath, 'utf8').split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            try {
                const { name, salt, hash } = JSON.parse(trimmed);
                if (typeof name === 'string' && typeof salt === 'string' && typeof hash === 'string')
                    users.set(name, { salt, hash });
                else
                    console.warn('[userStore] malformed registration line (missing fields):', trimmed);
            } catch {
                console.warn('[userStore] malformed registration line (bad JSON):', trimmed);
            }
        }
    }
    return { filePath, users };
}

async function hashPassword(password: string): Promise<Credentials> {
    const salt = randomBytes(16).toString('hex');
    const hash = (await scryptAsync(password, salt, 64)).toString('hex');
    return { salt, hash };
}

async function verifyPassword(password: string, salt: string, hash: string): Promise<boolean> {
    const candidate = await scryptAsync(password, salt, 64);
    const stored = Buffer.from(hash, 'hex');
    if (candidate.length !== stored.length) return false;
    return timingSafeEqual(candidate, stored);
}

// Registers a new user, appending one JSON-line to state.filePath. Returns an
// error (without touching the file) if the name contains whitespace, is
// already taken, or another registration for the same name is currently in flight.
export async function registerUser(state: UserStoreState, name: string, password: string): Promise<{ ok: true } | { ok: false; error: string }> {
    if (/\s/.test(name)) return { ok: false, error: 'Username cannot contain spaces' };
    if (state.users.has(name)) return { ok: false, error: 'Username already exists' };
    // Reserve the name synchronously, before any await, so a second concurrent
    // registerUser() call for the same name sees this reservation immediately
    // rather than racing past the check above while this one is still hashing.
    state.users.set(name, null);
    try {
        const creds = await hashPassword(password);
        await fs.promises.appendFile(state.filePath, JSON.stringify({ name, ...creds }) + '\n');
        state.users.set(name, creds);
        return { ok: true };
    } catch (e) {
        state.users.delete(name);  // release the reservation so the name can be retried
        throw e;
    }
}

// Whether `name` is a registered account - a completed registration only;
// an in-flight one (null, see UserStoreState's doc comment) doesn't count yet.
export function userExists(state: UserStoreState, name: string): boolean {
    return !!state.users.get(name);
}

// Verifies a login attempt against the stored credentials for `name`.
export async function verifyLogin(state: UserStoreState, name: string, password: string): Promise<boolean> {
    const creds = state.users.get(name);
    if (!creds) return false;  // never registered, or registration still in-flight
    return verifyPassword(password, creds.salt, creds.hash);
}
