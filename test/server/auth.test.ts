import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { startTestServer, connect, type TestServer } from './testServer.ts';

let server: TestServer;

before(async () => { server = await startTestServer(); });
after(async () => { await server.close(); });

test('register then login round-trips', async () => {
    const client = await connect(server.url);
    const reg = await client.req<{ name: string; finishedGames: unknown[] }>('REGISTER', { name: 'alice', password: 'hunter2' });
    assert.equal(reg.name, 'alice');
    assert.deepEqual(reg.finishedGames, []);
    // REGISTER auto-logs-in this connection - close it (and wait for the
    // server to actually process that close) before a second connection logs
    // in as the same user, or LOGIN sees it as still active and rejects with
    // 409 ("already logged in elsewhere").
    await client.close();

    const client2 = await connect(server.url);
    const login = await client2.req<{ name: string }>('LOGIN', { name: 'alice', password: 'hunter2' });
    assert.equal(login.name, 'alice');
    await client2.close();
});

test('login rejects a wrong password', async () => {
    const client = await connect(server.url);
    await client.req('REGISTER', { name: 'bob', password: 'correct-horse' });
    await client.close();

    const client2 = await connect(server.url);
    await assert.rejects(
        client2.req('LOGIN', { name: 'bob', password: 'wrong-password' }),
        (e: any) => e.statusCode === 401,
    );
    await client2.close();
});

test('registering an existing name is rejected', async () => {
    const client = await connect(server.url);
    await client.req('REGISTER', { name: 'carol', password: 'pw1' });
    await assert.rejects(
        client.req('REGISTER', { name: 'carol', password: 'pw2' }),
        (e: any) => e.statusCode === 409,
    );
    await client.close();
});

test('logging in from a second connection while already logged in is rejected', async () => {
    const client = await connect(server.url);
    await client.req('REGISTER', { name: 'dave', password: 'pw' });

    const client2 = await connect(server.url);
    await assert.rejects(
        client2.req('LOGIN', { name: 'dave', password: 'pw' }),
        (e: any) => e.statusCode === 409,
    );
    await client.close();
    await client2.close();
});

test('flogin takes over the existing connection, which receives auth/kicked before closing', async () => {
    const client = await connect(server.url);
    await client.req('REGISTER', { name: 'erin', password: 'pw' });

    const kicked = new Promise<{ name: string }>(resolve => client.onEvent('auth/kicked', resolve));

    const client2 = await connect(server.url);
    const flogin = await client2.req<{ name: string }>('FLOGIN', { name: 'erin', password: 'pw' });
    assert.equal(flogin.name, 'erin');

    const kickedMsg = await kicked;
    assert.equal(kickedMsg.name, 'erin');
    await client2.close();
});
