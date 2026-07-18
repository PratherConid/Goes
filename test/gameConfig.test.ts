// Regression tests for GameConfig/PlayerInfo copy + serialization, used for the
// online-game config payloads sent between client and server.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GameConfig, PlayerInfo } from '../shared/types.ts';

function sampleConfig() {
    const config = new GameConfig('rect', [9, 9], 2, 2, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], [[null, null], [null, null]], [null, null], { 1: new Set([1]), 2: new Set([2]) }, true, 'area', [0, 0], 'situational', false, null);
    config.players.set(1, new PlayerInfo('client', 'alice'));
    config.players.set(2, new PlayerInfo('serverEngine', 'Engine', 400, 1));
    return config;
}

test('GameConfig.copy() deep-clones players, stoneToPlayerMap, and turnList', () => {
    const original = sampleConfig();
    const copy = original.copy();

    copy.players.get(1)!.name = 'mutated';
    copy.stoneToPlayerMap[1]!.add(2);
    copy.turnList.push({ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] });
    copy.turnList[0]!.player = 2;
    copy.turnList[0]!.protected[0] = 1;

    assert.equal(original.players.get(1)!.name, 'alice');
    assert.deepEqual(original.stoneToPlayerMap[1], new Set([1]));
    assert.deepEqual(original.turnList, [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }]);
});

test('GameConfig.toJSON()/fromJSON() round-trips players and scalar fields', () => {
    const original = sampleConfig();
    const roundTripped = GameConfig.fromJSON(JSON.parse(JSON.stringify(original.toJSON())));

    assert.equal(roundTripped.boardType, original.boardType);
    assert.deepEqual(roundTripped.boardArgs, original.boardArgs);
    assert.equal(roundTripped.forcedPassOnly, original.forcedPassOnly);
    assert.equal(roundTripped.scoreRule, original.scoreRule);
    assert.deepEqual(roundTripped.komi, original.komi);
    assert.equal(roundTripped.players.size, 2);
    assert.equal(roundTripped.players.get(1)!.type, 'client');
    assert.equal(roundTripped.players.get(1)!.name, 'alice');
    assert.equal(roundTripped.players.get(2)!.emsim, 400);
    assert.deepEqual(roundTripped.stoneToPlayerMap, { 1: new Set([1]), 2: new Set([2]) },
        'stoneToPlayerMap survives the real JSON.stringify/JSON.parse wire path as equivalent Sets');
});

test('GameConfig.fromJSON() defaults scoreRule/komi/koRule/allowSuicide/maxPlies when absent', () => {
    const raw = {
        boardType: 'rect', boardArgs: [9, 9], numStones: 2, numPlayers: 2,
        turnList: [{ player: 1, stones: [1, 0], protected: [0, 0], friendly: [0, 0] }, { player: 2, stones: [0, 1], protected: [0, 0], friendly: [0, 0] }], stoneToPlayerMap: { 1: [1], 2: [2] }, forcedPassOnly: false,
    };
    const config = GameConfig.fromJSON(raw);
    assert.equal(config.scoreRule, 'area');
    assert.deepEqual(config.komi, [0, 0]);
    assert.equal(config.koRule, 'situational');
    assert.equal(config.allowSuicide, false);
    assert.equal(config.maxPlies, null);
});
