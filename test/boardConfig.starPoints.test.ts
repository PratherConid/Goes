// Regression test: computeStarPoints() must reproduce the traditional Go
// board star-point ("hoshi") layouts for common rectangular board sizes.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeStarPoints } from '../shared/boardConfig.ts';
import type { GameConfig } from '../shared/gameConfig.ts';
import { makeGameConfig, soloStoneTurns } from './boardStateHelpers.ts';

// computeStarPoints only reads boardDescr; the rest of the config is neutral filler.
function mkConfig(boardDescr: string): GameConfig {
    return makeGameConfig(boardDescr, soloStoneTurns(2));
}

test('9x9 rect board has 5 star points (4 corners at the 3-3 point + center)', () => {
    const points = computeStarPoints(mkConfig('rectB(9, 9);'));
    assert.equal(points.length, 5);
    const expected = [[-2, -2], [-2, 2], [2, -2], [2, 2], [0, 0]];
    for (const e of expected) assert.ok(points.some(p => p[0] === e[0] && p[1] === e[1]), `missing ${e}`);
});

test('13x13 rect board has 5 star points (4 corners at the 4-4 point + center)', () => {
    const points = computeStarPoints(mkConfig('rectB(13, 13);'));
    assert.equal(points.length, 5);
    const expected = [[-3, -3], [-3, 3], [3, -3], [3, 3], [0, 0]];
    for (const e of expected) assert.ok(points.some(p => p[0] === e[0] && p[1] === e[1]), `missing ${e}`);
});

test('19x19 rect board has 9 star points (4 corner + 4 edge + center)', () => {
    const points = computeStarPoints(mkConfig('rectB(19, 19);'));
    assert.equal(points.length, 9);
    const expected = [
        [-6, -6], [-6, 6], [6, -6], [6, 6],
        [-6, 0], [6, 0], [0, -6], [0, 6],
        [0, 0],
    ];
    for (const e of expected) assert.ok(points.some(p => p[0] === e[0] && p[1] === e[1]), `missing ${e}`);
});

test('a non-rect board has no star points', () => {
    assert.deepEqual(computeStarPoints(mkConfig('triB(9);')), []);
});

test('a board below every threshold has no star points', () => {
    assert.deepEqual(computeStarPoints(mkConfig('rectB(3, 3);')), []);
});
