import { BoardState } from '@shared/boardState.js';
import { rectangularBoard } from '@shared/boardConfig.js';
import { Renderer } from './renderer.js';

const config = rectangularBoard(9, 9);
const game   = new BoardState(2, 2, [1, 2], {1:1, 2:2}, true, new Array(config.N).fill(0), config);
const renderer = new Renderer(game);
renderer.init();
