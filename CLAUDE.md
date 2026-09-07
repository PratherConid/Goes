# Project conventions

## C++ code that mirrors TypeScript

`ai/src`/`ai/include` (C++) independently reimplements much of `shared/` (TypeScript) - e.g.
`shared/boardConfig.ts`'s board-construction functions are mirrored in
`ai/src/game/board_config.cpp`.

**Mirror the TS algorithm/structure as closely as possible** - same helper functions, same
recursive shape, same variable roles - not just the same external behavior via a different
approach, even if that approach is simpler. Only deviate for a genuine correctness issue (e.g. a
data-type constraint the TS side doesn't have - `BoardConfig::embed` here is exact-integer only,
so a board whose TS positions are irrational needs a different embedding scheme) or a large
performance overhead. A deviation like that should be scoped as narrowly as possible - e.g. the
embedding/coordinate values may need to differ, but the surrounding algorithm that builds and
glues the board together should still mirror its TS counterpart's helper functions and structure.

When writing or editing C++ code that mirrors an existing TypeScript counterpart, keep the C++
comment succinct: point at the TS function it mirrors and describe only what's *different* (data
types, embedding, etc.) - do not re-explain behavior the TS source already documents in full. The
reader is expected to have the TS comment/implementation as the canonical reference.

## Editing this file

Do not modify CLAUDE.md unless the user has explicitly specified to do so.
