#pragma once
#include "game/board_config.h"
#include <memory>
#include <string>

// Mirrors shared/cleg.ts - "CLEG": a small typed language for describing boards. See that file's
// own top comment for the full grammar/semantics (types, operators, builtins, the multiProd
// algorithm, etc.) - this header intentionally exposes only the entry points every actual C++
// consumer (train.cpp/server.cpp, via GameConfig::board_descr) needs, unlike the TS side's own
// module exports: nothing outside game/cleg.cpp ever inspects a parsed program's AST, a ClegType,
// or a ClegValue directly (the TS side's UI does, e.g. unparseCleg for the Configure Board popup -
// no such consumer exists here), so every one of those stays a private implementation detail of
// cleg.cpp, and ClegProgram itself is forward-declared only (see game/cleg.cpp for its definition -
// a shared_ptr<ClegProgram> can be held/passed/destroyed with ClegProgram incomplete here, same
// pimpl-idiom guarantee any C++17 standard library provides for shared_ptr's own type-erased
// deleter).
struct ClegProgram;

// Mirrors shared/cleg.ts's parseCleg() - throws std::runtime_error if source doesn't follow cleg's
// grammar.
std::shared_ptr<ClegProgram> parse_cleg(const std::string& source);

// Mirrors shared/cleg.ts's typecheckClegAsBoard() - throws if program doesn't type-check, or
// type-checks to something other than egr.
void typecheck_cleg_as_board(const ClegProgram& program);

// Mirrors shared/cleg.ts's buildBoardFromCleg() - typecheck_cleg_as_board, then evaluate and unwrap
// the resulting egr. The one entry point every GameConfig::board_descr -> BoardConfig call site
// (train.cpp, server.cpp) uses.
BoardConfig build_board_from_cleg(const ClegProgram& program);
