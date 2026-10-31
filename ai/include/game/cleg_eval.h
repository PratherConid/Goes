#pragma once
#include "game/board_config.h"
#include <cstdint>

// Mirrors shared/clegEval.ts's public surface, except that typecheck_cleg_as_board lives in
// game/cleg_check.h. `ClegProgram` is forward-declared only, as in game/cleg_parser.h. The builtin
// table and evaluator internals are declared in game/cleg_eval_internal.h instead, so this header
// (the one train.cpp/server.cpp include) stays free of game/cleg_base.h's AST/type-system internals.
struct ClegProgram;

// Mirrors shared/clegEval.ts's buildBoardFromCleg(); used by train.cpp/server.cpp.
BoardConfig build_board_from_cleg(const ClegProgram& program, uint32_t seed);
