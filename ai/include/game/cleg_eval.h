#pragma once
#include "game/board_config.h"

// Mirrors shared/clegEval.ts's own public surface (typecheckClegAsBoard/buildBoardFromCleg on the TS
// side; typecheck_cleg_as_board lives in game/cleg_check.h instead here - see that header's own top
// comment for why this port's own file layout differs slightly). `ClegProgram` here is
// forward-declared only, same pimpl-idiom reasoning as game/cleg_parser.h (see its own top comment).
//
// The builtin-function table and evaluator internals that back this (BUILTIN_FUNCTIONS/eval_expr/
// etc.) are declared in game/cleg_eval_internal.h instead, included only by game/cleg_eval.cpp and
// game/cleg_check.cpp - keeping THIS header (the one train.cpp/server.cpp actually include) free of
// any dependency on the AST/type-system internals in game/cleg_base.h, exactly like the original
// single-file cleg.h's own design intent (nothing outside cleg's own files ever inspects a parsed
// program's AST) despite cleg's own logic now spanning four files instead of one.
struct ClegProgram;

// Mirrors shared/clegEval.ts's buildBoardFromCleg() - typecheck_cleg_as_board (game/cleg_check.h),
// then evaluate and unwrap the resulting egr. The one entry point every GameConfig::board_descr ->
// BoardConfig call site (train.cpp, server.cpp) uses.
BoardConfig build_board_from_cleg(const ClegProgram& program);
