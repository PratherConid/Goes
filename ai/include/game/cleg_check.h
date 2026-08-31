#pragma once
#include "game/cleg_base.h"

// Mirrors shared/clegCheck.ts - the static type checker - see shared/clegBase.ts's own top comment
// for the cleg language itself and the four-file split this is one part of. `ClegType` is returned
// by value below, so (unlike game/cleg_parser.h/game/cleg_eval.h, which only ever hand `ClegProgram`
// around behind a pointer/reference) this header needs the real game/cleg_base.h, not just a forward
// declaration - `ClegProgram` itself is still only ever taken by const reference, so an external
// consumer including just this header still never needs to construct or store one directly (it comes
// from game/cleg_parser.h's own parse_cleg()).
//
// typecheck_cleg is declared here (not just typecheck_cleg_as_board, the one real public API
// function - see game/cleg_eval.cpp's own run_cleg_program, which needs the lower-level one directly)
// purely so game/cleg_eval.cpp can call it - mirrors shared/clegCheck.ts's own typecheckCleg being a
// real (named) export that only shared/clegEval.ts imports, never UI code.
//
// This file's own .cpp needs BUILTIN_FUNCTIONS/EGR_TYPE/is_set_elem_kind from
// game/cleg_eval_internal.h (not game/cleg_eval.h - see that header's own top comment for why they're
// split) - a real (non-type-only) mutual dependency with game/cleg_eval.cpp needing typecheck_cleg/
// typecheck_cleg_as_board from here. Safe because it's a .cpp-to-.h dependency in each direction, not
// a header-to-header cycle: neither header includes the other, only each one's own .cpp does.
ClegType typecheck_cleg(const ClegProgram& program);

// Mirrors shared/clegEval.ts's typecheckClegAsBoard() - throws if program doesn't type-check, or
// type-checks to something other than egr. Physically grouped with typecheck_cleg above (its only
// caller, besides game/cleg_eval.cpp's own build_board_from_cleg) rather than with the rest of the
// public API in game/cleg_eval.h, since this port's own file layout already had it here before this
// split.
void typecheck_cleg_as_board(const ClegProgram& program);
