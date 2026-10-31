#pragma once
#include "game/cleg_base.h"

// Mirrors shared/clegCheck.ts. Unlike game/cleg_parser.h/game/cleg_eval.h, includes the real
// game/cleg_base.h rather than forward-declaring `ClegProgram`, since ClegType is returned by value.
//
// The TS clegCheck.ts <-> clegEval.ts circular import becomes game/cleg_check.cpp including
// game/cleg_eval_internal.h and game/cleg_eval.cpp including this header - a .cpp-to-.h dependency
// each way, never a header cycle.
ClegType typecheck_cleg(const ClegProgram& program);

// Mirrors shared/clegEval.ts's typecheckClegAsBoard() - declared here next to typecheck_cleg rather
// than in game/cleg_eval.h.
void typecheck_cleg_as_board(const ClegProgram& program);
