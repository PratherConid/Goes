#pragma once
#include "game/cleg_base.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Internal (cross-file-only) declarations shared between game/cleg_eval.cpp and game/cleg_check.cpp
// - never included by an external consumer (train.cpp/server.cpp only include game/cleg_parser.h and
// game/cleg_eval.h, both of which forward-declare ClegProgram and stay free of any dependency on
// game/cleg_base.h's own AST/type-system internals - see game/cleg_eval.h's own top comment).
//
// This exists only because C++'s #include model can't give one header two different visibility
// levels the way a TS module's own named exports can: shared/clegEval.ts exports BUILTIN_FUNCTIONS
// for shared/clegCheck.ts to import, right alongside typecheckClegAsBoard/buildBoardFromCleg, with no
// equivalent leakage to an external `import` of just the latter two (a TS `import` only ever brings
// in the specific names asked for, never a whole header's worth of transitively-included types).
// Splitting BUILTIN_FUNCTIONS's own C++ declarations out into this second header is what recovers
// that same "some exports are for cleg's own other files only" distinction here, without pulling
// game/cleg_base.h's full ClegType/ClegValue/AST surface into game/cleg_eval.h itself.

using UserFuncTable = std::unordered_map<std::string, const FunctionDecl*>;

// Mirrors shared/clegEval.ts's BuiltinFunction.
using CheckCallFn = std::function<ClegType(const std::string&, const std::vector<ClegType>&)>;
// `funcs` is only ever needed by a builtin that itself calls a `func`-typed argument back (e.g.
// sub_hcublat's own `cond`, via call_user_function) - every other builtin's own `call` simply
// ignores it.
using CallFn = std::function<ClegValue(const std::vector<ClegValue>&, UserFuncTable&)>;
struct BuiltinFunction { CheckCallFn check_call; CallFn call; };

// Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS - built once via a function-local static
// (thread-safety is a non-issue for this single-threaded CLI tooling).
const std::unordered_map<std::string, BuiltinFunction>& builtin_functions();

// Used by game/cleg_check.cpp's own typecheck_cleg as a harmless placeholder ClegType (check_stmt's
// own return_type param is never actually read while checking a TopStmt - a TopStmt is never a
// ReturnStmt).
extern const ClegType EGR_TYPE;

// CTKind-based counterpart of game/cleg_parser.cpp's own set_elem_kind_words() - used by
// game/cleg_check.cpp's own SetLit case.
bool is_set_elem_kind(CTKind k);
