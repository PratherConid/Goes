#pragma once
#include "game/cleg_base.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Internal declarations shared by game/cleg_eval.cpp and game/cleg_check.cpp only - the C++
// counterpart of the shared/clegEval.ts exports that only shared/clegCheck.ts imports
// (BUILTIN_FUNCTIONS, EGR_TYPE). Split out of game/cleg_eval.h because a C++ header can't limit
// names to cleg's own files the way a TS named import effectively does; keeping them here keeps
// game/cleg_base.h's ClegType/ClegValue/AST surface out of game/cleg_eval.h.

using UserFuncTable = std::unordered_map<std::string, const FunctionDecl*>;

// Mirrors shared/clegEval.ts's GlobalInfo.
struct GlobalInfo {
    UserFuncTable funcs;
    ClegRandom random;
};

// Mirrors shared/clegEval.ts's BuiltinFunction.
using CheckCallFn = std::function<ClegType(const std::string&, const std::vector<ClegType>&)>;
using CallFn = std::function<ClegValue(const std::vector<ClegValue>&, GlobalInfo&)>;
struct BuiltinFunction { CheckCallFn check_call; CallFn call; };

// Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS - built once via a function-local static
// (thread-safety is a non-issue for this single-threaded CLI tooling).
const std::unordered_map<std::string, BuiltinFunction>& builtin_functions();

// Mirrors shared/clegEval.ts's exported EGR_TYPE.
extern const ClegType EGR_TYPE;

// Mirrors shared/clegBase.ts's SET_ELEM_KINDS membership test, on CTKind.
bool is_set_elem_kind(CTKind k);
