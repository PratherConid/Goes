#pragma once
#include <memory>
#include <string>

// Mirrors shared/clegParser.ts, minus unparseCleg (see game/cleg_base.h). `ClegProgram` is only
// forward-declared (shared_ptr's type-erased deleter allows an incomplete type), so consumers of
// this header (train.cpp/server.cpp) never see the AST.
struct ClegProgram;

// Mirrors shared/clegParser.ts's parseCleg() - throws std::runtime_error.
std::shared_ptr<ClegProgram> parse_cleg(const std::string& source);
