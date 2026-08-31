#pragma once
#include <memory>
#include <string>

// Mirrors shared/clegParser.ts - the lexer and recursive-descent parser (source text -> AST) - see
// shared/clegBase.ts's own top comment for the cleg language itself and the four-file split this is
// one part of. `ClegProgram` here is forward-declared only (defined in game/cleg_base.h) - a
// shared_ptr<ClegProgram> can be held/passed/destroyed with ClegProgram incomplete, same pimpl-idiom
// guarantee any C++17 standard library provides for shared_ptr's own type-erased deleter - so a
// consumer of just this header (train.cpp/server.cpp) never needs to see the real AST at all, unlike
// the TS side's UI (which also needs unparseCleg - not ported here, see game/cleg_base.h).
struct ClegProgram;

// Mirrors shared/clegParser.ts's parseCleg() - throws std::runtime_error if source doesn't follow
// cleg's grammar.
std::shared_ptr<ClegProgram> parse_cleg(const std::string& source);
