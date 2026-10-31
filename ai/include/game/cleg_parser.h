#pragma once
#include <memory>
#include <string>

// Mirrors shared/clegParser.ts. `ClegProgram`/`Stmt` are only forward-declared (shared_ptr's
// type-erased deleter allows an incomplete type), so consumers of this header (train.cpp/server.cpp)
// never see the AST.
struct ClegProgram;
struct Stmt;

// Mirrors shared/clegParser.ts's parseCleg() - throws std::runtime_error.
std::shared_ptr<ClegProgram> parse_cleg(const std::string& source);

// Mirrors shared/clegParser.ts's unparseCleg().
std::string unparse_cleg(const ClegProgram& program);

// Mirrors shared/clegParser.ts's unparseAssignTarget() - `stmt` is an AssignStmt.
std::string unparse_assign_target(const Stmt& stmt, int indent);
