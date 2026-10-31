#include "game/cleg_parser.h"
#include "game/cleg_base.h"
#include <cctype>
#include <set>
#include <stdexcept>

// Mirrors shared/clegParser.ts.

// ── Lexer ────────────────────────────────────────────────────────────────────

enum class TokKind { Ident, Number, String, Punct, Eof };
struct Token { TokKind kind; std::string text; size_t pos; };

static const std::string PUNCTUATION = "(){}[],;+-*/%!#";

static bool is_ident_start(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
static bool is_ident_cont(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

// Mirrors shared/clegParser.ts's tokenize().
static std::vector<Token> tokenize(const std::string& src) {
    std::vector<Token> tokens;
    size_t n = src.size(), i = 0;
    while (i < n) {
        char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c))) { i++; continue; }
        if (c == '/' && i + 1 < n && src[i + 1] == '/') { while (i < n && src[i] != '\n') i++; continue; }
        if (is_ident_start(c)) {
            size_t j = i + 1;
            while (j < n && is_ident_cont(src[j])) j++;
            tokens.push_back({TokKind::Ident, src.substr(i, j - i), i});
            i = j;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            size_t j = i + 1;
            while (j < n && (std::isdigit(static_cast<unsigned char>(src[j])) || src[j] == '.')) j++;
            tokens.push_back({TokKind::Number, src.substr(i, j - i), i});
            i = j;
            continue;
        }
        if (c == '"') {
            size_t j = i + 1;
            std::string out;
            while (j < n && src[j] != '"') {
                if (src[j] == '\\' && j + 1 < n) {
                    char esc = src[j + 1];
                    out += esc == 'n' ? '\n' : esc == 't' ? '\t' : esc;
                    j += 2;
                } else {
                    out += src[j];
                    j++;
                }
            }
            if (j >= n) throw std::runtime_error("cleg: unterminated string literal starting at position " + std::to_string(i));
            tokens.push_back({TokKind::String, out, i});
            i = j + 1;
            continue;
        }
        if (c == '=' || c == '<' || c == '>' || c == '!') {
            if (i + 1 < n && src[i + 1] == '=') {
                tokens.push_back({TokKind::Punct, std::string(1, c) + "=", i});
                i += 2;
                continue;
            }
            tokens.push_back({TokKind::Punct, std::string(1, c), i});
            i++;
            continue;
        }
        if (c == '&' && i + 1 < n && src[i + 1] == '&') { tokens.push_back({TokKind::Punct, "&&", i}); i += 2; continue; }
        if (c == '|' && i + 1 < n && src[i + 1] == '|') { tokens.push_back({TokKind::Punct, "||", i}); i += 2; continue; }
        if (c == '-' && i + 1 < n && src[i + 1] == '>') { tokens.push_back({TokKind::Punct, "->", i}); i += 2; continue; }
        if (PUNCTUATION.find(c) != std::string::npos) {
            tokens.push_back({TokKind::Punct, std::string(1, c), i});
            i++;
            continue;
        }
        throw std::runtime_error(std::string("cleg: unexpected character '") + c + "' at position " + std::to_string(i));
    }
    tokens.push_back({TokKind::Eof, "", n});
    return tokens;
}

// ── Parser ───────────────────────────────────────────────────────────────────

// Mirrors shared/clegParser.ts's TokenCursor, except peek_at clamps an out-of-range offset to the
// trailing Eof token instead of reading past the end (TS reads `undefined`) - equivalent for the
// parser's 1-2 token lookaheads.
class TokenCursor {
public:
    explicit TokenCursor(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    const Token& peek() const { return tokens_[pos_]; }
    const Token& peek_at(size_t offset) const {
        size_t idx = pos_ + offset;
        return idx < tokens_.size() ? tokens_[idx] : tokens_.back();
    }
    Token next() { return tokens_[pos_++]; }
    bool at_end() const { return peek().kind == TokKind::Eof; }
    bool is_punct(const std::string& p) const { auto& t = peek(); return t.kind == TokKind::Punct && t.text == p; }
    bool is_keyword(const std::string& k) const { auto& t = peek(); return t.kind == TokKind::Ident && t.text == k; }

    size_t save() const { return pos_; }
    void restore(size_t pos) { pos_ = pos; }

    void expect_punct(const std::string& p) {
        Token t = next();
        if (t.kind != TokKind::Punct || t.text != p)
            throw std::runtime_error("cleg: expected '" + p + "', got '" + (t.text.empty() ? "<eof>" : t.text) +
                "' at position " + std::to_string(t.pos));
    }
    std::string expect_ident() {
        Token t = next();
        if (t.kind != TokKind::Ident)
            throw std::runtime_error("cleg: expected an identifier, got '" + (t.text.empty() ? "<eof>" : t.text) +
                "' at position " + std::to_string(t.pos));
        return t.text;
    }

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;
};

static const std::set<std::string>& type_keywords() {
    static const std::set<std::string> kws = {
        "egr", "number", "string", "bool", "edge", "simp", "tri", "quad", "sel", "mod", "formsel", "lrs", "psel",
    };
    return kws;
}
// Mirrors shared/clegBase.ts's SET_ELEM_KINDS, keyed by base-type token for parse_type;
// is_set_elem_kind (game/cleg_eval.cpp) is the CTKind-keyed counterpart for check_expr.
static const std::set<std::string>& set_elem_kind_words() {
    static const std::set<std::string> kws = {"number", "string", "bool", "edge", "simp", "tri", "quad"};
    return kws;
}

// Mirrors shared/clegParser.ts's parseType() keyword-to-kind mapping ('tri' -> simp).
static CTKind base_type_kind(const std::string& base) {
    if (base == "egr") return CTKind::Egr;
    if (base == "number") return CTKind::Number;
    if (base == "string") return CTKind::String;
    if (base == "bool") return CTKind::Bool;
    if (base == "edge") return CTKind::Edge;
    if (base == "simp" || base == "tri") return CTKind::Simp;
    if (base == "quad") return CTKind::Quad;
    if (base == "sel") return CTKind::Sel;
    if (base == "mod") return CTKind::Mod;
    if (base == "formsel") return CTKind::Formsel;
    if (base == "lrs") return CTKind::Lrs;
    if (base == "psel") return CTKind::Psel;
    throw std::runtime_error("cleg: base_type_kind: unknown base type '" + base + "'");
}

// Forward-declared so parse_paren_type can use it ahead of its definition below.
template <typename T, typename F>
static std::vector<T> parse_comma_separated(TokenCursor& c, const std::string& close, F parse_one);

static ClegType parse_paren_type(TokenCursor& c);

static ClegType parse_type(TokenCursor& c) {
    ClegType type;
    if (c.is_punct("(")) {
        type = parse_paren_type(c);
    } else {
        std::string base = c.expect_ident();
        if (!type_keywords().count(base))
            throw std::runtime_error("cleg: expected a type (egr/number/string/bool/edge/simp/tri/quad/sel/mod/formsel/lrs), got '" + base + "'");
        type = ClegType{base_type_kind(base), nullptr};
        if (c.is_punct("{")) {
            if (!set_elem_kind_words().count(base))
                throw std::runtime_error(
                    "cleg: '" + base + "{}' is not a supported set type - sets of egr, sets of sets, and sets of "
                    "arrays are not supported");
            c.next();
            c.expect_punct("}");
            type = ClegType{CTKind::Set, std::make_shared<ClegType>(std::move(type))};
        }
    }
    while (c.is_punct("[")) {
        c.next();
        c.expect_punct("]");
        type = ClegType{CTKind::Array, std::make_shared<ClegType>(std::move(type))};
    }
    return type;
}

// Mirrors shared/clegParser.ts's parseParenType().
static ClegType parse_paren_type(TokenCursor& c) {
    c.expect_punct("(");
    std::vector<ClegType> items = parse_comma_separated<ClegType>(c, ")", [&]() { return parse_type(c); });
    if (c.is_punct("->")) {
        c.next();
        ClegType return_type = parse_type(c);
        return ClegType{CTKind::Func, nullptr, std::move(items), std::make_shared<ClegType>(std::move(return_type))};
    }
    if (items.size() == 1 && items[0].kind == CTKind::Func) return items[0];
    throw std::runtime_error("cleg: expected '->' after a parenthesized parameter list");
}

// Mirrors shared/clegParser.ts's isTypeStart().
static bool is_type_start(TokenCursor& c) {
    auto& t = c.peek();
    if (t.kind == TokKind::Ident && type_keywords().count(t.text) > 0) return true;
    if (!c.is_punct("(")) return false;
    size_t pos = c.save();
    try {
        parse_type(c);
        c.restore(pos);
        return true;
    } catch (...) {
        c.restore(pos);
        return false;
    }
}

// Mirrors shared/clegParser.ts's isFunctionDeclStart().
static bool is_function_decl_start(TokenCursor& c) {
    size_t pos = c.save();
    try {
        parse_type(c);
        bool ok = c.peek().kind == TokKind::Ident && c.peek_at(1).kind == TokKind::Punct && c.peek_at(1).text == "(";
        c.restore(pos);
        return ok;
    } catch (...) {
        c.restore(pos);
        return false;
    }
}

template <typename T, typename F>
static std::vector<T> parse_comma_separated(TokenCursor& c, const std::string& close, F parse_one) {
    std::vector<T> items;
    if (!c.is_punct(close)) {
        items.push_back(parse_one());
        while (c.is_punct(",")) { c.next(); items.push_back(parse_one()); }
    }
    c.expect_punct(close);
    return items;
}

static Stmt parse_block(TokenCursor& c);
static Stmt parse_stmt(TokenCursor& c);
static Expr parse_expr(TokenCursor& c);

static FunctionDecl parse_function_decl(TokenCursor& c) {
    ClegType return_type = parse_type(c);
    std::string name = c.expect_ident();
    c.expect_punct("(");
    auto params = parse_comma_separated<Param>(c, ")", [&]() {
        ClegType type = parse_type(c);
        std::string pname = c.expect_ident();
        return Param{type, pname};
    });
    Stmt body = parse_block(c);
    return FunctionDecl{return_type, name, std::move(params), std::move(body)};
}

static Stmt parse_block(TokenCursor& c) {
    c.expect_punct("{");
    std::vector<Stmt> stmts;
    while (!c.is_punct("}")) stmts.push_back(parse_stmt(c));
    c.expect_punct("}");
    Stmt block; block.kind = StmtKind::Block; block.stmts = std::move(stmts);
    return block;
}

// Mirrors shared/clegParser.ts's isAssignStart().
static bool is_assign_start(TokenCursor& c) {
    if (c.peek().kind != TokKind::Ident) return false;
    if (c.peek_at(1).kind == TokKind::Punct && c.peek_at(1).text == "=") return true;
    if (!(c.peek_at(1).kind == TokKind::Punct && c.peek_at(1).text == "[")) return false;
    size_t pos = c.save();
    try {
        c.next();
        while (c.is_punct("[")) { c.next(); parse_expr(c); c.expect_punct("]"); }
        bool ok = c.is_punct("=");
        c.restore(pos);
        return ok;
    } catch (...) {
        c.restore(pos);
        return false;
    }
}

static Stmt parse_if_stmt(TokenCursor& c);
static Stmt parse_for_stmt(TokenCursor& c);
static Stmt parse_while_stmt(TokenCursor& c);
static Stmt parse_break_stmt(TokenCursor& c);
static Stmt parse_continue_stmt(TokenCursor& c);
static Stmt parse_return_stmt(TokenCursor& c);

static Stmt parse_var_decl_no_semi(TokenCursor& c) {
    ClegType type = parse_type(c);
    std::string name = c.expect_ident();
    c.expect_punct("=");
    Expr init = parse_expr(c);
    Stmt s; s.kind = StmtKind::VarDecl; s.decl_type = type; s.name = name;
    s.expr = std::make_shared<Expr>(std::move(init));
    return s;
}
static Stmt parse_var_decl(TokenCursor& c) {
    Stmt decl = parse_var_decl_no_semi(c);
    c.expect_punct(";");
    return decl;
}
static Stmt parse_assign_stmt_no_semi(TokenCursor& c) {
    std::string name = c.expect_ident();
    std::vector<Expr> indices;
    while (c.is_punct("[")) {
        c.next();
        indices.push_back(parse_expr(c));
        c.expect_punct("]");
    }
    c.expect_punct("=");
    Expr value = parse_expr(c);
    Stmt s; s.kind = StmtKind::AssignStmt; s.name = name; s.indices = std::move(indices);
    s.expr = std::make_shared<Expr>(std::move(value));
    return s;
}
static Stmt parse_assign_stmt(TokenCursor& c) {
    Stmt stmt = parse_assign_stmt_no_semi(c);
    c.expect_punct(";");
    return stmt;
}

static Stmt parse_stmt(TokenCursor& c) {
    if (c.is_punct("{")) return parse_block(c);
    if (c.is_keyword("if")) return parse_if_stmt(c);
    if (c.is_keyword("for")) return parse_for_stmt(c);
    if (c.is_keyword("while")) return parse_while_stmt(c);
    if (c.is_keyword("break")) return parse_break_stmt(c);
    if (c.is_keyword("continue")) return parse_continue_stmt(c);
    if (c.is_keyword("return")) return parse_return_stmt(c);
    if (is_type_start(c)) return parse_var_decl(c);
    if (is_assign_start(c)) return parse_assign_stmt(c);
    Expr expr = parse_expr(c);
    c.expect_punct(";");
    Stmt s; s.kind = StmtKind::ExprStmt; s.expr = std::make_shared<Expr>(std::move(expr));
    return s;
}

static Stmt parse_if_stmt(TokenCursor& c) {
    c.next(); // 'if'
    c.expect_punct("(");
    Expr cond = parse_expr(c);
    c.expect_punct(")");
    Stmt then_b = parse_block(c);
    std::shared_ptr<Stmt> else_s;
    if (c.is_keyword("else")) {
        c.next();
        else_s = std::make_shared<Stmt>(c.is_keyword("if") ? parse_if_stmt(c) : parse_block(c));
    }
    Stmt s; s.kind = StmtKind::IfStmt;
    s.cond = std::make_shared<Expr>(std::move(cond));
    s.then_stmt = std::make_shared<Stmt>(std::move(then_b));
    s.else_stmt = else_s;
    return s;
}

static std::shared_ptr<Stmt> parse_for_init(TokenCursor& c) {
    if (c.is_punct(";")) return nullptr;
    if (is_type_start(c)) return std::make_shared<Stmt>(parse_var_decl_no_semi(c));
    if (is_assign_start(c)) return std::make_shared<Stmt>(parse_assign_stmt_no_semi(c));
    Expr e = parse_expr(c);
    Stmt s; s.kind = StmtKind::ExprStmt; s.expr = std::make_shared<Expr>(std::move(e));
    return std::make_shared<Stmt>(std::move(s));
}
static std::shared_ptr<Stmt> parse_for_update(TokenCursor& c) {
    if (c.is_punct(")")) return nullptr;
    if (is_assign_start(c)) return std::make_shared<Stmt>(parse_assign_stmt_no_semi(c));
    Expr e = parse_expr(c);
    Stmt s; s.kind = StmtKind::ExprStmt; s.expr = std::make_shared<Expr>(std::move(e));
    return std::make_shared<Stmt>(std::move(s));
}
static Stmt parse_for_stmt(TokenCursor& c) {
    c.next(); // 'for'
    c.expect_punct("(");
    auto init = parse_for_init(c);
    c.expect_punct(";");
    std::shared_ptr<Expr> cond = c.is_punct(";") ? nullptr : std::make_shared<Expr>(parse_expr(c));
    c.expect_punct(";");
    auto update = parse_for_update(c);
    c.expect_punct(")");
    Stmt body = parse_block(c);
    Stmt s; s.kind = StmtKind::ForStmt;
    s.for_init = init; s.cond = cond; s.for_update = update;
    s.body = std::make_shared<Stmt>(std::move(body));
    return s;
}

static Stmt parse_while_stmt(TokenCursor& c) {
    c.next(); // 'while'
    c.expect_punct("(");
    Expr cond = parse_expr(c);
    c.expect_punct(")");
    Stmt body = parse_block(c);
    Stmt s; s.kind = StmtKind::WhileStmt;
    s.cond = std::make_shared<Expr>(std::move(cond));
    s.body = std::make_shared<Stmt>(std::move(body));
    return s;
}

static Stmt parse_break_stmt(TokenCursor& c) {
    c.next(); // 'break'
    c.expect_punct(";");
    Stmt s; s.kind = StmtKind::BreakStmt;
    return s;
}

static Stmt parse_continue_stmt(TokenCursor& c) {
    c.next(); // 'continue'
    c.expect_punct(";");
    Stmt s; s.kind = StmtKind::ContinueStmt;
    return s;
}

static Stmt parse_return_stmt(TokenCursor& c) {
    c.next(); // 'return'
    Expr value = parse_expr(c);
    c.expect_punct(";");
    Stmt s; s.kind = StmtKind::ReturnStmt; s.expr = std::make_shared<Expr>(std::move(value));
    return s;
}

static bool is_punct_in(TokenCursor& c, const std::set<std::string>& ops) {
    auto& t = c.peek();
    return t.kind == TokKind::Punct && ops.count(t.text) > 0;
}
static Expr make_binary(std::string op, Expr left, Expr right) {
    Expr e; e.kind = ExprKind::BinaryExpr; e.op = std::move(op);
    e.left = std::make_shared<Expr>(std::move(left));
    e.right = std::make_shared<Expr>(std::move(right));
    return e;
}

static Expr parse_logical_and(TokenCursor& c);
static Expr parse_equality(TokenCursor& c);
static Expr parse_relational(TokenCursor& c);
static Expr parse_additive(TokenCursor& c);
static Expr parse_multiplicative(TokenCursor& c);
static Expr parse_unary(TokenCursor& c);
static Expr parse_postfix(TokenCursor& c);
static Expr parse_atom(TokenCursor& c);

// These mirror shared/clegParser.ts's parseExpr()...parseUnary() chain. Each loop consumes the
// operator into a local before parsing the right operand: C++ leaves function-argument evaluation
// order unspecified (JS is left-to-right), so doing both inside one make_binary(...) call could parse
// the right operand before c.next() advanced past the operator.
static Expr parse_expr(TokenCursor& c) {
    static const std::set<std::string> OR_OPS = {"||"};
    Expr left = parse_logical_and(c);
    while (is_punct_in(c, OR_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_logical_and(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_logical_and(TokenCursor& c) {
    static const std::set<std::string> AND_OPS = {"&&"};
    Expr left = parse_equality(c);
    while (is_punct_in(c, AND_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_equality(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_equality(TokenCursor& c) {
    static const std::set<std::string> EQ_OPS = {"==", "!="};
    Expr left = parse_relational(c);
    while (is_punct_in(c, EQ_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_relational(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_relational(TokenCursor& c) {
    static const std::set<std::string> REL_OPS = {"<", ">", "<=", ">="};
    Expr left = parse_additive(c);
    while (is_punct_in(c, REL_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_additive(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_additive(TokenCursor& c) {
    static const std::set<std::string> ADD_OPS = {"+", "-"};
    Expr left = parse_multiplicative(c);
    while (is_punct_in(c, ADD_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_multiplicative(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_multiplicative(TokenCursor& c) {
    static const std::set<std::string> MUL_OPS = {"*", "/", "%"};
    Expr left = parse_unary(c);
    while (is_punct_in(c, MUL_OPS)) {
        std::string op = c.next().text;
        Expr right = parse_unary(c);
        left = make_binary(std::move(op), std::move(left), std::move(right));
    }
    return left;
}
static Expr parse_unary(TokenCursor& c) {
    if (c.is_punct("-")) {
        c.next();
        Expr e; e.kind = ExprKind::UnaryExpr; e.op = "-"; e.operand = std::make_shared<Expr>(parse_unary(c));
        return e;
    }
    if (c.is_punct("!")) {
        c.next();
        Expr e; e.kind = ExprKind::UnaryExpr; e.op = "!"; e.operand = std::make_shared<Expr>(parse_unary(c));
        return e;
    }
    return parse_postfix(c);
}

// Mirrors shared/clegParser.ts's parsePostfix().
static Expr parse_postfix(TokenCursor& c) {
    Expr e = parse_atom(c);
    while (c.is_punct("[")) {
        c.next();
        Expr index = parse_expr(c);
        c.expect_punct("]");
        Expr idx_expr; idx_expr.kind = ExprKind::IndexExpr;
        idx_expr.left = std::make_shared<Expr>(std::move(e));
        idx_expr.right = std::make_shared<Expr>(std::move(index));
        e = std::move(idx_expr);
    }
    return e;
}

// Mirrors shared/clegParser.ts's parseAtom().
static Expr parse_atom(TokenCursor& c) {
    const Token& tok = c.peek();
    if (tok.kind == TokKind::Number) {
        c.next();
        Expr e; e.kind = ExprKind::NumberLit; e.number_value = std::stod(tok.text);
        return e;
    }
    if (tok.kind == TokKind::String) {
        c.next();
        Expr e; e.kind = ExprKind::StringLit; e.string_value = tok.text;
        return e;
    }
    if (c.is_punct("[")) {
        c.next();
        Expr e; e.kind = ExprKind::ArrayLit;
        e.elements = parse_comma_separated<Expr>(c, "]", [&]() { return parse_expr(c); });
        return e;
    }
    if (c.is_punct("{")) {
        c.next();
        Expr e; e.kind = ExprKind::SetLit;
        e.elements = parse_comma_separated<Expr>(c, "}", [&]() { return parse_expr(c); });
        return e;
    }
    if (c.is_punct("(")) {
        c.next();
        Expr inner = parse_expr(c);
        c.expect_punct(")");
        return inner;
    }
    if (tok.kind == TokKind::Ident) {
        if (tok.text == "true") { c.next(); Expr e; e.kind = ExprKind::BoolLit; e.bool_value = true; return e; }
        if (tok.text == "false") { c.next(); Expr e; e.kind = ExprKind::BoolLit; e.bool_value = false; return e; }
        if (tok.text == "nil") {
            c.next();
            c.expect_punct("(");
            ClegType type = parse_type(c);
            c.expect_punct(")");
            Expr e; e.kind = ExprKind::NilExpr; e.nil_type = type;
            return e;
        }
        std::string name = c.expect_ident();
        if (c.is_punct("(")) {
            c.next();
            Expr e; e.kind = ExprKind::CallExpr; e.string_value = name;
            e.elements = parse_comma_separated<Expr>(c, ")", [&]() -> Expr {
                if (c.is_punct("#")) { c.next(); Expr h; h.kind = ExprKind::HoleExpr; return h; }
                return parse_expr(c);
            });
            return e;
        }
        Expr e; e.kind = ExprKind::Identifier; e.string_value = name;
        return e;
    }
    throw std::runtime_error(
        "cleg: unexpected token '" + (tok.text.empty() ? "<eof>" : tok.text) + "' at position " + std::to_string(tok.pos));
}

// Mirrors shared/clegParser.ts's parseCleg(); parse_cleg() below wraps the result in a shared_ptr.
static ClegProgram parse_cleg_impl(const std::string& source) {
    TokenCursor c(tokenize(source));
    ClegProgram program;
    while (!c.at_end()) {
        if (is_function_decl_start(c)) { program.functions.push_back(parse_function_decl(c)); continue; }
        if (is_type_start(c)) { program.stmts.push_back(parse_var_decl(c)); continue; }
        if (is_assign_start(c)) { program.stmts.push_back(parse_assign_stmt(c)); continue; }
        Expr expr = parse_expr(c);
        c.expect_punct(";");
        Stmt s; s.kind = StmtKind::ExprStmt; s.expr = std::make_shared<Expr>(std::move(expr));
        program.stmts.push_back(std::move(s));
    }
    return program;
}

// ── Public API ───────────────────────────────────────────────────────────────

std::shared_ptr<ClegProgram> parse_cleg(const std::string& source) {
    return std::make_shared<ClegProgram>(parse_cleg_impl(source));
}
