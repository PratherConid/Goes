#include "game/cleg_eval.h"
#include "game/cleg_eval_internal.h"
#include "game/cleg_check.h"
#include "game/topology.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// Mirrors shared/clegEval.ts - the builtin-function table and the tree-walking evaluator, plus the
// build_board_from_cleg entry point - see game/cleg_eval.h's own top comment for the full context.
// This file only points out where the C++ port differs from the TS source; the TS comments are the
// canonical reference for WHAT each piece does, not repeated here.
//   - TypeEnv/ValueEnv's parent chain (ValueEnv is this file's own; TypeEnv is game/cleg_check.cpp's)
//     uses raw pointers into the enclosing call's own stack frames (safe here: every env is
//     stack-allocated and never outlives the recursive check/eval call that owns it) rather than TS's
//     GC'd object graph.

// ── Predefined (board-construction) functions ─────────────────────────────────

// Forward-declared here (real definitions live in the Evaluation section below, their own more
// natural spot) purely so sub_hcublat's own call lambda - defined further down in THIS section, but
// needing to call back into a func-typed argument - can use them.
static ClegValue call_user_function(const FunctionDecl& fn, const std::vector<ClegValue>& args, UserFuncTable& funcs);
static std::vector<ClegValue> fill_holes(
    const std::vector<std::optional<ClegValue>>& bound_args, const std::vector<ClegValue>& supplied_args);

// Mirrors shared/clegEval.ts's fixedSignature().
static CheckCallFn fixed_signature(std::vector<ClegType> params, ClegType return_type) {
    return [params, return_type](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
        if (arg_types.size() != params.size())
            throw std::runtime_error(
                "cleg: '" + callee + "' expects " + std::to_string(params.size()) + " argument(s), got " +
                std::to_string(arg_types.size()));
        for (size_t i = 0; i < arg_types.size(); i++)
            if (!type_equals(arg_types[i], params[i]))
                throw std::runtime_error(
                    "cleg: '" + callee + "' argument " + std::to_string(i + 1) + ": expected " +
                    type_to_string(params[i]) + ", got " + type_to_string(arg_types[i]));
        return return_type;
    };
}

static const ClegType NUMBER_TYPE{CTKind::Number, nullptr};
static const ClegType STRING_TYPE{CTKind::String, nullptr};
const ClegType EGR_TYPE{CTKind::Egr, nullptr};
static const ClegType EDGE_TYPE{CTKind::Edge, nullptr};
static const ClegType TRI_TYPE{CTKind::Tri, nullptr};
static const ClegType QUAD_TYPE{CTKind::Quad, nullptr};
static const ClegType MOD_TYPE{CTKind::Mod, nullptr};
static const ClegType MSEL_TYPE{CTKind::Msel, nullptr};

// CTKind-based counterpart of set_elem_kind_words() (game/cleg_parser.cpp) - used by the type
// checker's SetLit case (game/cleg_check.cpp).
bool is_set_elem_kind(CTKind k) {
    return k == CTKind::Number || k == CTKind::String || k == CTKind::Bool ||
           k == CTKind::Edge || k == CTKind::Tri || k == CTKind::Quad;
}

static std::string selector_type_word(SelectorType t) {
    switch (t) {
        case SelectorType::Node: return "node";
        case SelectorType::Edge: return "edge";
        case SelectorType::Tri: return "tri";
        case SelectorType::Quad: return "quad";
    }
    throw std::runtime_error("cleg: selector_type_word: unexpected SelectorType");
}
// Mirrors shared/clegEval.ts's SELECTOR_SET_ELEM_KIND.
static CTKind selector_set_elem_kind(SelectorType t) {
    switch (t) {
        case SelectorType::Node: return CTKind::Number;
        case SelectorType::Edge: return CTKind::Edge;
        case SelectorType::Tri: return CTKind::Tri;
        case SelectorType::Quad: return CTKind::Quad;
    }
    throw std::runtime_error("cleg: selector_set_elem_kind: unexpected SelectorType");
}

// Mirrors shared/clegEval.ts's setValueToSelectedVals() - builds a `raw` Selector wrapping `values`
// (a `set`'s own deduplicated ClegValue vector) directly, populating whichever of Selector's own
// raw_nodes/raw_edges/raw_tris/raw_quads (game/selector.h) matches want_kind.
static Selector raw_selector_from_set(SelectorType want_kind, const std::vector<ClegValue>& values) {
    Selector sel; sel.op = SelectorOp::Raw; sel.type = want_kind;
    switch (want_kind) {
        case SelectorType::Node:
            for (auto& v : values) sel.raw_nodes.insert(static_cast<int>(std::llround(v.number)));
            break;
        case SelectorType::Edge:
            for (auto& v : values) sel.raw_edges.push_back(v.edge_v);
            break;
        case SelectorType::Tri:
            for (auto& v : values) sel.raw_tris.push_back(v.tri_v);
            break;
        case SelectorType::Quad:
            for (auto& v : values) sel.raw_quads.push_back(v.quad_v);
            break;
    }
    return sel;
}

using SelectorParseFn = Selector (*)(const std::string&);

// Mirrors shared/clegEval.ts's resolveSelectorArg().
static Selector resolve_selector_arg(
    const std::string& callee, const ClegValue& arg, SelectorType want_kind, SelectorParseFn parse_fn)
{
    if (arg.kind == CTKind::String) return parse_fn(arg.str);
    if (arg.kind == CTKind::Set) {
        if (arg.elem.kind != selector_set_elem_kind(want_kind))
            throw std::runtime_error(
                "cleg: '" + callee + "' expects a " + selector_type_word(want_kind) + " selector, got a set of " +
                type_to_string(arg.elem));
        return raw_selector_from_set(want_kind, arg.arr_v);
    }
    // arg.kind == CTKind::Sel
    if (arg.sel_type != want_kind)
        throw std::runtime_error(
            "cleg: '" + callee + "' expects a " + selector_type_word(want_kind) + " selector, got a '" +
            selector_type_word(arg.sel_type) + "' selector");
    return arg.sel_v;
}

// One entry per shared/boardConfig.ts's own PrescribedBoardMap row - old_kind is build_board_config's
// own dispatch string (e.g. "rect"); the cleg name registered below is always old_kind + "B" (e.g.
// "rectB"), matching PrescribedBoardMap's own cleg-name field exactly, so this table alone (rather
// than a separate PrescribedBoard enum/PrescribedBoardMap/PrescribedBoardFns trio like the TS side
// has) is enough to drive registration - no UI here needs PrescribedBoardMap's own display-only
// argStr/desc fields.
struct PrescribedEntry { std::string old_kind; std::vector<BoardArgKind> arg_kinds; };
static const std::vector<PrescribedEntry>& prescribed_boards() {
    static const std::vector<PrescribedEntry> table = {
        {"line", {BoardArgKind::Number}},
        {"rect", {BoardArgKind::Number, BoardArgKind::Number}},
        {"rectd", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"cublat", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"hcub", {BoardArgKind::Number, BoardArgKind::CommaSeparatedNumbers}},
        {"tri", {BoardArgKind::Number}},
        {"regpoly", {BoardArgKind::Number}},
        {"tetra", {}},
        {"dodeca", {}},
        {"icosa", {}},
        {"trihex", {BoardArgKind::Number}},
        {"hex", {BoardArgKind::Number}},
        {"hexdel", {BoardArgKind::Number}},
        {"snubsq", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"snubsqtri", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"twsq", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"gtsq", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"star", {BoardArgKind::Number}},
        {"octa", {}},
        {"sier", {BoardArgKind::Number, BoardArgKind::Number}},
        {"ortho", {BoardArgKind::Number}},
        {"dodflake", {BoardArgKind::Number}},
        {"icoflake", {BoardArgKind::Number}},
        {"octaflake", {BoardArgKind::Number}},
        {"polyflake", {BoardArgKind::Number, BoardArgKind::Number}},
        {"cpolyflake", {BoardArgKind::Number, BoardArgKind::Number}},
        {"cpentflake", {BoardArgKind::Number}},
        {"menger", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::ZeroOneList}},
        {"ap", {BoardArgKind::Number}},
    };
    return table;
}

static ClegType arg_kind_to_cleg_type(BoardArgKind k) {
    switch (k) {
        case BoardArgKind::Number: return NUMBER_TYPE;
        case BoardArgKind::CommaSeparatedNumbers: return ClegType{CTKind::Array, std::make_shared<ClegType>(NUMBER_TYPE)};
        case BoardArgKind::ZeroOneList: return STRING_TYPE;
    }
    throw std::runtime_error("cleg: arg_kind_to_cleg_type: unexpected BoardArgKind");
}

// Mirrors shared/clegEval.ts's valueToBoardArgEntry().
static BoardArgEntry value_to_board_arg_entry(BoardArgKind kind, const ClegValue& val) {
    switch (kind) {
        case BoardArgKind::Number:
            if (val.kind != CTKind::Number)
                throw std::runtime_error("cleg: expected a number argument, got " + type_to_string(cleg_value_type(val)));
            return num_arg(static_cast<int>(val.number));
        case BoardArgKind::CommaSeparatedNumbers: {
            if (val.kind != CTKind::Array || val.elem.kind != CTKind::Number)
                throw std::runtime_error("cleg: expected a number[] argument, got " + type_to_string(cleg_value_type(val)));
            std::vector<int> nums;
            for (auto& v : val.arr_v) nums.push_back(static_cast<int>(v.number));
            return csv_arg(std::move(nums));
        }
        case BoardArgKind::ZeroOneList: {
            if (val.kind != CTKind::String)
                throw std::runtime_error("cleg: expected a string argument, got " + type_to_string(cleg_value_type(val)));
            // Reuses shared/types.ts's own ZeroOneList validation, mirrored directly here (a plain
            // 0/1-character string, each char becoming one int element).
            std::vector<int> values;
            for (char ch : val.str) {
                if (ch != '0' && ch != '1')
                    throw std::runtime_error("cleg: expected a string of 0s and 1s, got '" + val.str + "'");
                values.push_back(ch - '0');
            }
            return zol_arg(std::move(values));
        }
    }
    throw std::runtime_error("cleg: value_to_board_arg_entry: unexpected BoardArgKind");
}

// ── multiProd: N-ary Cartesian board product, restricted by a MultiSelector ────

// Inverse of selector_set_elem_kind above - used only by resolve_any_kind_selector_arg below, which
// (unlike resolve_selector_arg) has no fixed want_kind to check a set's element type against.
static bool is_selector_set_elem_kind(CTKind k) {
    return k == CTKind::Number || k == CTKind::Edge || k == CTKind::Tri || k == CTKind::Quad;
}
static SelectorType selector_type_from_set_elem_kind(CTKind k) {
    switch (k) {
        case CTKind::Number: return SelectorType::Node;
        case CTKind::Edge: return SelectorType::Edge;
        case CTKind::Tri: return SelectorType::Tri;
        case CTKind::Quad: return SelectorType::Quad;
        default: throw std::runtime_error("cleg: selector_type_from_set_elem_kind: not a selector-set element kind");
    }
}

// Mirrors shared/clegEval.ts's resolveAnyKindSelectorArg() - a `sel` value (used directly), a
// `string` (parsed via game/selector.h's own context-free parse_selector, whichever kind the text
// itself turns out to be), or a `set` (of number/edge/tri/quad, wrapped into a `raw` Selector, its
// own kind read off the set's own element type).
static Selector resolve_any_kind_selector_arg(const std::string& callee, const ClegValue& arg) {
    if (arg.kind == CTKind::Sel) return arg.sel_v;
    if (arg.kind == CTKind::String) return parse_selector(arg.str);
    if (arg.kind == CTKind::Set) {
        if (!is_selector_set_elem_kind(arg.elem.kind))
            throw std::runtime_error(
                "cleg: '" + callee + "': a selector set must be a set of number/edge/tri/quad, got a set of " +
                type_to_string(arg.elem));
        SelectorType want_kind = selector_type_from_set_elem_kind(arg.elem.kind);
        return raw_selector_from_set(want_kind, arg.arr_v);
    }
    throw std::runtime_error("cleg: '" + callee + "': expected sel, string, or set, got " + type_to_string(cleg_value_type(arg)));
}

// Mirrors shared/clegEval.ts's FullProductIndex/makeFullProductIndex/fullIndexOf/tupleOfFullIndex - a
// long long index space (rather than TS's own double, which is exact up to 2^53) since the full,
// unrestricted product of several sizable boards can exceed 32-bit int range.
struct FullProductIndex { std::vector<int> Ns; std::vector<long long> stride; long long total; };

static FullProductIndex make_full_product_index(const std::vector<BoardConfig>& boards) {
    FullProductIndex fpi;
    fpi.Ns.resize(boards.size());
    for (size_t i = 0; i < boards.size(); i++) fpi.Ns[i] = boards[i].N;
    fpi.stride.resize(fpi.Ns.size());
    fpi.stride.back() = 1;
    for (int k = static_cast<int>(fpi.Ns.size()) - 2; k >= 0; k--) fpi.stride[k] = fpi.stride[k + 1] * fpi.Ns[k + 1];
    long long total = 1;
    for (int n : fpi.Ns) total *= n;
    fpi.total = total;
    return fpi;
}
static long long full_index_of(const FullProductIndex& fpi, const std::vector<int>& tuple) {
    long long sum = 0;
    for (size_t k = 0; k < tuple.size(); k++) sum += static_cast<long long>(tuple[k]) * fpi.stride[k];
    return sum;
}
static std::vector<int> tuple_of_full_index(const FullProductIndex& fpi, long long idx) {
    std::vector<int> tuple(fpi.Ns.size());
    for (size_t k = 0; k < fpi.Ns.size(); k++) tuple[k] = static_cast<int>((idx / fpi.stride[k]) % fpi.Ns[k]);
    return tuple;
}

// Mirrors shared/clegEval.ts's restrictBoardBySelector().
struct RestrictResult { BoardConfig bc; std::vector<int> survivors; };
static RestrictResult restrict_board_by_selector(const BoardConfig& board, const Selector& sel) {
    if (sel.type == SelectorType::Node) {
        auto kept = select_node(board.adj, board.embed, sel);
        std::vector<int> survivors(kept.begin(), kept.end()); // std::set already ascending
        return {node_induced_subgraph(board, kept), std::move(survivors)};
    }
    if (sel.type == SelectorType::Edge) {
        auto edges = select_edge(board.adj, board.embed, sel);
        std::set<int> kept;
        for (auto& e : edges) { kept.insert(e.n1); kept.insert(e.n2); }
        std::vector<int> survivors(kept.begin(), kept.end());
        return {edge_induced_subgraph(board, edges), std::move(survivors)};
    }
    throw std::runtime_error(
        "cleg: multiProd: msBase's own selector must be a node or edge selector, got a '" +
        selector_type_word(sel.type) + "' selector");
}

static std::set<long long> universal_original_indices(const FullProductIndex& fpi) {
    std::set<long long> all;
    for (long long i = 0; i < fpi.total; i++) all.insert(i);
    return all;
}

// Mirrors shared/clegEval.ts's buildFromOriginalIndices() - no defaultProductProjMat/Embedding here
// (unlike TS, C++'s BoardConfig has no projection-matrix field at all - it never renders, see
// board_config.h's own top comment), so the fresh BoardConfig is built directly.
struct BuiltResult { BoardConfig bc; std::vector<long long> orig_index; };
static BuiltResult build_from_original_indices(
    const std::vector<BoardConfig>& boards, const FullProductIndex& fpi, const std::set<long long>& kept_original)
{
    std::vector<long long> orig_index(kept_original.begin(), kept_original.end()); // std::set already ascending
    std::vector<std::vector<int>> tuples;
    tuples.reserve(orig_index.size());
    for (long long idx : orig_index) tuples.push_back(tuple_of_full_index(fpi, idx));
    unsigned emb_dim = 0;
    for (auto& b : boards) emb_dim += b.emb_dim;
    size_t K = orig_index.size();
    std::vector<std::vector<unsigned>> pos(K);
    for (size_t i = 0; i < K; i++) {
        pos[i].reserve(emb_dim);
        for (size_t k = 0; k < boards.size(); k++) {
            auto& bp = boards[k].embed[tuples[i][k]];
            pos[i].insert(pos[i].end(), bp.begin(), bp.end());
        }
    }
    auto adj = zero_adj(static_cast<int>(K));
    for (size_t a = 0; a < K; a++) {
        for (size_t b = a + 1; b < K; b++) {
            int diff_coord = -1;
            bool too_many_diffs = false;
            for (size_t k = 0; k < boards.size(); k++) {
                if (tuples[a][k] != tuples[b][k]) {
                    if (diff_coord != -1) { too_many_diffs = true; break; }
                    diff_coord = static_cast<int>(k);
                }
            }
            if (!too_many_diffs && diff_coord >= 0 &&
                boards[diff_coord].adj[tuples[a][diff_coord]][tuples[b][diff_coord]]) {
                adj[a][b] = 1;
                adj[b][a] = 1;
            }
        }
    }
    BoardConfig bc{static_cast<int>(K), std::move(adj), emb_dim, std::move(pos)};
    return {std::move(bc), std::move(orig_index)};
}

// Mirrors shared/clegEval.ts's evalMultiSelector() - see its own doc comment for the full algorithm.
static BuiltResult eval_multi_selector(
    const std::vector<BoardConfig>& boards, const FullProductIndex& fpi, const MultiSelector& msel)
{
    switch (msel.op) {
        case MSelOp::All:
            return build_from_original_indices(boards, fpi, universal_original_indices(fpi));
        case MSelOp::Base: {
            auto restricted = restrict_board_by_selector(boards[msel.number], *msel.sel);
            std::vector<BoardConfig> factor_boards = boards;
            factor_boards[msel.number] = restricted.bc;
            BoardConfig bc = factor_boards[0];
            for (size_t i = 1; i < factor_boards.size(); i++) bc = product(bc, factor_boards[i]);
            std::vector<int> local_ns(factor_boards.size());
            for (size_t i = 0; i < factor_boards.size(); i++) local_ns[i] = factor_boards[i].N;
            std::vector<long long> local_stride(local_ns.size());
            local_stride.back() = 1;
            for (int k = static_cast<int>(local_ns.size()) - 2; k >= 0; k--)
                local_stride[k] = local_stride[k + 1] * local_ns[k + 1];
            std::vector<long long> orig_index(bc.N);
            for (int local = 0; local < bc.N; local++) {
                std::vector<int> tuple(local_ns.size());
                for (size_t k = 0; k < local_ns.size(); k++)
                    tuple[k] = static_cast<int>((local / local_stride[k]) % local_ns[k]);
                tuple[msel.number] = restricted.survivors[tuple[msel.number]];
                orig_index[local] = full_index_of(fpi, tuple);
            }
            return {std::move(bc), std::move(orig_index)};
        }
        case MSelOp::Union: {
            std::set<long long> kept;
            for (auto& item : msel.items) {
                auto r = eval_multi_selector(boards, fpi, item);
                kept.insert(r.orig_index.begin(), r.orig_index.end());
            }
            return build_from_original_indices(boards, fpi, kept);
        }
        case MSelOp::Inter: {
            if (msel.items.empty()) return build_from_original_indices(boards, fpi, universal_original_indices(fpi));
            auto first = eval_multi_selector(boards, fpi, msel.items[0]);
            std::set<long long> kept(first.orig_index.begin(), first.orig_index.end());
            for (size_t i = 1; i < msel.items.size(); i++) {
                auto next = eval_multi_selector(boards, fpi, msel.items[i]);
                std::set<long long> next_set(next.orig_index.begin(), next.orig_index.end());
                std::set<long long> out;
                for (long long idx : kept) if (next_set.count(idx)) out.insert(idx);
                kept = std::move(out);
            }
            return build_from_original_indices(boards, fpi, kept);
        }
        case MSelOp::Diff: {
            auto ra = eval_multi_selector(boards, fpi, *msel.a);
            auto rb = eval_multi_selector(boards, fpi, *msel.b);
            std::set<long long> b_set(rb.orig_index.begin(), rb.orig_index.end());
            std::set<long long> out;
            for (long long idx : ra.orig_index) if (!b_set.count(idx)) out.insert(idx);
            return build_from_original_indices(boards, fpi, out);
        }
    }
    throw std::runtime_error("cleg: eval_multi_selector: unexpected MSelOp");
}

// Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS - built once via a function-local static (thread-safety
// is a non-issue for this single-threaded CLI tooling), one lambda per row instead of TS's top-level
// `BUILTIN_FUNCTIONS['x'] = {...}` assignments.
const std::unordered_map<std::string, BuiltinFunction>& builtin_functions() {
    static const std::unordered_map<std::string, BuiltinFunction> table = [] {
        std::unordered_map<std::string, BuiltinFunction> m;

        // One builtin per shared/boardConfig.ts's own PrescribedBoardMap/PrescribedBoardFns entry -
        // see prescribed_boards()'s own doc comment above.
        for (auto& entry : prescribed_boards()) {
            std::string cleg_name = entry.old_kind + "B";
            std::vector<ClegType> params;
            for (auto k : entry.arg_kinds) params.push_back(arg_kind_to_cleg_type(k));
            std::string old_kind = entry.old_kind;
            std::vector<BoardArgKind> arg_kinds = entry.arg_kinds;
            m[cleg_name] = BuiltinFunction{
                fixed_signature(params, EGR_TYPE),
                [old_kind, arg_kinds](const std::vector<ClegValue>& args, UserFuncTable&) {
                    std::vector<BoardArgEntry> board_args;
                    for (size_t i = 0; i < arg_kinds.size(); i++)
                        board_args.push_back(value_to_board_arg_entry(arg_kinds[i], args[i]));
                    return make_egr(build_board_config(old_kind, board_args));
                },
            };
        }

        m["len"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 1)
                    throw std::runtime_error("cleg: '" + callee + "' expects 1 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Array && arg_types[0].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected an array or set, got " + type_to_string(arg_types[0]));
                return NUMBER_TYPE;
            },
            [](const std::vector<ClegValue>& args, UserFuncTable&) { return make_number(static_cast<double>(args[0].arr_v.size())); },
        };

        // `has(x, e)`: whether `x` (a `T[]` or `T{}`) contains `e` (a `T`), as a `bool` - like `len`,
        // its result depends on the actual argument types (here, argument 2's own required type,
        // taken from argument 1's element type), hence the hand-written check_call/call pair. `T` is
        // restricted to is_set_elem_kind (number/string/bool/edge/tri/quad) for BOTH `T[]` and `T{}`
        // - even though an array's own element type is normally unrestricted, nothing outside
        // is_set_elem_kind has a defined equality in this language, so `has` can't be given a
        // well-defined meaning for one either. Compares by cleg_set_key, the same equality every set
        // operation already uses.
        m["has"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 2)
                    throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Array && arg_types[0].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected an array or set, got " + type_to_string(arg_types[0]));
                const ClegType& elem = *arg_types[0].elem;
                if (!is_set_elem_kind(elem.kind))
                    throw std::runtime_error(
                        "cleg: '" + callee + "' argument 1: element type " + type_to_string(elem) + " has no defined "
                        "equality - only number/string/bool/edge/tri/quad elements are supported");
                if (!type_equals(arg_types[1], elem))
                    throw std::runtime_error(
                        "cleg: '" + callee + "' argument 2: expected " + type_to_string(elem) + " (the element type of "
                        "argument 1), got " + type_to_string(arg_types[1]));
                return ClegType{CTKind::Bool, nullptr};
            },
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                std::string key = cleg_set_key(args[1]);
                for (auto& v : args[0].arr_v) if (cleg_set_key(v) == key) return make_bool(true);
                return make_bool(false);
            },
        };

        CheckCallFn rand_rm_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 2)
                throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected a set, got " + type_to_string(arg_types[0]));
            if (arg_types[1].kind != CTKind::Number)
                throw std::runtime_error("cleg: '" + callee + "' argument 2: expected number, got " + type_to_string(arg_types[1]));
            return arg_types[0];
        };
        m["randRmN"] = BuiltinFunction{
            rand_rm_check_call,
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                double count = args[1].number;
                if (count != std::floor(count) || count < 0)
                    throw std::runtime_error("cleg: 'randRmN' count must be a nonnegative integer, got " + format_number_display(count));
                auto kept = randomly_remove(args[0].arr_v, static_cast<int>(count));
                ClegValue v; v.kind = CTKind::Set; v.elem = args[0].elem; v.arr_v = std::move(kept);
                return v;
            },
        };
        m["randRmP"] = BuiltinFunction{
            rand_rm_check_call,
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                double frac = args[1].number;
                if (!std::isfinite(frac) || frac < 0)
                    throw std::runtime_error("cleg: 'randRmP' portion must be a nonnegative number, got " + format_number_display(frac));
                int remove_count = static_cast<int>(std::floor(frac * static_cast<double>(args[0].arr_v.size())));
                auto kept = randomly_remove(args[0].arr_v, remove_count);
                ClegValue v; v.kind = CTKind::Set; v.elem = args[0].elem; v.arr_v = std::move(kept);
                return v;
            },
        };

        // Mirrors shared/clegEval.ts's `abs`/`sqrt` - fixed-signature `number -> number`. `sqrt`
        // throws at evaluation time for a negative argument (not statically knowable from `number`'s
        // type alone) rather than returning NaN, matching every other cleg evaluation-time validity
        // check.
        m["abs"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, NUMBER_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) { return make_number(std::abs(args[0].number)); },
        };
        m["sqrt"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, NUMBER_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                double v = args[0].number;
                if (v < 0) throw std::runtime_error("cleg: 'sqrt' argument must be nonnegative, got " + format_number_display(v));
                return make_number(std::sqrt(v));
            },
        };

        m["mkEdge"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE}, EDGE_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                return make_edge_v(make_board_edge(static_cast<int>(args[0].number), static_cast<int>(args[1].number)));
            },
        };
        m["mkTri"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE}, TRI_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                return make_tri_v(make_board_triangle(
                    static_cast<int>(args[0].number), static_cast<int>(args[1].number), static_cast<int>(args[2].number)));
            },
        };
        m["mkQuad"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE}, QUAD_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                return make_quad_v(make_board_quad(
                    static_cast<int>(args[0].number), static_cast<int>(args[1].number),
                    static_cast<int>(args[2].number), static_cast<int>(args[3].number)));
            },
        };

        m["prod"] = BuiltinFunction{
            fixed_signature({EGR_TYPE, EGR_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) { return make_egr(product(*args[0].egr_v, *args[1].egr_v)); },
        };

        m["mkSel"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 1)
                    throw std::runtime_error("cleg: '" + callee + "' expects 1 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::String && arg_types[0].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected string or set, got " + type_to_string(arg_types[0]));
                return ClegType{CTKind::Sel, nullptr};
            },
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                Selector sel = resolve_any_kind_selector_arg("mkSel", args[0]);
                ClegValue v; v.kind = CTKind::Sel; v.sel_type = sel.type; v.sel_v = std::move(sel);
                return v;
            },
        };

        m["rectify"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&, UserFuncTable&) { return make_mod(BoardModifier{ModifierKind::Rectify}); },
        };
        m["globalCentralize"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&, UserFuncTable&) { return make_mod(BoardModifier{ModifierKind::GlobalCentralize}); },
        };
        m["quadOctarize"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&, UserFuncTable&) { return make_mod(BoardModifier{ModifierKind::QuadOctarize}); },
        };
        m["edgeSplit"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::EdgeSplit}; bm.split_n = static_cast<int>(args[0].number);
                return make_mod(bm);
            },
        };
        m["mergeClose"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::MergeClose}; bm.dist = args[0].number;
                return make_mod(bm);
            },
        };
        m["scale"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::Scale}; bm.dist = args[0].number;
                return make_mod(bm);
            },
        };

        CheckCallFn induced_subgraph_mod_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 1)
                throw std::runtime_error("cleg: '" + callee + "' expects 1 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::Sel && arg_types[0].kind != CTKind::String && arg_types[0].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected sel, string, or set, got " + type_to_string(arg_types[0]));
            return MOD_TYPE;
        };
        m["nis"] = BuiltinFunction{
            induced_subgraph_mod_check_call,
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::NodeInducedSubgraph};
                bm.sel = resolve_selector_arg("nis", args[0], SelectorType::Node, parse_node_selector);
                return make_mod(bm);
            },
        };
        m["eis"] = BuiltinFunction{
            induced_subgraph_mod_check_call,
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::EdgeInducedSubgraph};
                bm.sel = resolve_selector_arg("eis", args[0], SelectorType::Edge, parse_edge_selector);
                return make_mod(bm);
            },
        };

        auto select_set_check_call = [](ClegType elem_type) -> CheckCallFn {
            return [elem_type](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 2)
                    throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Sel && arg_types[0].kind != CTKind::String && arg_types[0].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected sel, string, or set, got " + type_to_string(arg_types[0]));
                if (arg_types[1].kind != CTKind::Egr)
                    throw std::runtime_error("cleg: '" + callee + "' argument 2: expected egr, got " + type_to_string(arg_types[1]));
                return ClegType{CTKind::Set, std::make_shared<ClegType>(elem_type)};
            };
        };
        m["selectNode"] = BuiltinFunction{
            select_set_check_call(NUMBER_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                Selector sel = resolve_selector_arg("selectNode", args[0], SelectorType::Node, parse_node_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto nodes = select_node(bc.adj, bc.embed, sel);
                std::vector<ClegValue> values;
                for (int n : nodes) values.push_back(make_number(n));
                return make_cleg_set(NUMBER_TYPE, std::move(values));
            },
        };
        m["selectEdge"] = BuiltinFunction{
            select_set_check_call(EDGE_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                Selector sel = resolve_selector_arg("selectEdge", args[0], SelectorType::Edge, parse_edge_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto edges = select_edge(bc.adj, bc.embed, sel);
                std::vector<ClegValue> values;
                for (auto& e : edges) values.push_back(make_edge_v(e));
                return make_cleg_set(EDGE_TYPE, std::move(values));
            },
        };
        m["selectTriangle"] = BuiltinFunction{
            select_set_check_call(TRI_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                Selector sel = resolve_selector_arg("selectTriangle", args[0], SelectorType::Tri, parse_triangle_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto tris = select_triangle(bc.adj, bc.embed, sel);
                std::vector<ClegValue> values;
                for (auto& t : tris) values.push_back(make_tri_v(t));
                return make_cleg_set(TRI_TYPE, std::move(values));
            },
        };
        m["selectQuad"] = BuiltinFunction{
            select_set_check_call(QUAD_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                Selector sel = resolve_selector_arg("selectQuad", args[0], SelectorType::Quad, parse_quad_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto quads = select_quad(bc.adj, bc.embed, sel);
                std::vector<ClegValue> values;
                for (auto& q : quads) values.push_back(make_quad_v(q));
                return make_cleg_set(QUAD_TYPE, std::move(values));
            },
        };

        CheckCallFn form_mod_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 1 && arg_types.size() != 2)
                throw std::runtime_error("cleg: '" + callee + "' expects 1 or 2 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::Number)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
            if (arg_types.size() == 2 && arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
            return MOD_TYPE;
        };
        m["triangleForm"] = BuiltinFunction{
            form_mod_check_call,
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::TriangleForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("triangleForm", args[1], SelectorType::Tri, parse_triangle_selector);
                return make_mod(bm);
            },
        };
        m["quadForm"] = BuiltinFunction{
            form_mod_check_call,
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::QuadForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("quadForm", args[1], SelectorType::Quad, parse_quad_selector);
                return make_mod(bm);
            },
        };

        m["form"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() < 2)
                    throw std::runtime_error(
                        "cleg: '" + callee + "' expects at least 2 argument(s) (w, and >= 1 sel), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Number)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
                for (size_t i = 1; i < arg_types.size(); i++)
                    if (arg_types[i].kind != CTKind::Sel && arg_types[i].kind != CTKind::String && arg_types[i].kind != CTKind::Set)
                        throw std::runtime_error("cleg: '" + callee + "' argument " + std::to_string(i + 1) + ": expected sel, string, or set, got " + type_to_string(arg_types[i]));
                return MOD_TYPE;
            },
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::Form};
                bm.split_n = static_cast<int>(args[0].number);
                for (size_t i = 1; i < args.size(); i++) bm.form_sels.push_back(resolve_any_kind_selector_arg("form", args[i]));
                return make_mod(bm);
            },
        };

        CheckCallFn centralize_mod_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 0 && arg_types.size() != 1)
                throw std::runtime_error("cleg: '" + callee + "' expects 0 or 1 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types.size() == 1 && arg_types[0].kind != CTKind::Sel && arg_types[0].kind != CTKind::String && arg_types[0].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected sel, string, or set, got " + type_to_string(arg_types[0]));
            return MOD_TYPE;
        };
        m["triCentralize"] = BuiltinFunction{
            centralize_mod_check_call,
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::TriCentralize};
                if (args.size() == 1) bm.form_sel = resolve_selector_arg("triCentralize", args[0], SelectorType::Tri, parse_triangle_selector);
                return make_mod(bm);
            },
        };
        m["quadCentralize"] = BuiltinFunction{
            centralize_mod_check_call,
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::QuadCentralize};
                if (args.size() == 1) bm.form_sel = resolve_selector_arg("quadCentralize", args[0], SelectorType::Quad, parse_quad_selector);
                return make_mod(bm);
            },
        };

        m["centralize"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() < 1)
                    throw std::runtime_error("cleg: '" + callee + "' expects at least 1 argument(s), got " + std::to_string(arg_types.size()));
                for (size_t i = 0; i < arg_types.size(); i++)
                    if (arg_types[i].kind != CTKind::Sel && arg_types[i].kind != CTKind::String && arg_types[i].kind != CTKind::Set)
                        throw std::runtime_error("cleg: '" + callee + "' argument " + std::to_string(i + 1) + ": expected sel, string, or set, got " + type_to_string(arg_types[i]));
                return MOD_TYPE;
            },
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                BoardModifier bm{ModifierKind::Centralize};
                for (auto& a : args) bm.form_sels.push_back(resolve_any_kind_selector_arg("centralize", a));
                return make_mod(bm);
            },
        };

        m["modify"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(MOD_TYPE)}, EGR_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                std::vector<BoardModifier> mods;
                for (auto& v : args[0].arr_v) mods.push_back(v.mod_v);
                return make_egr(apply_modifiers(*args[1].egr_v, mods));
            },
        };

        m["msAll"] = BuiltinFunction{
            fixed_signature({}, MSEL_TYPE), [](const std::vector<ClegValue>&, UserFuncTable&) { return make_msel(MultiSelector{MSelOp::All}); },
        };
        m["msBase"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 2)
                    throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Number)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
                if (arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
                return MSEL_TYPE;
            },
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                double number = args[0].number;
                if (number != std::floor(number) || number < 0)
                    throw std::runtime_error("cleg: msBase: number must be a nonnegative integer, got " + format_number_display(number));
                MultiSelector ms; ms.op = MSelOp::Base; ms.number = static_cast<int>(number);
                ms.sel = std::make_shared<Selector>(resolve_any_kind_selector_arg("msBase", args[1]));
                return make_msel(ms);
            },
        };
        m["msUnion"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(MSEL_TYPE)}}, MSEL_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                MultiSelector ms; ms.op = MSelOp::Union;
                for (auto& v : args[0].arr_v) ms.items.push_back(v.msel_v);
                return make_msel(ms);
            },
        };
        m["msInter"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(MSEL_TYPE)}}, MSEL_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                MultiSelector ms; ms.op = MSelOp::Inter;
                for (auto& v : args[0].arr_v) ms.items.push_back(v.msel_v);
                return make_msel(ms);
            },
        };
        m["msDiff"] = BuiltinFunction{
            fixed_signature({MSEL_TYPE, MSEL_TYPE}, MSEL_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                MultiSelector ms; ms.op = MSelOp::Diff;
                ms.a = std::make_shared<MultiSelector>(args[0].msel_v);
                ms.b = std::make_shared<MultiSelector>(args[1].msel_v);
                return make_msel(ms);
            },
        };
        m["multiProd"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(EGR_TYPE)}, MSEL_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable&) {
                std::vector<BoardConfig> boards;
                for (auto& v : args[0].arr_v) boards.push_back(*v.egr_v);
                if (boards.empty()) throw std::runtime_error("cleg: multiProd: boards must be non-empty");
                FullProductIndex fpi = make_full_product_index(boards);
                auto result = eval_multi_selector(boards, fpi, args[1].msel_v);
                return make_egr(std::move(result.bc));
            },
        };

        // Mirrors shared/clegEval.ts's subHcublatB(bounds, cond): a "sub-region" of an N-dimensional
        // hypercubical lattice - `bounds` is an N-length array of `[lo, hi]` pairs (inclusive
        // bounds, one pair per dimension, describing the bounding hyperrectangle - not necessarily
        // integers themselves: `lo` is rounded UP (std::ceil) and `hi` rounded DOWN (std::floor) to
        // the nearest integer lattice point before use, so a non-integer bound just trims the
        // lattice down to the integer points genuinely inside `[lo, hi]` rather than being
        // rejected); `cond` decides which lattice points inside it actually become nodes, called once per candidate
        // point (as that point's own N ABSOLUTE coordinates, a number[]) via call_user_function -
        // fill_holes lets `cond` be a plain reference OR a partial application, same as any other
        // func-typed argument call. Surviving nodes keep the plain grid adjacency (connected iff
        // their coordinates differ by exactly 1 in exactly one dimension), via the same full-
        // lattice-index/stride bookkeeping as this file's own hypercuboid_board (board_config.cpp)
        // and its own TS-side analog, hypercuboidBoard, to avoid an O(survivors^2) adjacency scan.
        // ONE deviation from the TS side, scoped to the embedding only (see game/cleg_base.h's own
        // top comment on why a deviation like this is sometimes needed): TS re-centers the final
        // embedding by subtracting each dimension's own (generally negative, or a half-integer)
        // midpoint, but BoardConfig::embed here is exact non-negative-integer-only - so, exactly
        // like hypercuboid_board's own convention, each surviving node's embedding position is
        // simply its own LOCAL (0-based, always a non-negative integer) lattice coordinate, never
        // `lo`-shifted or re-centered; `cond` itself is still always called with the point's real
        // ABSOLUTE coordinates (`lo[i] + local[i]`), exactly matching the TS side.
        m["subHcublatB"] = BuiltinFunction{
            fixed_signature(
                {
                    ClegType{CTKind::Array, std::make_shared<ClegType>(ClegType{CTKind::Array, std::make_shared<ClegType>(NUMBER_TYPE)})},
                    ClegType{
                        CTKind::Func, nullptr,
                        {ClegType{CTKind::Array, std::make_shared<ClegType>(NUMBER_TYPE)}},
                        std::make_shared<ClegType>(ClegType{CTKind::Bool, nullptr}),
                    },
                },
                EGR_TYPE),
            [](const std::vector<ClegValue>& args, UserFuncTable& funcs) {
                const auto& bounds_arr = args[0].arr_v;
                size_t k = bounds_arr.size();
                if (k == 0) throw std::runtime_error("cleg: 'subHcublatB' bounds must be non-empty");
                std::vector<int> lo(k), dims(k);
                for (size_t i = 0; i < k; i++) {
                    const auto& pair = bounds_arr[i].arr_v;
                    if (pair.size() != 2)
                        throw std::runtime_error(
                            "cleg: 'subHcublatB' bounds[" + std::to_string(i) + "] must have exactly 2 entries "
                            "(lower, upper), got " + std::to_string(pair.size()));
                    // `lo`/`hi` need not themselves be integers - rounded to the nearest integer
                    // lattice point INWARD (lo up, hi down) before use, so e.g. [0.5, 2.5] becomes
                    // the integer range [1, 2], not an error.
                    double a = std::ceil(pair[0].number), b = std::floor(pair[1].number);
                    if (a > b)
                        throw std::runtime_error(
                            "cleg: 'subHcublatB' bounds[" + std::to_string(i) + "] has no integer lattice point in "
                            "range after rounding (lower up, upper down), got [" + format_number_display(a) + ", " +
                            format_number_display(b) + "]");
                    lo[i] = static_cast<int>(a);
                    dims[i] = static_cast<int>(b - a) + 1;
                }
                const ClegValue& cond = args[1];

                std::vector<long long> strides(k);
                strides[0] = 1;
                for (size_t i = 1; i < k; i++) strides[i] = strides[i - 1] * dims[i - 1];
                long long full_n = 1;
                for (int d : dims) full_n *= d;
                auto local_coords_of = [&](long long n) {
                    std::vector<int> coords(k);
                    for (size_t i = 0; i < k; i++) { coords[i] = static_cast<int>(n % dims[i]); n /= dims[i]; }
                    return coords;
                };

                // Only surviving (cond-kept) nodes get a board index (compacted, in ascending
                // full-lattice-index order) - board_idx_of maps a full-lattice index to that
                // compacted index, absent for a point cond rejected.
                std::unordered_map<long long, int> board_idx_of;
                std::vector<std::vector<int>> surviving_local;
                std::vector<std::vector<unsigned>> pos;
                for (long long n = 0; n < full_n; n++) {
                    std::vector<int> local = local_coords_of(n);
                    ClegValue point_arg; point_arg.kind = CTKind::Array; point_arg.elem = NUMBER_TYPE;
                    for (size_t i = 0; i < k; i++) point_arg.arr_v.push_back(make_number(local[i] + lo[i]));
                    bool keep = call_user_function(
                        *funcs.at(cond.func_name), fill_holes(cond.func_bound_args, {point_arg}), funcs).boolean;
                    if (!keep) continue;
                    board_idx_of[n] = static_cast<int>(surviving_local.size());
                    surviving_local.push_back(local);
                    std::vector<unsigned> p(k);
                    for (size_t i = 0; i < k; i++) p[i] = static_cast<unsigned>(local[i]);
                    pos.push_back(std::move(p));
                }
                int N = static_cast<int>(surviving_local.size());

                auto adj = zero_adj(N);
                for (int bi = 0; bi < N; bi++) {
                    const auto& local = surviving_local[bi];
                    for (size_t i = 0; i < k; i++)
                        for (int delta : {1, -1}) {
                            int nc = local[i] + delta;
                            if (nc < 0 || nc >= dims[i]) continue;
                            std::vector<int> nlocal = local;
                            nlocal[i] = nc;
                            long long flat = 0;
                            for (size_t j = 0; j < k; j++) flat += nlocal[j] * strides[j];
                            auto it = board_idx_of.find(flat);
                            if (it == board_idx_of.end()) continue;
                            adj[bi][it->second] = 1;
                        }
                }
                return make_egr(BoardConfig{N, std::move(adj), static_cast<unsigned>(k), std::move(pos)});
            },
        };

        return m;
    }();
    return table;
}

// ── Evaluation ───────────────────────────────────────────────────────────────

// Mirrors shared/clegEval.ts's ValueEnv - same raw-pointer parent-chain convention as game/cleg_check.cpp's
// own TypeEnv.
struct ValueEnv {
    std::unordered_map<std::string, ClegValue> vars;
    ValueEnv* parent = nullptr;
};
static ClegValue* lookup_value_ptr(ValueEnv& env, const std::string& name) {
    for (ValueEnv* e = &env; e; e = e->parent) {
        auto it = e->vars.find(name);
        if (it != e->vars.end()) return &it->second;
    }
    return nullptr;
}
// Unreachable in a program that has passed typecheck_cleg - every Identifier there already resolved
// to a declared variable. Mirrors shared/clegEval.ts's lookupValue().
static ClegValue lookup_value(ValueEnv& env, const std::string& name) {
    ClegValue* v = lookup_value_ptr(env, name);
    if (!v) throw std::runtime_error("cleg: undeclared variable '" + name + "'");
    return *v;
}
// Mirrors shared/clegEval.ts's setValue() - mutates `name`'s existing binding in whichever env of the
// chain declared it.
static void set_value(ValueEnv& env, const std::string& name, ClegValue value) {
    ClegValue* v = lookup_value_ptr(env, name);
    if (!v) throw std::runtime_error("cleg: undeclared variable '" + name + "'");
    *v = std::move(value);
}

// Deep-clones an array value's own array structure (recursively, for a nested `T[][]`) so indexed
// mutation (`arr[i] = x;`, see eval_stmt's own AssignStmt case) can never be observed through
// another variable that was previously assigned `= arr` or received it as a function argument -
// this language's arrays are value types, not shared references (see shared/clegBase.ts's own top
// comment). Every other ClegValue kind either can't be mutated in place at all, or (an array
// ELEMENT that isn't itself an array - a number/egr/sel/set/...) can only ever be wholesale
// REPLACED via an indexed assignment, never mutated internally - so only the array *structure*
// itself needs a fresh copy, not every value reachable from it; called at every site a value is
// bound to a (potentially long-lived, aliasable) variable - VarDecl's init, a whole-value
// AssignStmt, and a function argument's own param binding - a no-op passthrough for anything that
// isn't (or doesn't contain) an array.
static ClegValue clone_array_value(const ClegValue& v) {
    if (v.kind != CTKind::Array) return v;
    ClegValue out = v;
    out.arr_v.clear();
    out.arr_v.reserve(v.arr_v.size());
    for (auto& e : v.arr_v) out.arr_v.push_back(clone_array_value(e));
    return out;
}

// Validates that `idx` is a usable index into an array of `length` elements, returning it as a
// size_t - shared by eval_expr's own IndexExpr (read) case and eval_stmt's own indexed AssignStmt
// (write) case, so both report an out-of-bounds index the same way.
static size_t validate_array_index(double idx, size_t length) {
    if (idx != std::floor(idx) || idx < 0 || idx >= static_cast<double>(length))
        throw std::runtime_error(
            "cleg: array index " + format_number_display(idx) + " out of bounds for array of length " +
            std::to_string(length));
    return static_cast<size_t>(idx);
}

// Mirrors shared/clegEval.ts's fillHoles() - interleaves `supplied_args` (in order) into `bound_args`'
// own nullopt ("still uninstantiated") slots, producing the full argument list the original function
// actually needs - used by eval_expr's own CallExpr case whenever it calls through a func value,
// whether that value is a plain function-pointer reference (every slot nullopt, so this is just
// `supplied_args` unchanged) or a partial application (see ClegValue's own 'func' doc comment) - one
// shared interleaving rule for both, rather than treating them as two different cases.
static std::vector<ClegValue> fill_holes(
    const std::vector<std::optional<ClegValue>>& bound_args, const std::vector<ClegValue>& supplied_args)
{
    std::vector<ClegValue> out;
    out.reserve(bound_args.size());
    size_t i = 0;
    for (auto& b : bound_args) out.push_back(b ? *b : supplied_args[i++]);
    return out;
}

// Mirrors shared/clegEval.ts's ReturnSignal - thrown to unwind out of nested blocks/if-statements on
// `return`, always caught by call_user_function below.
struct ReturnSignal { ClegValue value; };
// Mirrors shared/clegEval.ts's BreakSignal/ContinueSignal - thrown by BreakStmt/ContinueStmt, caught by
// the innermost enclosing ForStmt/WhileStmt's own try/catch (see eval_stmt's own cases). A
// ReturnSignal thrown from inside a loop body is neither of these, so it passes through untouched.
struct BreakSignal {};
struct ContinueSignal {};

static void eval_block(const Stmt& block, ValueEnv* parent, UserFuncTable& funcs);
static void eval_stmt(const Stmt& stmt, ValueEnv& env, UserFuncTable& funcs);
static ClegValue eval_expr(const Expr& expr, ValueEnv& env, UserFuncTable& funcs);
static ClegValue call_user_function(const FunctionDecl& fn, const std::vector<ClegValue>& args, UserFuncTable& funcs);

// Mirrors shared/clegEval.ts's mergeBoundArgs() - merges a partial-application CallExpr's own `args` (at
// least one is a HoleExpr) into `bound_args`' own currently-open (nullopt) slots, in order -
// evaluating each non-hole argument now (once, eagerly) and leaving each hole slot open, producing
// the NEW bound_args for the resulting (possibly still-partial) closure. Starting from a fresh
// all-nullopt bound_args (a bare top-level function name, nothing bound yet) or an existing value's
// own bound_args (a plain pointer, still all-nullopt, or an already-partial closure) is the exact
// same operation, just a different starting point.
static std::vector<std::optional<ClegValue>> merge_bound_args(
    const std::vector<std::optional<ClegValue>>& bound_args, const std::vector<Expr>& args,
    ValueEnv& env, UserFuncTable& funcs)
{
    std::vector<std::optional<ClegValue>> out;
    out.reserve(bound_args.size());
    size_t j = 0;
    for (auto& b : bound_args) {
        if (b) { out.push_back(b); continue; }
        const Expr& a = args[j++];
        if (a.kind == ExprKind::HoleExpr) out.push_back(std::nullopt);
        else out.push_back(clone_array_value(eval_expr(a, env, funcs)));
    }
    return out;
}

static void eval_block(const Stmt& block, ValueEnv* parent, UserFuncTable& funcs) {
    ValueEnv env; env.parent = parent;
    for (auto& stmt : block.stmts) eval_stmt(stmt, env, funcs);
}

static void eval_stmt(const Stmt& stmt, ValueEnv& env, UserFuncTable& funcs) {
    switch (stmt.kind) {
        case StmtKind::VarDecl:
            env.vars[stmt.name] = clone_array_value(eval_expr(*stmt.expr, env, funcs));
            return;
        case StmtKind::AssignStmt: {
            ClegValue value = clone_array_value(eval_expr(*stmt.expr, env, funcs));
            if (stmt.indices.empty()) {
                set_value(env, stmt.name, std::move(value));
                return;
            }
            // Walk down into the array named `stmt.name`, following every index but the last (each
            // of which - check_stmt's own AssignStmt case already guarantees - lands on another
            // array), then mutate the final slot in place. `target` points into the SAME object
            // stored in `env` (lookup_value_ptr never copies), which is safe to mutate directly
            // precisely because clone_array_value already guarantees nothing else aliases it (see
            // that function's own doc comment).
            ClegValue* target = lookup_value_ptr(env, stmt.name);
            // Unreachable in a program that has passed typecheck_cleg - see lookup_value's own comment.
            if (!target) throw std::runtime_error("cleg: undeclared variable '" + stmt.name + "'");
            for (size_t i = 0; i + 1 < stmt.indices.size(); i++) {
                double idx_value = eval_expr(stmt.indices[i], env, funcs).number;
                size_t idx = validate_array_index(idx_value, target->arr_v.size());
                target = &target->arr_v[idx];
            }
            double last_idx_value = eval_expr(stmt.indices.back(), env, funcs).number;
            size_t last_idx = validate_array_index(last_idx_value, target->arr_v.size());
            target->arr_v[last_idx] = std::move(value);
            return;
        }
        case StmtKind::IfStmt: {
            ClegValue cond = eval_expr(*stmt.cond, env, funcs);
            if (cond.boolean) eval_block(*stmt.then_stmt, &env, funcs);
            else if (stmt.else_stmt) {
                if (stmt.else_stmt->kind == StmtKind::Block) eval_block(*stmt.else_stmt, &env, funcs);
                else eval_stmt(*stmt.else_stmt, env, funcs);
            }
            return;
        }
        case StmtKind::ForStmt: {
            // One scope for the whole loop (init's own variable, if any, persists across every
            // iteration) - body gets its own further-nested scope each iteration via eval_block, same
            // as any other BLOCK - see ForStmt's own doc comment (shared/clegBase.ts). The try/catch
            // around eval_block is BreakStmt/ContinueStmt's own unwind target - `continue` still runs
            // `for_update` before the next `cond` check, exactly like real C++; `break` skips straight
            // past the loop entirely, never running `for_update` again.
            ValueEnv loop_env; loop_env.parent = &env;
            if (stmt.for_init) eval_stmt(*stmt.for_init, loop_env, funcs);
            while (!stmt.cond || eval_expr(*stmt.cond, loop_env, funcs).boolean) {
                try {
                    eval_block(*stmt.body, &loop_env, funcs);
                } catch (BreakSignal&) {
                    break;
                } catch (ContinueSignal&) {
                    // fall through to for_update below
                }
                if (stmt.for_update) eval_stmt(*stmt.for_update, loop_env, funcs);
            }
            return;
        }
        case StmtKind::WhileStmt: {
            // Same BreakStmt/ContinueStmt unwind target as ForStmt above - see its own comment.
            while (eval_expr(*stmt.cond, env, funcs).boolean) {
                try {
                    eval_block(*stmt.body, &env, funcs);
                } catch (BreakSignal&) {
                    break;
                } catch (ContinueSignal&) {
                    // next iteration
                }
            }
            return;
        }
        case StmtKind::BreakStmt:
            throw BreakSignal{};
        case StmtKind::ContinueStmt:
            throw ContinueSignal{};
        case StmtKind::ReturnStmt:
            throw ReturnSignal{eval_expr(*stmt.expr, env, funcs)};
        case StmtKind::ExprStmt:
            eval_expr(*stmt.expr, env, funcs);
            return;
        case StmtKind::Block:
            eval_block(stmt, &env, funcs);
            return;
    }
}

static ClegValue eval_expr(const Expr& expr, ValueEnv& env, UserFuncTable& funcs) {
    switch (expr.kind) {
        case ExprKind::NumberLit: return make_number(expr.number_value);
        case ExprKind::StringLit: return make_string(expr.string_value);
        case ExprKind::BoolLit: return make_bool(expr.bool_value);
        case ExprKind::Identifier: {
            ClegValue* v = lookup_value_ptr(env, expr.string_value);
            if (v) return *v;
            // Not a variable - a bare reference to one of program's own top-level functions, used as
            // a function-pointer value (check_expr already confirmed this resolves and is func-typed).
            const FunctionDecl* fn = funcs.at(expr.string_value);
            ClegValue fv; fv.kind = CTKind::Func; fv.func_name = expr.string_value;
            for (auto& p : fn->params) { fv.func_params.push_back(p.type); fv.func_bound_args.push_back(std::nullopt); }
            fv.func_return_type = fn->return_type;
            return fv;
        }
        case ExprKind::ArrayLit: {
            std::vector<ClegValue> values;
            for (auto& e : expr.elements) values.push_back(eval_expr(e, env, funcs));
            // typecheck_cleg already rejected an empty or mixed-element-type literal, so the first
            // value's own type is always the array's element type.
            ClegValue v; v.kind = CTKind::Array; v.elem = cleg_value_type(values[0]); v.arr_v = std::move(values);
            return v;
        }
        case ExprKind::SetLit: {
            std::vector<ClegValue> values;
            for (auto& e : expr.elements) values.push_back(eval_expr(e, env, funcs));
            // Same evaluation-order hazard as game/cleg_parser.cpp's own parse_expr precedence chain
            // (see its own comment) - cleg_value_type(values[0]) must be computed in its own
            // statement, before std::move(values) is ever handed to make_cleg_set as a sibling
            // argument.
            ClegType elem = cleg_value_type(values[0]);
            return make_cleg_set(std::move(elem), std::move(values));
        }
        case ExprKind::CallExpr: {
            if (std::any_of(expr.elements.begin(), expr.elements.end(),
                             [](const Expr& a) { return a.kind == ExprKind::HoleExpr; })) {
                // Partial application - check_expr already confirmed expr.string_value names either
                // a top-level function or a local func-typed variable (never a builtin) whenever any
                // arg is '#'. merge_bound_args evaluates each non-hole argument now (once, eagerly).
                // Starting from a variable's own bound_args (rather than a fresh all-nullopt one) is
                // what lets this further-apply an already-partial closure.
                ClegValue* var_value = lookup_value_ptr(env, expr.string_value);
                const std::string& name = var_value ? var_value->func_name : expr.string_value;
                const FunctionDecl* fn = funcs.at(name);
                std::vector<std::optional<ClegValue>> starting_bound_args = var_value
                    ? var_value->func_bound_args
                    : std::vector<std::optional<ClegValue>>(fn->params.size());
                auto bound_args = merge_bound_args(starting_bound_args, expr.elements, env, funcs);
                ClegValue fv; fv.kind = CTKind::Func; fv.func_name = name; fv.func_bound_args = bound_args;
                for (size_t i = 0; i < fn->params.size(); i++) if (!bound_args[i]) fv.func_params.push_back(fn->params[i].type);
                fv.func_return_type = fn->return_type;
                return fv;
            }
            std::vector<ClegValue> args;
            for (auto& a : expr.elements) args.push_back(eval_expr(a, env, funcs));
            auto& builtins = builtin_functions();
            auto bit = builtins.find(expr.string_value);
            if (bit != builtins.end()) return bit->second.call(args, funcs);
            // A local variable of func type shadows a same-named top-level function - see
            // check_expr's own CallExpr case, which already required this to resolve the same way.
            // fill_holes handles a plain (never-partially-applied) function value transparently,
            // since its own bound_args is all nullopt.
            ClegValue* var_value = lookup_value_ptr(env, expr.string_value);
            if (var_value) return call_user_function(*funcs.at(var_value->func_name), fill_holes(var_value->func_bound_args, args), funcs);
            return call_user_function(*funcs.at(expr.string_value), args, funcs);
        }
        case ExprKind::BinaryExpr: {
            // `&&`/`||` short-circuit here, before ever reaching binary_operator_overloads() below -
            // see logical_overload's own doc comment (game/cleg_base.cpp).
            if (expr.op == "&&" || expr.op == "||") {
                bool l = eval_expr(*expr.left, env, funcs).boolean;
                if (expr.op == "&&") return l ? eval_expr(*expr.right, env, funcs) : make_bool(false);
                return l ? make_bool(true) : eval_expr(*expr.right, env, funcs);
            }
            ClegValue l = eval_expr(*expr.left, env, funcs);
            ClegValue r = eval_expr(*expr.right, env, funcs);
            ClegType lt = cleg_value_type(l), rt = cleg_value_type(r);
            for (auto& overload : binary_operator_overloads().at(expr.op)) {
                auto m = overload.match(lt, rt);
                if (m) return m->eval(l, r);
            }
            // Unreachable in a program that has passed typecheck_cleg.
            throw std::runtime_error("cleg: operator '" + expr.op + "' has no overload for these operand types at runtime");
        }
        case ExprKind::UnaryExpr:
            if (expr.op == "-") return make_number(-eval_expr(*expr.operand, env, funcs).number);
            return make_bool(!eval_expr(*expr.operand, env, funcs).boolean);
        case ExprKind::NilExpr: {
            ClegValue v; v.kind = CTKind::Array; v.elem = expr.nil_type;
            return v;
        }
        case ExprKind::IndexExpr: {
            ClegValue arr = eval_expr(*expr.left, env, funcs);
            double idx_value = eval_expr(*expr.right, env, funcs).number;
            size_t idx = validate_array_index(idx_value, arr.arr_v.size());
            return arr.arr_v[idx];
        }
    }
    throw std::runtime_error("cleg: eval_expr: unexpected ExprKind");
}

static ClegValue call_user_function(const FunctionDecl& fn, const std::vector<ClegValue>& args, UserFuncTable& funcs) {
    ValueEnv env;
    // clone_array_value here (not just at the call site's own VarDecl/AssignStmt) is what makes an
    // array argument a genuine value-copy rather than a reference to the caller's own array, exactly
    // like passing one to another variable - see clone_array_value's own doc comment.
    for (size_t i = 0; i < fn.params.size(); i++) env.vars[fn.params[i].name] = clone_array_value(args[i]);
    try {
        eval_block(fn.body, &env, funcs);
    } catch (ReturnSignal& r) {
        return std::move(r.value);
    }
    throw std::runtime_error("cleg: function '" + fn.name + "' fell off its own end without a 'return'");
}

// Mirrors shared/clegEval.ts's runClegProgram() - always re-typechecks even though every public entry
// point here (typecheck_cleg_as_board, build_board_from_cleg) already does too on its own path, same
// "cheap relative to actually evaluating" reasoning as the TS side's own doc comment.
static ClegValue run_cleg_program(const ClegProgram& program) {
    typecheck_cleg(program);
    UserFuncTable funcs;
    for (auto& fn : program.functions) funcs[fn.name] = &fn;

    ValueEnv env;
    std::optional<ClegValue> result;
    for (auto& stmt : program.stmts) {
        if (stmt.kind == StmtKind::ExprStmt) result = eval_expr(*stmt.expr, env, funcs);
        else eval_stmt(stmt, env, funcs);
    }
    return std::move(*result); // typecheck_cleg already required the last top-level statement to be an ExprStmt
}

// ── Public API ───────────────────────────────────────────────────────────────

BoardConfig build_board_from_cleg(const ClegProgram& program) {
    typecheck_cleg_as_board(program);
    ClegValue result = run_cleg_program(program);
    return std::move(*result.egr_v);
}
