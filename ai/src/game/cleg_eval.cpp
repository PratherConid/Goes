#include "game/cleg_eval.h"
#include "game/cleg_eval_internal.h"
#include "game/cleg_check.h"
#include "game/topology.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// Mirrors shared/clegEval.ts (see game/cleg_eval.h's top comment). The TS comments are the canonical
// reference; comments here only note where the port differs.
//   - ValueEnv's (and game/cleg_check.cpp's TypeEnv's) parent chain uses raw pointers into the
//     enclosing call's stack frames (safe: every env is stack-allocated and never outlives the
//     recursive check/eval call that owns it) rather than TS's GC'd object graph.

// ── Predefined (board-construction) functions ─────────────────────────────────

// Defined in the Evaluation section below; forward-declared for the subHcublatB builders here.
static ClegValue call_user_function(const FunctionDecl& fn, const std::vector<ClegValue>& args, GlobalInfo& globals);
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
// Mirrors shared/clegEval.ts's SIMP_TYPE.
static const ClegType SIMP_TYPE{CTKind::Simp, nullptr};
static const ClegType QUAD_TYPE{CTKind::Quad, nullptr};
static const ClegType MOD_TYPE{CTKind::Mod, nullptr};
// Mirrors shared/clegEval.ts's FORMSEL_TYPE.
static const ClegType FORMSEL_TYPE{CTKind::Formsel, nullptr};
// Mirrors shared/clegEval.ts's LRS_TYPE.
static const ClegType LRS_TYPE{CTKind::Lrs, nullptr};
static const ClegType PSEL_TYPE{CTKind::Psel, nullptr};

// Mirrors shared/clegEval.ts's HCUBLAT_BOUNDS_TYPE/HCUBLAT_POINT_TYPE/HCUBLAT_COND_TYPE/
// HCUBLAT_ECOND_TYPE/HCUBLAT_ADJFN_TYPE.
static const ClegType HCUBLAT_BOUNDS_TYPE{
    CTKind::Array, std::make_shared<ClegType>(ClegType{CTKind::Array, std::make_shared<ClegType>(NUMBER_TYPE)})};
static const ClegType HCUBLAT_POINT_TYPE{CTKind::Array, std::make_shared<ClegType>(NUMBER_TYPE)};
static const ClegType HCUBLAT_COND_TYPE{
    CTKind::Func, nullptr, {HCUBLAT_POINT_TYPE}, std::make_shared<ClegType>(ClegType{CTKind::Bool, nullptr})};
static const ClegType HCUBLAT_ECOND_TYPE{
    CTKind::Func, nullptr, {HCUBLAT_POINT_TYPE, HCUBLAT_POINT_TYPE},
    std::make_shared<ClegType>(ClegType{CTKind::Bool, nullptr})};
static const ClegType HCUBLAT_ADJFN_TYPE{
    CTKind::Func, nullptr, {HCUBLAT_POINT_TYPE},
    std::make_shared<ClegType>(ClegType{CTKind::Array, std::make_shared<ClegType>(HCUBLAT_POINT_TYPE)})};

// Mirrors shared/clegBase.ts's SET_ELEM_KINDS membership test, on CTKind (game/cleg_parser.cpp's
// set_elem_kind_words() is the keyword-string counterpart).
bool is_set_elem_kind(CTKind k) {
    return k == CTKind::Number || k == CTKind::String || k == CTKind::Bool ||
           k == CTKind::Edge || k == CTKind::Simp || k == CTKind::Quad;
}

// Stands in for TS's direct template-literal interpolation of a SelectorType string (e.g. 'simp2',
// no space) in error messages.
static std::string selector_type_word(const SelectorType& t) {
    switch (t.kind) {
        case SelectorKind::Node: return "node";
        case SelectorKind::Edge: return "edge";
        case SelectorKind::Simp: return "simp" + std::to_string(t.n);
        case SelectorKind::Quad: return "quad";
    }
    throw std::runtime_error("cleg: selector_type_word: unexpected SelectorType");
}
// Mirrors shared/clegEval.ts's selectorSetElemKind().
static CTKind selector_set_elem_kind(const SelectorType& t) {
    if (t.kind == SelectorKind::Node) return CTKind::Number;
    if (t.kind == SelectorKind::Edge) return CTKind::Edge;
    if (t.kind == SelectorKind::Quad) return CTKind::Quad;
    return CTKind::Simp;
}

// Mirrors shared/clegEval.ts's setValueToSelectedVals(), but returns the whole `raw` Selector (TS's
// callers build it), filling the raw_* field (game/selector.h) that matches want_kind.
static Selector raw_selector_from_set(const SelectorType& want_kind, const std::vector<ClegValue>& values) {
    Selector sel; sel.op = SelectorOp::Raw; sel.type = want_kind;
    switch (want_kind.kind) {
        case SelectorKind::Node:
            for (auto& v : values) sel.raw_nodes.insert(static_cast<int>(std::llround(v.number)));
            break;
        case SelectorKind::Edge:
            for (auto& v : values) sel.raw_edges.push_back(v.edge_v);
            break;
        case SelectorKind::Simp:
            for (auto& v : values) sel.raw_simps.push_back(v.simp_v);
            break;
        case SelectorKind::Quad:
            for (auto& v : values) sel.raw_quads.push_back(v.quad_v);
            break;
    }
    return sel;
}

// std::function rather than a function pointer, since simpCentralize/simpCentering pass a closure
// capturing `n` (TS's inline `(s) => parseSimpSelector(n, s)`).
using SelectorParseFn = std::function<Selector(const std::string&)>;

// Mirrors shared/clegEval.ts's resolveSelectorArg().
static Selector resolve_selector_arg(
    const std::string& callee, const ClegValue& arg, const SelectorType& want_kind, SelectorParseFn parse_fn)
{
    if (arg.kind == CTKind::String) return parse_fn(arg.str);
    if (arg.kind == CTKind::Set) {
        if (arg.elem.kind != selector_set_elem_kind(want_kind))
            throw std::runtime_error(
                "cleg: '" + callee + "' expects a " + selector_type_word(want_kind) + " selector, got a set of " +
                type_to_string(arg.elem));
        if (want_kind.kind == SelectorKind::Simp)
            for (auto& v : arg.arr_v) {
                int actual_n = static_cast<int>(v.simp_v.nodes.size()) - 1;
                if (actual_n != want_kind.n)
                    throw std::runtime_error(
                        "cleg: '" + callee + "' expects a set of simp " + std::to_string(want_kind.n) +
                        " elements, got one of arity " + std::to_string(actual_n));
            }
        return raw_selector_from_set(want_kind, arg.arr_v);
    }
    // arg.kind == CTKind::Sel
    if (arg.sel_type != want_kind)
        throw std::runtime_error(
            "cleg: '" + callee + "' expects a " + selector_type_word(want_kind) + " selector, got a '" +
            selector_type_word(arg.sel_type) + "' selector");
    return arg.sel_v;
}

// One entry per shared/boardConfig.ts's PrescribedBoardMap row, standing in for TS's
// PrescribedBoard/PrescribedBoardMap/PrescribedBoardFns trio (nothing here needs the display-only
// argStr/desc). old_kind is build_board_config's dispatch string; the cleg name is old_kind + "B".
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
        {"snubsq", {BoardArgKind::Number, BoardArgKind::Number}},
        {"twsq", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"gtsq", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"star", {BoardArgKind::Number}},
        {"octa", {}},
        {"sier", {BoardArgKind::Number, BoardArgKind::Number}},
        {"simplex", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::Number}},
        {"ortho", {BoardArgKind::Number}},
        {"reg24Cell", {}},
        {"reg120Cell", {}},
        {"reg600Cell", {}},
        {"dodflake", {BoardArgKind::Number}},
        {"icoflake", {BoardArgKind::Number}},
        {"octaflake", {BoardArgKind::Number}},
        {"polyflake", {BoardArgKind::Number, BoardArgKind::Number}},
        {"cpolyflake", {BoardArgKind::Number, BoardArgKind::Number}},
        {"cpentflake", {BoardArgKind::Number}},
        {"menger", {BoardArgKind::Number, BoardArgKind::Number, BoardArgKind::ZeroOneList}},
        {"ap", {BoardArgKind::Number}},
        {"diamondCubic", {BoardArgKind::Number}},
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
            // Inlines shared/types.ts's parseBoardArgToken(ZeroOneList, ...) validation.
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

// ── multiProd: N-ary Cartesian board product, restricted by a ProdSelector ────

// Mirrors shared/clegEval.ts's selectorTypeBySetElem(), writing `out` and returning false (TS: null)
// when the kind can't be determined.
static bool selector_type_by_set_elem(CTKind k, SelectorType& out) {
    if (k == CTKind::Number) { out = SelectorType{SelectorKind::Node}; return true; }
    if (k == CTKind::Edge) { out = SelectorType{SelectorKind::Edge}; return true; }
    if (k == CTKind::Quad) { out = SelectorType{SelectorKind::Quad}; return true; }
    return false;
}

// Mirrors shared/clegEval.ts's resolveAnyKindSelectorArg().
static Selector resolve_any_kind_selector_arg(const std::string& callee, const ClegValue& arg) {
    if (arg.kind == CTKind::Sel) return arg.sel_v;
    if (arg.kind == CTKind::String) return parse_selector(arg.str);
    if (arg.kind == CTKind::Set) {
        if (arg.elem.kind == CTKind::Simp) {
            if (arg.arr_v.empty())
                throw std::runtime_error(
                    "cleg: '" + callee + "': an empty simp{} set's own arity can't be determined - pass a "
                    "non-empty set, a string, or a sel value instead");
            int n = static_cast<int>(arg.arr_v[0].simp_v.nodes.size()) - 1;
            for (auto& v : arg.arr_v) {
                int actual_n = static_cast<int>(v.simp_v.nodes.size()) - 1;
                if (actual_n != n)
                    throw std::runtime_error(
                        "cleg: '" + callee + "': a simp{} set must have uniform arity - got both simp " +
                        std::to_string(n) + " and simp " + std::to_string(actual_n));
            }
            return raw_selector_from_set(simp_type(n), arg.arr_v);
        }
        SelectorType want_kind;
        if (!selector_type_by_set_elem(arg.elem.kind, want_kind))
            throw std::runtime_error(
                "cleg: '" + callee + "': a selector set must be a set of number/edge/simp/quad, got a set of " +
                type_to_string(arg.elem));
        return raw_selector_from_set(want_kind, arg.arr_v);
    }
    throw std::runtime_error("cleg: '" + callee + "': expected sel, string, or set, got " + type_to_string(cleg_value_type(arg)));
}

// Mirrors shared/clegEval.ts's FullProductIndex/makeFullProductIndex/fullIndexOf/tupleOfFullIndex,
// with a long long index space (TS's double is exact up to 2^53) since an unrestricted product of
// several sizable boards can exceed 32-bit int range.
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
static RestrictResult restrict_board_by_selector(const BoardConfig& board, const Selector& sel, ClegRandom& random) {
    if (sel.type == SelectorType{SelectorKind::Node}) {
        auto kept = select_node(board.adj, board.embed, sel, random);
        std::vector<int> survivors(kept.begin(), kept.end()); // std::set already ascending
        return {node_induced_subgraph(board, kept), std::move(survivors)};
    }
    if (sel.type == SelectorType{SelectorKind::Edge}) {
        auto edges = select_edge(board.adj, board.embed, sel, random);
        std::set<int> kept;
        for (auto& e : edges) { kept.insert(e.n1); kept.insert(e.n2); }
        std::vector<int> survivors(kept.begin(), kept.end());
        return {edge_induced_subgraph(board, edges), std::move(survivors)};
    }
    throw std::runtime_error(
        "cleg: multiProd: psBase's own selector must be a node or edge selector, got a '" +
        selector_type_word(sel.type) + "' selector");
}

// Mirrors shared/clegEval.ts's restrictBoardByNodeAndEdgeSelector().
static RestrictResult restrict_board_by_node_and_edge_selector(
    const BoardConfig& board, const Selector& node_sel, const Selector& edge_sel, ClegRandom& random)
{
    auto extra_nodes = select_node(board.adj, board.embed, node_sel, random);
    auto edges = select_edge(board.adj, board.embed, edge_sel, random);
    std::set<int> kept = extra_nodes;
    for (auto& e : edges) { kept.insert(e.n1); kept.insert(e.n2); }
    std::vector<int> survivors(kept.begin(), kept.end());
    return {node_edge_induced_subgraph(board, extra_nodes, edges), std::move(survivors)};
}

struct BuiltResult { BoardConfig bc; std::vector<long long> orig_index; };

// Mirrors shared/clegEval.ts's edgeKey().
static std::string edge_key(long long a, long long b) {
    if (a > b) std::swap(a, b);
    return std::to_string(a) + "," + std::to_string(b);
}

// Mirrors shared/clegEval.ts's originalEdgeKeysOf().
static std::set<std::string> original_edge_keys_of(const BoardConfig& bc, const std::vector<long long>& orig_index) {
    std::set<std::string> keys;
    for (int i = 0; i < bc.N; i++)
        for (int j = i + 1; j < bc.N; j++)
            if (bc.adj[i][j]) keys.insert(edge_key(orig_index[i], orig_index[j]));
    return keys;
}

// Mirrors shared/clegEval.ts's fullProductResult().
static BuiltResult full_product_result(const std::vector<BoardConfig>& boards, const FullProductIndex& fpi) {
    BoardConfig bc = boards[0];
    for (size_t i = 1; i < boards.size(); i++) bc = product(bc, boards[i]);
    std::vector<long long> orig_index(fpi.total);
    for (long long i = 0; i < fpi.total; i++) orig_index[i] = i;
    return {std::move(bc), std::move(orig_index)};
}

// Mirrors shared/clegEval.ts's buildFromNodesAndEdges().
static BuiltResult build_from_nodes_and_edges(
    const std::vector<BoardConfig>& boards, const FullProductIndex& fpi,
    const std::set<long long>& kept_original, const std::set<std::string>& edge_keys)
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
    std::unordered_map<long long, int> local_index_of_original;
    for (size_t i = 0; i < orig_index.size(); i++) local_index_of_original[orig_index[i]] = static_cast<int>(i);
    auto adj = zero_adj(static_cast<int>(K));
    for (auto& key : edge_keys) {
        size_t comma = key.find(',');
        int a = local_index_of_original.at(std::stoll(key.substr(0, comma)));
        int b = local_index_of_original.at(std::stoll(key.substr(comma + 1)));
        adj[a][b] = 1;
        adj[b][a] = 1;
    }
    BoardConfig bc{static_cast<int>(K), std::move(adj), emb_dim, std::move(pos)};
    return {std::move(bc), std::move(orig_index)};
}

// Mirrors shared/clegEval.ts's combineRestrictedFactor().
static BuiltResult combine_restricted_factor(
    const std::vector<BoardConfig>& boards, const FullProductIndex& fpi, int number,
    const BoardConfig& restricted, const std::vector<int>& survivors)
{
    std::vector<BoardConfig> factor_boards = boards;
    factor_boards[number] = restricted;
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
        tuple[number] = survivors[tuple[number]];
        orig_index[local] = full_index_of(fpi, tuple);
    }
    return {std::move(bc), std::move(orig_index)};
}

// Mirrors shared/clegEval.ts's evalProdSelector().
static BuiltResult eval_prod_selector(
    const std::vector<BoardConfig>& boards, const FullProductIndex& fpi, const ProdSelector& psel, ClegRandom& random)
{
    switch (psel.op) {
        case PSelOp::All:
            return full_product_result(boards, fpi);
        case PSelOp::Base: {
            auto restricted = restrict_board_by_selector(boards[psel.number], *psel.sel, random);
            return combine_restricted_factor(boards, fpi, psel.number, restricted.bc, restricted.survivors);
        }
        case PSelOp::BaseNE: {
            auto restricted = restrict_board_by_node_and_edge_selector(boards[psel.number], *psel.node_sel, *psel.edge_sel, random);
            return combine_restricted_factor(boards, fpi, psel.number, restricted.bc, restricted.survivors);
        }
        case PSelOp::Union: {
            std::set<long long> nodes;
            std::set<std::string> edges;
            for (auto& item : psel.items) {
                auto r = eval_prod_selector(boards, fpi, item, random);
                nodes.insert(r.orig_index.begin(), r.orig_index.end());
                auto item_edges = original_edge_keys_of(r.bc, r.orig_index);
                edges.insert(item_edges.begin(), item_edges.end());
            }
            return build_from_nodes_and_edges(boards, fpi, nodes, edges);
        }
        case PSelOp::Inter: {
            if (psel.items.empty()) return full_product_result(boards, fpi);
            auto first = eval_prod_selector(boards, fpi, psel.items[0], random);
            std::set<long long> nodes(first.orig_index.begin(), first.orig_index.end());
            std::set<std::string> edges = original_edge_keys_of(first.bc, first.orig_index);
            for (size_t i = 1; i < psel.items.size(); i++) {
                auto next = eval_prod_selector(boards, fpi, psel.items[i], random);
                std::set<long long> next_nodes(next.orig_index.begin(), next.orig_index.end());
                std::set<long long> kept_nodes;
                for (long long idx : nodes) if (next_nodes.count(idx)) kept_nodes.insert(idx);
                nodes = std::move(kept_nodes);
                auto next_edges = original_edge_keys_of(next.bc, next.orig_index);
                std::set<std::string> kept_edges;
                for (auto& key : edges) if (next_edges.count(key)) kept_edges.insert(key);
                edges = std::move(kept_edges);
            }
            return build_from_nodes_and_edges(boards, fpi, nodes, edges);
        }
        case PSelOp::Diff: {
            auto ra = eval_prod_selector(boards, fpi, *psel.a, random);
            auto rb = eval_prod_selector(boards, fpi, *psel.b, random);
            std::set<long long> b_nodes(rb.orig_index.begin(), rb.orig_index.end());
            std::set<long long> nodes;
            for (long long idx : ra.orig_index) if (!b_nodes.count(idx)) nodes.insert(idx);
            auto a_edges = original_edge_keys_of(ra.bc, ra.orig_index);
            std::set<std::string> edges;
            for (auto& key : a_edges) {
                size_t comma = key.find(',');
                long long ea = std::stoll(key.substr(0, comma));
                long long eb = std::stoll(key.substr(comma + 1));
                if (nodes.count(ea) && nodes.count(eb)) edges.insert(key);
            }
            return build_from_nodes_and_edges(boards, fpi, nodes, edges);
        }
    }
    throw std::runtime_error("cleg: eval_prod_selector: unexpected PSelOp");
}

// Mirrors shared/clegEval.ts's parseHcublatBounds(), returning `lo`/`dims` through out-params.
static void parse_hcublat_bounds(const ClegValue& bounds_val, std::vector<int>& lo, std::vector<int>& dims) {
    const auto& bounds_arr = bounds_val.arr_v;
    size_t k = bounds_arr.size();
    if (k == 0) throw std::runtime_error("cleg: 'subHcublatB' bounds must be non-empty");
    lo.resize(k); dims.resize(k);
    for (size_t i = 0; i < k; i++) {
        const auto& pair = bounds_arr[i].arr_v;
        if (pair.size() != 2)
            throw std::runtime_error(
                "cleg: 'subHcublatB' bounds[" + std::to_string(i) + "] must have exactly 2 entries "
                "(lower, upper), got " + std::to_string(pair.size()));
        double a = std::ceil(pair[0].number), b = std::floor(pair[1].number);
        if (a > b)
            throw std::runtime_error(
                "cleg: 'subHcublatB' bounds[" + std::to_string(i) + "] has no integer lattice point in "
                "range after rounding (lower up, upper down), got [" + format_number_display(a) + ", " +
                format_number_display(b) + "]");
        lo[i] = static_cast<int>(a);
        dims[i] = static_cast<int>(b - a) + 1;
    }
}

// Mirrors shared/clegEval.ts's buildSubHcublat(), except for the embedding: BoardConfig::embed is
// non-negative-integer-only, so instead of recenterPositions() each surviving node's position
// (`pos`) is its local, 0-based lattice coordinate (hypercuboid_board's convention). `cond`/`econd`
// still receive the real absolute coordinates (`abs_pos`), as in TS.
static ClegValue build_sub_hcublat(
    const ClegValue& bounds_val, const ClegValue& cond_val,
    const std::function<bool(const std::vector<int>&, const std::vector<int>&)>& econd, GlobalInfo& globals
) {
    std::vector<int> lo, dims;
    parse_hcublat_bounds(bounds_val, lo, dims);
    size_t k = lo.size();

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

    std::unordered_map<long long, int> board_idx_of;
    std::vector<std::vector<int>> surviving_local;
    std::vector<std::vector<int>> abs_pos;
    std::vector<std::vector<unsigned>> pos;
    for (long long n = 0; n < full_n; n++) {
        std::vector<int> local = local_coords_of(n);
        std::vector<int> point(k);
        for (size_t i = 0; i < k; i++) point[i] = local[i] + lo[i];
        ClegValue point_arg; point_arg.kind = CTKind::Array; point_arg.elem = NUMBER_TYPE;
        for (int c : point) point_arg.arr_v.push_back(make_number(c));
        bool keep = call_user_function(
            *globals.funcs.at(cond_val.func_name), fill_holes(cond_val.func_bound_args, {point_arg}), globals).boolean;
        if (!keep) continue;
        board_idx_of[n] = static_cast<int>(surviving_local.size());
        surviving_local.push_back(local);
        abs_pos.push_back(std::move(point));
        std::vector<unsigned> p(k);
        for (size_t i = 0; i < k; i++) p[i] = static_cast<unsigned>(local[i]);
        pos.push_back(std::move(p));
    }
    int N = static_cast<int>(surviving_local.size());

    auto adj = zero_adj(N);
    for (int bi = 0; bi < N; bi++) {
        const auto& local = surviving_local[bi];
        for (size_t i = 0; i < k; i++) {
            int nc = local[i] + 1;
            if (nc >= dims[i]) continue;
            std::vector<int> nlocal = local;
            nlocal[i] = nc;
            long long flat = 0;
            for (size_t j = 0; j < k; j++) flat += nlocal[j] * strides[j];
            auto it = board_idx_of.find(flat);
            if (it == board_idx_of.end()) continue;
            int nbi = it->second;
            if (!econd(abs_pos[bi], abs_pos[nbi])) continue;
            adj[bi][nbi] = 1;
            adj[nbi][bi] = 1;
        }
    }
    return make_egr(BoardConfig{N, std::move(adj), static_cast<unsigned>(k), std::move(pos)});
}

// Mirrors shared/clegEval.ts's `point.join(',')` key in buildSubHcublatCustomAdj().
static std::string hcublat_point_key(const std::vector<int>& point) {
    std::string key;
    for (size_t i = 0; i < point.size(); i++) { if (i) key += ','; key += std::to_string(point[i]); }
    return key;
}

// Mirrors shared/clegEval.ts's buildSubHcublatCustomAdj(), with the same local-coordinate embedding
// deviation as build_sub_hcublat above (`cond`/`adj_fn` still receive absolute `abs_pos`).
static ClegValue build_sub_hcublat_custom_adj(
    const ClegValue& bounds_val, const ClegValue& cond_val,
    const std::function<std::vector<std::vector<int>>(const std::vector<int>&)>& adj_fn, GlobalInfo& globals
) {
    std::vector<int> lo, dims;
    parse_hcublat_bounds(bounds_val, lo, dims);
    size_t k = lo.size();

    long long full_n = 1;
    for (int d : dims) full_n *= d;
    auto local_coords_of = [&](long long n) {
        std::vector<int> coords(k);
        for (size_t i = 0; i < k; i++) { coords[i] = static_cast<int>(n % dims[i]); n /= dims[i]; }
        return coords;
    };

    std::unordered_map<std::string, int> board_idx_of_point;
    std::vector<std::vector<int>> abs_pos;
    std::vector<std::vector<unsigned>> pos;
    for (long long n = 0; n < full_n; n++) {
        std::vector<int> local = local_coords_of(n);
        std::vector<int> point(k);
        for (size_t i = 0; i < k; i++) point[i] = local[i] + lo[i];
        ClegValue point_arg; point_arg.kind = CTKind::Array; point_arg.elem = NUMBER_TYPE;
        for (int c : point) point_arg.arr_v.push_back(make_number(c));
        bool keep = call_user_function(
            *globals.funcs.at(cond_val.func_name), fill_holes(cond_val.func_bound_args, {point_arg}), globals).boolean;
        if (!keep) continue;
        board_idx_of_point[hcublat_point_key(point)] = static_cast<int>(abs_pos.size());
        std::vector<unsigned> p(k);
        for (size_t i = 0; i < k; i++) p[i] = static_cast<unsigned>(local[i]);
        abs_pos.push_back(std::move(point));
        pos.push_back(std::move(p));
    }
    int N = static_cast<int>(abs_pos.size());

    auto adj = zero_adj(N);
    for (int bi = 0; bi < N; bi++) {
        for (const auto& target : adj_fn(abs_pos[bi])) {
            auto it = board_idx_of_point.find(hcublat_point_key(target));
            if (it == board_idx_of_point.end() || it->second == bi) continue;
            adj[bi][it->second] = 1;
            adj[it->second][bi] = 1;
        }
    }
    return make_egr(BoardConfig{N, std::move(adj), static_cast<unsigned>(k), std::move(pos)});
}

// Mirrors shared/clegEval.ts's compareClegSetElems(), as a std::sort-style "less" predicate. Strings
// compare by UTF-8 byte rather than UTF-16 code unit - the same order except between U+E000-U+FFFF
// and astral characters.
static bool cleg_set_elem_less(const ClegValue& a, const ClegValue& b) {
    switch (a.kind) {
        case CTKind::Number: return a.number < b.number;
        case CTKind::String: return a.str < b.str;
        case CTKind::Bool: return !a.boolean && b.boolean;
        case CTKind::Edge:
            return std::vector<int>{a.edge_v.n1, a.edge_v.n2} < std::vector<int>{b.edge_v.n1, b.edge_v.n2};
        case CTKind::Simp: return a.simp_v.nodes < b.simp_v.nodes;
        case CTKind::Quad:
            return std::vector<int>{a.quad_v.n1, a.quad_v.n2, a.quad_v.n3, a.quad_v.n4} <
                   std::vector<int>{b.quad_v.n1, b.quad_v.n2, b.quad_v.n3, b.quad_v.n4};
        default:
            throw std::runtime_error("cleg: '" + type_to_string(cleg_value_type(a)) + "' cannot be a set element");
    }
}

// Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS - populated inside one function-local static
// initializer rather than by TS's top-level `BUILTIN_FUNCTIONS['x'] = {...}` assignments.
const std::unordered_map<std::string, BuiltinFunction>& builtin_functions() {
    static const std::unordered_map<std::string, BuiltinFunction> table = [] {
        std::unordered_map<std::string, BuiltinFunction> m;

        // Mirrors the TS PrescribedBoardMap registration loop - see prescribed_boards() above.
        for (auto& entry : prescribed_boards()) {
            std::string cleg_name = entry.old_kind + "B";
            std::vector<ClegType> params;
            for (auto k : entry.arg_kinds) params.push_back(arg_kind_to_cleg_type(k));
            std::string old_kind = entry.old_kind;
            std::vector<BoardArgKind> arg_kinds = entry.arg_kinds;
            m[cleg_name] = BuiltinFunction{
                fixed_signature(params, EGR_TYPE),
                [old_kind, arg_kinds](const std::vector<ClegValue>& args, GlobalInfo&) {
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
            [](const std::vector<ClegValue>& args, GlobalInfo&) { return make_number(static_cast<double>(args[0].arr_v.size())); },
        };

        // Mirrors shared/clegEval.ts's hasCheckCall()/BUILTIN_FUNCTIONS['has'].
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
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                std::string key = cleg_set_key(args[1]);
                for (auto& v : args[0].arr_v) if (cleg_set_key(v) == key) return make_bool(true);
                return make_bool(false);
            },
        };

        // Mirrors shared/clegEval.ts's toSetCheckCall()/BUILTIN_FUNCTIONS['toSet'].
        m["toSet"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 1)
                    throw std::runtime_error("cleg: '" + callee + "' expects 1 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Array)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected an array, got " + type_to_string(arg_types[0]));
                const ClegType& elem = *arg_types[0].elem;
                if (!is_set_elem_kind(elem.kind))
                    throw std::runtime_error(
                        "cleg: '" + callee + "' argument 1: element type " + type_to_string(elem) + " has no defined "
                        "equality - only number/string/bool/edge/tri/quad elements are supported");
                return ClegType{CTKind::Set, std::make_shared<ClegType>(elem)};
            },
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                return make_cleg_set(args[0].elem, args[0].arr_v);
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
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                double count = args[1].number;
                if (count != std::floor(count) || count < 0)
                    throw std::runtime_error("cleg: 'randRmN' count must be a nonnegative integer, got " + format_number_display(count));
                auto kept = randomly_remove(args[0].arr_v, static_cast<int>(count), cleg_set_elem_less, globals.random);
                ClegValue v; v.kind = CTKind::Set; v.elem = args[0].elem; v.arr_v = std::move(kept);
                return v;
            },
        };
        m["randRmP"] = BuiltinFunction{
            rand_rm_check_call,
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                double frac = args[1].number;
                if (!std::isfinite(frac) || frac < 0)
                    throw std::runtime_error("cleg: 'randRmP' portion must be a nonnegative number, got " + format_number_display(frac));
                int remove_count = static_cast<int>(std::floor(frac * static_cast<double>(args[0].arr_v.size())));
                auto kept = randomly_remove(args[0].arr_v, remove_count, cleg_set_elem_less, globals.random);
                ClegValue v; v.kind = CTKind::Set; v.elem = args[0].elem; v.arr_v = std::move(kept);
                return v;
            },
        };

        // Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS['randTakeN']/['randTakeP'].
        m["randTakeN"] = BuiltinFunction{
            rand_rm_check_call,
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                double count = args[1].number;
                if (count != std::floor(count) || count < 0)
                    throw std::runtime_error("cleg: 'randTakeN' count must be a nonnegative integer, got " + format_number_display(count));
                auto kept = randomly_take(args[0].arr_v, static_cast<int>(count), cleg_set_elem_less, globals.random);
                ClegValue v; v.kind = CTKind::Set; v.elem = args[0].elem; v.arr_v = std::move(kept);
                return v;
            },
        };
        m["randTakeP"] = BuiltinFunction{
            rand_rm_check_call,
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                double frac = args[1].number;
                if (!std::isfinite(frac) || frac < 0)
                    throw std::runtime_error("cleg: 'randTakeP' portion must be a nonnegative number, got " + format_number_display(frac));
                int take_count = static_cast<int>(std::floor(frac * static_cast<double>(args[0].arr_v.size())));
                auto kept = randomly_take(args[0].arr_v, take_count, cleg_set_elem_less, globals.random);
                ClegValue v; v.kind = CTKind::Set; v.elem = args[0].elem; v.arr_v = std::move(kept);
                return v;
            },
        };

        // Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS['abs']/['sqrt'].
        m["abs"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, NUMBER_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) { return make_number(std::abs(args[0].number)); },
        };
        m["sqrt"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, NUMBER_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                double v = args[0].number;
                if (v < 0) throw std::runtime_error("cleg: 'sqrt' argument must be nonnegative, got " + format_number_display(v));
                return make_number(std::sqrt(v));
            },
        };
        // Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS['pow'].
        m["pow"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE}, NUMBER_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) { return make_number(std::pow(args[0].number, args[1].number)); },
        };

        // Mirrors shared/clegEval.ts's trigFn()/BUILTIN_FUNCTIONS['sin'/'cos'/.../'csc'].
        auto trig_builtin = [](double (*f)(double)) {
            return BuiltinFunction{
                fixed_signature({NUMBER_TYPE}, NUMBER_TYPE),
                [f](const std::vector<ClegValue>& args, GlobalInfo&) { return make_number(f(args[0].number)); },
            };
        };
        m["sin"] = trig_builtin([](double x) { return std::sin(x); });
        m["cos"] = trig_builtin([](double x) { return std::cos(x); });
        m["tan"] = trig_builtin([](double x) { return std::tan(x); });
        m["cot"] = trig_builtin([](double x) { return 1.0 / std::tan(x); });
        m["sec"] = trig_builtin([](double x) { return 1.0 / std::cos(x); });
        m["csc"] = trig_builtin([](double x) { return 1.0 / std::sin(x); });

        // Mirrors shared/clegEval.ts's minMaxCheckCall().
        CheckCallFn min_max_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() == 1 && arg_types[0].kind == CTKind::Array &&
                arg_types[0].elem && arg_types[0].elem->kind == CTKind::Number)
                return NUMBER_TYPE;
            bool all_number = !arg_types.empty();
            for (auto& t : arg_types) if (t.kind != CTKind::Number) { all_number = false; break; }
            if (all_number) return NUMBER_TYPE;
            std::string joined;
            for (size_t i = 0; i < arg_types.size(); i++) {
                if (i) joined += ", ";
                joined += type_to_string(arg_types[i]);
            }
            throw std::runtime_error(
                "cleg: '" + callee + "' expects one or more number arguments, or a single number[] argument, got (" +
                joined + ")");
        };
        // Mirrors shared/clegEval.ts's makeMinMax().
        auto make_min_max = [min_max_check_call](std::string name, double (*reduce)(double, double)) {
            return BuiltinFunction{
                min_max_check_call,
                [name, reduce](const std::vector<ClegValue>& args, GlobalInfo&) {
                    std::vector<double> nums;
                    if (args.size() == 1 && args[0].kind == CTKind::Array) {
                        if (args[0].arr_v.empty())
                            throw std::runtime_error("cleg: '" + name + "' requires a nonempty array, got an empty array");
                        for (auto& v : args[0].arr_v) nums.push_back(v.number);
                    } else {
                        for (auto& v : args) nums.push_back(v.number);
                    }
                    double result = nums[0];
                    for (size_t i = 1; i < nums.size(); i++) result = reduce(result, nums[i]);
                    return make_number(result);
                },
            };
        };
        m["max"] = make_min_max("max", [](double a, double b) { return std::max(a, b); });
        m["min"] = make_min_max("min", [](double a, double b) { return std::min(a, b); });

        // Mirrors shared/clegEval.ts's rangeCheckCall()/BUILTIN_FUNCTIONS['range'].
        m["range"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() < 1 || arg_types.size() > 3)
                    throw std::runtime_error("cleg: '" + callee + "' expects 1 to 3 argument(s), got " + std::to_string(arg_types.size()));
                for (size_t i = 0; i < arg_types.size(); i++)
                    if (arg_types[i].kind != CTKind::Number)
                        throw std::runtime_error(
                            "cleg: '" + callee + "' argument " + std::to_string(i + 1) + " must be a number, got " +
                            type_to_string(arg_types[i]));
                return ClegType{CTKind::Array, std::make_shared<ClegType>(NUMBER_TYPE)};
            },
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                std::vector<double> nums;
                for (auto& a : args) nums.push_back(a.number);
                for (size_t i = 0; i < nums.size(); i++)
                    if (nums[i] != std::floor(nums[i]))
                        throw std::runtime_error(
                            "cleg: 'range' argument " + std::to_string(i + 1) + " must be an integer, got " +
                            format_number_display(nums[i]));
                double start = 0, stop, step = 1;
                if (nums.size() == 1) { stop = nums[0]; }
                else if (nums.size() == 2) { start = nums[0]; stop = nums[1]; }
                else { start = nums[0]; stop = nums[1]; step = nums[2]; }
                if (step == 0) throw std::runtime_error("cleg: 'range' step must not be zero");
                ClegValue out; out.kind = CTKind::Array; out.elem = NUMBER_TYPE;
                if (step > 0) for (double n = start; n < stop; n += step) out.arr_v.push_back(make_number(n));
                else for (double n = start; n > stop; n += step) out.arr_v.push_back(make_number(n));
                return out;
            },
        };

        m["mkEdge"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE}, EDGE_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                return make_edge_v(make_board_edge(static_cast<int>(args[0].number), static_cast<int>(args[1].number)));
            },
        };
        // Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS['mkTri'].
        m["mkTri"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE}, SIMP_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                return make_simp_v(make_board_simplex(
                    {static_cast<int>(args[0].number), static_cast<int>(args[1].number), static_cast<int>(args[2].number)}));
            },
        };
        // Mirrors shared/clegEval.ts's mkSimpCheckCall()/BUILTIN_FUNCTIONS['mkSimp'].
        m["mkSimp"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() < 3)
                    throw std::runtime_error(
                        "cleg: '" + callee + "' expects at least 3 arguments (a simplex needs >= 2 arity, "
                        "i.e. >= 3 nodes), got " + std::to_string(arg_types.size()));
                for (size_t i = 0; i < arg_types.size(); i++)
                    if (arg_types[i].kind != CTKind::Number)
                        throw std::runtime_error(
                            "cleg: '" + callee + "' argument " + std::to_string(i + 1) + " must be a number, got " +
                            type_to_string(arg_types[i]));
                return SIMP_TYPE;
            },
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                std::vector<int> nodes;
                for (auto& a : args) nodes.push_back(static_cast<int>(a.number));
                return make_simp_v(make_board_simplex(std::move(nodes)));
            },
        };
        m["mkQuad"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE}, QUAD_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                return make_quad_v(make_board_quad(
                    static_cast<int>(args[0].number), static_cast<int>(args[1].number),
                    static_cast<int>(args[2].number), static_cast<int>(args[3].number)));
            },
        };

        m["prod"] = BuiltinFunction{
            fixed_signature({EGR_TYPE, EGR_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) { return make_egr(product(*args[0].egr_v, *args[1].egr_v)); },
        };

        m["mkSel"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 1)
                    throw std::runtime_error("cleg: '" + callee + "' expects 1 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::String && arg_types[0].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected string or set, got " + type_to_string(arg_types[0]));
                return ClegType{CTKind::Sel, nullptr};
            },
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                Selector sel = resolve_any_kind_selector_arg("mkSel", args[0]);
                ClegValue v; v.kind = CTKind::Sel; v.sel_type = sel.type; v.sel_v = std::move(sel);
                return v;
            },
        };

        m["rectify"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&, GlobalInfo&) { return make_mod(BoardModifier{ModifierKind::Rectify}); },
        };
        m["truncate"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&, GlobalInfo&) { return make_mod(BoardModifier{ModifierKind::Truncate}); },
        };
        m["globalCentralize"] = BuiltinFunction{
            fixed_signature({}, MOD_TYPE), [](const std::vector<ClegValue>&, GlobalInfo&) { return make_mod(BoardModifier{ModifierKind::GlobalCentralize}); },
        };
        m["edgeSplit"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::EdgeSplit}; bm.split_n = static_cast<int>(args[0].number);
                return make_mod(bm);
            },
        };
        m["mergeClose"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::MergeClose}; bm.dist = args[0].number;
                return make_mod(bm);
            },
        };
        m["scale"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE}, MOD_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
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
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::NodeInducedSubgraph};
                bm.sel = resolve_selector_arg("nis", args[0], SelectorType{SelectorKind::Node}, parse_node_selector);
                return make_mod(bm);
            },
        };
        m["eis"] = BuiltinFunction{
            induced_subgraph_mod_check_call,
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::EdgeInducedSubgraph};
                bm.sel = resolve_selector_arg("eis", args[0], SelectorType{SelectorKind::Edge}, parse_edge_selector);
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
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                Selector sel = resolve_selector_arg("selectNode", args[0], SelectorType{SelectorKind::Node}, parse_node_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto nodes = select_node(bc.adj, bc.embed, sel, globals.random);
                std::vector<ClegValue> values;
                for (int n : nodes) values.push_back(make_number(n));
                return make_cleg_set(NUMBER_TYPE, std::move(values));
            },
        };
        m["selectEdge"] = BuiltinFunction{
            select_set_check_call(EDGE_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                Selector sel = resolve_selector_arg("selectEdge", args[0], SelectorType{SelectorKind::Edge}, parse_edge_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto edges = select_edge(bc.adj, bc.embed, sel, globals.random);
                std::vector<ClegValue> values;
                for (auto& e : edges) values.push_back(make_edge_v(e));
                return make_cleg_set(EDGE_TYPE, std::move(values));
            },
        };
        m["selectTriangle"] = BuiltinFunction{
            select_set_check_call(SIMP_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                Selector sel = resolve_selector_arg("selectTriangle", args[0], simp_type(2), parse_triangle_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto tris = select_triangle(bc.adj, bc.embed, sel, globals.random);
                std::vector<ClegValue> values;
                for (auto& t : tris) values.push_back(make_simp_v(t));
                return make_cleg_set(SIMP_TYPE, std::move(values));
            },
        };
        // Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS['selectSimp'].
        m["selectSimp"] = BuiltinFunction{
            select_set_check_call(SIMP_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                Selector sel = resolve_any_kind_selector_arg("selectSimp", args[0]);
                if (simp_n(sel.type) < 0)
                    throw std::runtime_error(
                        "cleg: 'selectSimp' expects a simp selector, got a '" + selector_type_word(sel.type) + "' selector");
                const BoardConfig& bc = *args[1].egr_v;
                auto simps = select_simp(bc.adj, bc.embed, sel, globals.random);
                std::vector<ClegValue> values;
                for (auto& t : simps) values.push_back(make_simp_v(t));
                return make_cleg_set(SIMP_TYPE, std::move(values));
            },
        };
        m["selectQuad"] = BuiltinFunction{
            select_set_check_call(QUAD_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                Selector sel = resolve_selector_arg("selectQuad", args[0], SelectorType{SelectorKind::Quad}, parse_quad_selector);
                const BoardConfig& bc = *args[1].egr_v;
                auto quads = select_quad(bc.adj, bc.embed, sel, globals.random);
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
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::TriangleForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("triangleForm", args[1], simp_type(2), parse_triangle_selector);
                return make_mod(bm);
            },
        };
        m["quadForm"] = BuiltinFunction{
            form_mod_check_call,
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::QuadForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("quadForm", args[1], SelectorType{SelectorKind::Quad}, parse_quad_selector);
                return make_mod(bm);
            },
        };
        m["quadDiagForm"] = BuiltinFunction{
            form_mod_check_call,
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::QuadDiagForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("quadDiagForm", args[1], SelectorType{SelectorKind::Quad}, parse_quad_selector);
                return make_mod(bm);
            },
        };
        m["quadKnightForm"] = BuiltinFunction{
            form_mod_check_call,
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::QuadKnightForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("quadKnightForm", args[1], SelectorType{SelectorKind::Quad}, parse_quad_selector);
                return make_mod(bm);
            },
        };
        m["quadBishopForm"] = BuiltinFunction{
            form_mod_check_call,
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::QuadBishopForm};
                bm.split_n = static_cast<int>(args[0].number);
                if (args.size() == 2) bm.form_sel = resolve_selector_arg("quadBishopForm", args[1], SelectorType{SelectorKind::Quad}, parse_quad_selector);
                return make_mod(bm);
            },
        };

        // Mirrors shared/clegEval.ts's FORMSEL_TYPE_STRINGS/isFormSelTypeString()/FORMSEL_KIND_OF.
        static const std::vector<std::string> form_sel_type_strings =
            {"triForm", "quadForm", "quadDiagForm", "quadKnightForm", "quadBishopForm"};
        auto is_form_sel_type_string = [](const std::string& s) {
            return std::find(form_sel_type_strings.begin(), form_sel_type_strings.end(), s) != form_sel_type_strings.end();
        };
        auto form_sel_kind_of = [](const std::string& s) {
            if (s == "triForm") return FormSelKind::TriForm;
            if (s == "quadForm") return FormSelKind::QuadForm;
            if (s == "quadDiagForm") return FormSelKind::QuadDiagForm;
            if (s == "quadKnightForm") return FormSelKind::QuadKnightForm;
            return FormSelKind::QuadBishopForm;
        };

        // Mirrors shared/clegEval.ts's mkFormSelCheckCall()/BUILTIN_FUNCTIONS['mkFormSel'].
        CheckCallFn mk_form_sel_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 1 && arg_types.size() != 2)
                throw std::runtime_error("cleg: '" + callee + "' expects 1 or 2 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::String)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected string, got " + type_to_string(arg_types[0]));
            if (arg_types.size() == 2 && arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
            return FORMSEL_TYPE;
        };
        m["mkFormSel"] = BuiltinFunction{
            mk_form_sel_check_call,
            [is_form_sel_type_string, form_sel_kind_of](const std::vector<ClegValue>& args, GlobalInfo&) {
                const std::string& type_str = args[0].str;
                if (!is_form_sel_type_string(type_str))
                    throw std::runtime_error("cleg: 'mkFormSel' argument 1: unknown form selector type '" + type_str + "'");
                std::optional<Selector> sel;
                if (args.size() == 2) sel = resolve_any_kind_selector_arg("mkFormSel", args[1]);
                SelectorType want_type = type_str == "triForm" ? simp_type(2) : SelectorType{SelectorKind::Quad};
                if (sel && sel->type != want_type)
                    throw std::runtime_error(
                        "cleg: 'mkFormSel' expects a " + std::string(type_str == "triForm" ? "triangle (simp 2)" : "quad") +
                        " selector for '" + type_str + "', got a '" + selector_type_word(sel->type) + "' selector");
                return make_form_sel_v(FormSelector{form_sel_kind_of(type_str), sel});
            },
        };

        // Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS['form'].
        m["form"] = BuiltinFunction{
            fixed_signature({NUMBER_TYPE, ClegType{CTKind::Array, std::make_shared<ClegType>(FORMSEL_TYPE)}}, MOD_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::Form};
                bm.split_n = static_cast<int>(args[0].number);
                for (auto& v : args[1].arr_v) bm.form_sels.push_back(v.form_sel_v);
                return make_mod(bm);
            },
        };

        // Mirrors shared/clegEval.ts's LRS_TYPE_STRINGS/isLRSTypeString()/QUAD_LRS_KIND_OF/
        // SIMP_LRS_KIND_OF.
        static const std::vector<std::string> lrs_type_strings =
            {"quadCentralize", "quadCentering", "quadOctarize", "simpCentralize", "simpCentering"};
        auto is_lrs_type_string = [](const std::string& s) {
            return std::find(lrs_type_strings.begin(), lrs_type_strings.end(), s) != lrs_type_strings.end();
        };
        auto quad_lrs_kind_of = [](const std::string& s) {
            if (s == "quadCentralize") return LocalReplaceKind::QuadCentralize;
            if (s == "quadCentering") return LocalReplaceKind::QuadCentering;
            return LocalReplaceKind::QuadOctarize;
        };
        auto simp_lrs_kind_of = [](const std::string& s) {
            return s == "simpCentralize" ? LocalReplaceKind::SimpCentralize : LocalReplaceKind::SimpCentering;
        };

        // Mirrors shared/clegEval.ts's mkLRSCheckCall()/BUILTIN_FUNCTIONS['mkLRS'].
        CheckCallFn mk_lrs_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 1 && arg_types.size() != 2)
                throw std::runtime_error("cleg: '" + callee + "' expects 1 or 2 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::String)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected string, got " + type_to_string(arg_types[0]));
            if (arg_types.size() == 2 && arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
            return LRS_TYPE;
        };
        m["mkLRS"] = BuiltinFunction{
            mk_lrs_check_call,
            [is_lrs_type_string, quad_lrs_kind_of, simp_lrs_kind_of](const std::vector<ClegValue>& args, GlobalInfo&) {
                const std::string& type_str = args[0].str;
                if (!is_lrs_type_string(type_str))
                    throw std::runtime_error("cleg: 'mkLRS' argument 1: unknown lrs type '" + type_str + "'");
                std::optional<Selector> sel;
                if (args.size() == 2) sel = resolve_any_kind_selector_arg("mkLRS", args[1]);
                if (type_str == "simpCentralize" || type_str == "simpCentering") {
                    if (!sel)
                        throw std::runtime_error(
                            "cleg: 'mkLRS' requires argument 2 for '" + type_str + "', to infer its own simplex arity from");
                    int n = simp_n(sel->type);
                    if (n < 0)
                        throw std::runtime_error(
                            "cleg: 'mkLRS' expects a simp selector for '" + type_str + "', got a '" +
                            selector_type_word(sel->type) + "' selector");
                    return make_lrs_v(LocalReplaceSelector{simp_lrs_kind_of(type_str), n, sel});
                }
                if (sel && sel->type != SelectorType{SelectorKind::Quad})
                    throw std::runtime_error(
                        "cleg: 'mkLRS' expects a quad selector for '" + type_str + "', got a '" +
                        selector_type_word(sel->type) + "' selector");
                return make_lrs_v(LocalReplaceSelector{quad_lrs_kind_of(type_str), 0, sel});
            },
        };

        // Mirrors shared/clegEval.ts's BUILTIN_FUNCTIONS['localReplace'].
        m["localReplace"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(LRS_TYPE)}}, MOD_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                BoardModifier bm{ModifierKind::LocalReplace};
                for (auto& v : args[0].arr_v) bm.selectors.push_back(v.lrs_v);
                return make_mod(bm);
            },
        };

        // Mirrors shared/clegEval.ts's localReplaceModCheckCall()/oneShapeMod() and the
        // quadCentralize/.../triCentering builtins.
        CheckCallFn local_replace_mod_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 0 && arg_types.size() != 1)
                throw std::runtime_error("cleg: '" + callee + "' expects 0 or 1 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types.size() == 1 && arg_types[0].kind != CTKind::Sel && arg_types[0].kind != CTKind::String && arg_types[0].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected sel, string, or set, got " + type_to_string(arg_types[0]));
            return MOD_TYPE;
        };
        auto one_shape_mod = [](LocalReplaceKind kind, std::optional<Selector> sel) -> ClegValue {
            BoardModifier bm{ModifierKind::LocalReplace};
            bm.selectors.push_back(LocalReplaceSelector{kind, 0, sel});
            return make_mod(bm);
        };
        m["quadCentralize"] = BuiltinFunction{
            local_replace_mod_check_call,
            [one_shape_mod](const std::vector<ClegValue>& args, GlobalInfo&) {
                std::optional<Selector> sel;
                if (args.size() == 1) sel = resolve_selector_arg("quadCentralize", args[0], SelectorType{SelectorKind::Quad}, parse_quad_selector);
                return one_shape_mod(LocalReplaceKind::QuadCentralize, sel);
            },
        };
        m["quadCentering"] = BuiltinFunction{
            local_replace_mod_check_call,
            [one_shape_mod](const std::vector<ClegValue>& args, GlobalInfo&) {
                std::optional<Selector> sel;
                if (args.size() == 1) sel = resolve_selector_arg("quadCentering", args[0], SelectorType{SelectorKind::Quad}, parse_quad_selector);
                return one_shape_mod(LocalReplaceKind::QuadCentering, sel);
            },
        };
        m["quadOctarize"] = BuiltinFunction{
            local_replace_mod_check_call,
            [one_shape_mod](const std::vector<ClegValue>& args, GlobalInfo&) {
                std::optional<Selector> sel;
                if (args.size() == 1) sel = resolve_selector_arg("quadOctarize", args[0], SelectorType{SelectorKind::Quad}, parse_quad_selector);
                return one_shape_mod(LocalReplaceKind::QuadOctarize, sel);
            },
        };

        CheckCallFn simp_centralize_mod_check_call = [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
            if (arg_types.size() != 1 && arg_types.size() != 2)
                throw std::runtime_error("cleg: '" + callee + "' expects 1 or 2 argument(s), got " + std::to_string(arg_types.size()));
            if (arg_types[0].kind != CTKind::Number)
                throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
            if (arg_types.size() == 2 && arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
            return MOD_TYPE;
        };
        auto one_shape_simp_mod = [](LocalReplaceKind kind, int n, std::optional<Selector> sel) -> ClegValue {
            BoardModifier bm{ModifierKind::LocalReplace};
            bm.selectors.push_back(LocalReplaceSelector{kind, n, sel});
            return make_mod(bm);
        };
        m["simpCentralize"] = BuiltinFunction{
            simp_centralize_mod_check_call,
            [one_shape_simp_mod](const std::vector<ClegValue>& args, GlobalInfo&) {
                int n = static_cast<int>(args[0].number);
                std::optional<Selector> sel;
                if (args.size() == 2)
                    sel = resolve_selector_arg(
                        "simpCentralize", args[1], simp_type(n), [n](const std::string& s) { return parse_simp_selector(n, s); });
                return one_shape_simp_mod(LocalReplaceKind::SimpCentralize, n, sel);
            },
        };
        m["simpCentering"] = BuiltinFunction{
            simp_centralize_mod_check_call,
            [one_shape_simp_mod](const std::vector<ClegValue>& args, GlobalInfo&) {
                int n = static_cast<int>(args[0].number);
                std::optional<Selector> sel;
                if (args.size() == 2)
                    sel = resolve_selector_arg(
                        "simpCentering", args[1], simp_type(n), [n](const std::string& s) { return parse_simp_selector(n, s); });
                return one_shape_simp_mod(LocalReplaceKind::SimpCentering, n, sel);
            },
        };
        m["triCentralize"] = BuiltinFunction{
            local_replace_mod_check_call,
            [one_shape_simp_mod](const std::vector<ClegValue>& args, GlobalInfo&) {
                std::optional<Selector> sel;
                if (args.size() == 1) sel = resolve_selector_arg("triCentralize", args[0], simp_type(2), parse_triangle_selector);
                return one_shape_simp_mod(LocalReplaceKind::SimpCentralize, 2, sel);
            },
        };
        m["triCentering"] = BuiltinFunction{
            local_replace_mod_check_call,
            [one_shape_simp_mod](const std::vector<ClegValue>& args, GlobalInfo&) {
                std::optional<Selector> sel;
                if (args.size() == 1) sel = resolve_selector_arg("triCentering", args[0], simp_type(2), parse_triangle_selector);
                return one_shape_simp_mod(LocalReplaceKind::SimpCentering, 2, sel);
            },
        };

        m["modify"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(MOD_TYPE)}, EGR_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                std::vector<BoardModifier> mods;
                for (auto& v : args[0].arr_v) mods.push_back(v.mod_v);
                return make_egr(apply_modifiers(*args[1].egr_v, mods, globals.random));
            },
        };

        m["psAll"] = BuiltinFunction{
            fixed_signature({}, PSEL_TYPE), [](const std::vector<ClegValue>&, GlobalInfo&) { return make_psel(ProdSelector{PSelOp::All}); },
        };
        m["psBase"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 2)
                    throw std::runtime_error("cleg: '" + callee + "' expects 2 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Number)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
                if (arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
                return PSEL_TYPE;
            },
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                double number = args[0].number;
                if (number != std::floor(number) || number < 0)
                    throw std::runtime_error("cleg: psBase: number must be a nonnegative integer, got " + format_number_display(number));
                ProdSelector ps; ps.op = PSelOp::Base; ps.number = static_cast<int>(number);
                ps.sel = std::make_shared<Selector>(resolve_any_kind_selector_arg("psBase", args[1]));
                return make_psel(ps);
            },
        };
        // Mirrors shared/clegEval.ts's psBaseNECheckCall()/BUILTIN_FUNCTIONS['psBaseNE'].
        m["psBaseNE"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 3)
                    throw std::runtime_error("cleg: '" + callee + "' expects 3 argument(s), got " + std::to_string(arg_types.size()));
                if (arg_types[0].kind != CTKind::Number)
                    throw std::runtime_error("cleg: '" + callee + "' argument 1: expected number, got " + type_to_string(arg_types[0]));
                if (arg_types[1].kind != CTKind::Sel && arg_types[1].kind != CTKind::String && arg_types[1].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 2: expected sel, string, or set, got " + type_to_string(arg_types[1]));
                if (arg_types[2].kind != CTKind::Sel && arg_types[2].kind != CTKind::String && arg_types[2].kind != CTKind::Set)
                    throw std::runtime_error("cleg: '" + callee + "' argument 3: expected sel, string, or set, got " + type_to_string(arg_types[2]));
                return PSEL_TYPE;
            },
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                double number = args[0].number;
                if (number != std::floor(number) || number < 0)
                    throw std::runtime_error("cleg: psBaseNE: number must be a nonnegative integer, got " + format_number_display(number));
                ProdSelector ps; ps.op = PSelOp::BaseNE; ps.number = static_cast<int>(number);
                ps.node_sel = std::make_shared<Selector>(
                    resolve_selector_arg("psBaseNE", args[1], SelectorType{SelectorKind::Node}, parse_node_selector));
                ps.edge_sel = std::make_shared<Selector>(
                    resolve_selector_arg("psBaseNE", args[2], SelectorType{SelectorKind::Edge}, parse_edge_selector));
                return make_psel(ps);
            },
        };
        m["psUnion"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(PSEL_TYPE)}}, PSEL_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                ProdSelector ps; ps.op = PSelOp::Union;
                for (auto& v : args[0].arr_v) ps.items.push_back(v.psel_v);
                return make_psel(ps);
            },
        };
        m["psInter"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(PSEL_TYPE)}}, PSEL_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                ProdSelector ps; ps.op = PSelOp::Inter;
                for (auto& v : args[0].arr_v) ps.items.push_back(v.psel_v);
                return make_psel(ps);
            },
        };
        m["psDiff"] = BuiltinFunction{
            fixed_signature({PSEL_TYPE, PSEL_TYPE}, PSEL_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo&) {
                ProdSelector ps; ps.op = PSelOp::Diff;
                ps.a = std::make_shared<ProdSelector>(args[0].psel_v);
                ps.b = std::make_shared<ProdSelector>(args[1].psel_v);
                return make_psel(ps);
            },
        };
        m["multiProd"] = BuiltinFunction{
            fixed_signature({ClegType{CTKind::Array, std::make_shared<ClegType>(EGR_TYPE)}, PSEL_TYPE}, EGR_TYPE),
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                std::vector<BoardConfig> boards;
                for (auto& v : args[0].arr_v) boards.push_back(*v.egr_v);
                if (boards.empty()) throw std::runtime_error("cleg: multiProd: boards must be non-empty");
                FullProductIndex fpi = make_full_product_index(boards);
                auto result = eval_prod_selector(boards, fpi, args[1].psel_v, globals.random);
                return make_egr(std::move(result.bc));
            },
        };

        // Mirrors shared/clegEval.ts's subHcublatCheckCall()/BUILTIN_FUNCTIONS['subHcublatB'], with
        // econdCallback()/adjFnCallback() inlined as the lambdas below.
        m["subHcublatB"] = BuiltinFunction{
            [](const std::string& callee, const std::vector<ClegType>& arg_types) -> ClegType {
                if (arg_types.size() != 2 && arg_types.size() != 3)
                    throw std::runtime_error("cleg: '" + callee + "' expects 2 or 3 argument(s), got " + std::to_string(arg_types.size()));
                if (!type_equals(arg_types[0], HCUBLAT_BOUNDS_TYPE))
                    throw std::runtime_error(
                        "cleg: '" + callee + "' argument 1: expected " + type_to_string(HCUBLAT_BOUNDS_TYPE) + ", got " +
                        type_to_string(arg_types[0]));
                if (!type_equals(arg_types[1], HCUBLAT_COND_TYPE))
                    throw std::runtime_error(
                        "cleg: '" + callee + "' argument 2: expected " + type_to_string(HCUBLAT_COND_TYPE) + ", got " +
                        type_to_string(arg_types[1]));
                if (arg_types.size() == 3 && !type_equals(arg_types[2], HCUBLAT_ECOND_TYPE) && !type_equals(arg_types[2], HCUBLAT_ADJFN_TYPE))
                    throw std::runtime_error(
                        "cleg: '" + callee + "' argument 3: expected " + type_to_string(HCUBLAT_ECOND_TYPE) + " or " +
                        type_to_string(HCUBLAT_ADJFN_TYPE) + ", got " + type_to_string(arg_types[2]));
                return EGR_TYPE;
            },
            [](const std::vector<ClegValue>& args, GlobalInfo& globals) {
                if (args.size() == 2) {
                    auto econd = [](const std::vector<int>&, const std::vector<int>&) { return true; };
                    return build_sub_hcublat(args[0], args[1], econd, globals);
                }
                if (args[2].func_params.size() == 2) {
                    std::function<bool(const std::vector<int>&, const std::vector<int>&)> econd =
                        [&args, &globals](const std::vector<int>& a, const std::vector<int>& b) {
                            const ClegValue& econd_val = args[2];
                            ClegValue a_arg; a_arg.kind = CTKind::Array; a_arg.elem = NUMBER_TYPE;
                            for (int c : a) a_arg.arr_v.push_back(make_number(c));
                            ClegValue b_arg; b_arg.kind = CTKind::Array; b_arg.elem = NUMBER_TYPE;
                            for (int c : b) b_arg.arr_v.push_back(make_number(c));
                            return call_user_function(
                                *globals.funcs.at(econd_val.func_name), fill_holes(econd_val.func_bound_args, {a_arg, b_arg}), globals).boolean;
                        };
                    return build_sub_hcublat(args[0], args[1], econd, globals);
                }
                std::function<std::vector<std::vector<int>>(const std::vector<int>&)> adj_fn =
                    [&args, &globals](const std::vector<int>& point) {
                        const ClegValue& adj_fn_val = args[2];
                        ClegValue point_arg; point_arg.kind = CTKind::Array; point_arg.elem = NUMBER_TYPE;
                        for (int c : point) point_arg.arr_v.push_back(make_number(c));
                        ClegValue result = call_user_function(
                            *globals.funcs.at(adj_fn_val.func_name), fill_holes(adj_fn_val.func_bound_args, {point_arg}), globals);
                        std::vector<std::vector<int>> targets;
                        for (auto& inner : result.arr_v) {
                            std::vector<int> t;
                            for (auto& c : inner.arr_v) t.push_back(static_cast<int>(std::llround(c.number)));
                            targets.push_back(std::move(t));
                        }
                        return targets;
                    };
                return build_sub_hcublat_custom_adj(args[0], args[1], adj_fn, globals);
            },
        };

        return m;
    }();
    return table;
}

// ── Evaluation ───────────────────────────────────────────────────────────────

// Mirrors shared/clegEval.ts's ValueEnv (raw-pointer parent chain - see this file's top comment).
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
// Mirrors shared/clegEval.ts's lookupValue().
static ClegValue lookup_value(ValueEnv& env, const std::string& name) {
    ClegValue* v = lookup_value_ptr(env, name);
    if (!v) throw std::runtime_error("cleg: undeclared variable '" + name + "'");
    return *v;
}
// Mirrors shared/clegEval.ts's setValue().
static void set_value(ValueEnv& env, const std::string& name, ClegValue value) {
    ClegValue* v = lookup_value_ptr(env, name);
    if (!v) throw std::runtime_error("cleg: undeclared variable '" + name + "'");
    *v = std::move(value);
}

// Mirrors shared/clegEval.ts's cloneArrayValue().
static ClegValue clone_array_value(const ClegValue& v) {
    if (v.kind != CTKind::Array) return v;
    ClegValue out = v;
    out.arr_v.clear();
    out.arr_v.reserve(v.arr_v.size());
    for (auto& e : v.arr_v) out.arr_v.push_back(clone_array_value(e));
    return out;
}

// Mirrors shared/clegEval.ts's validateArrayIndex(), returning the index as a size_t.
static size_t validate_array_index(double idx, size_t length) {
    if (idx != std::floor(idx) || idx < 0 || idx >= static_cast<double>(length))
        throw std::runtime_error(
            "cleg: array index " + format_number_display(idx) + " out of bounds for array of length " +
            std::to_string(length));
    return static_cast<size_t>(idx);
}

// Mirrors shared/clegEval.ts's fillHoles(), with std::nullopt for TS's null slots.
static std::vector<ClegValue> fill_holes(
    const std::vector<std::optional<ClegValue>>& bound_args, const std::vector<ClegValue>& supplied_args)
{
    std::vector<ClegValue> out;
    out.reserve(bound_args.size());
    size_t i = 0;
    for (auto& b : bound_args) out.push_back(b ? *b : supplied_args[i++]);
    return out;
}

// Mirrors shared/clegEval.ts's ReturnSignal.
struct ReturnSignal { ClegValue value; };
// Mirrors shared/clegEval.ts's BreakSignal/ContinueSignal.
struct BreakSignal {};
struct ContinueSignal {};

static void eval_block(const Stmt& block, ValueEnv* parent, GlobalInfo& globals);
static void eval_stmt(const Stmt& stmt, ValueEnv& env, GlobalInfo& globals);
static ClegValue eval_expr(const Expr& expr, ValueEnv& env, GlobalInfo& globals);
static ClegValue call_user_function(const FunctionDecl& fn, const std::vector<ClegValue>& args, GlobalInfo& globals);

// Mirrors shared/clegEval.ts's mergeBoundArgs(), with std::nullopt for TS's null slots.
static std::vector<std::optional<ClegValue>> merge_bound_args(
    const std::vector<std::optional<ClegValue>>& bound_args, const std::vector<Expr>& args,
    ValueEnv& env, GlobalInfo& globals)
{
    std::vector<std::optional<ClegValue>> out;
    out.reserve(bound_args.size());
    size_t j = 0;
    for (auto& b : bound_args) {
        if (b) { out.push_back(b); continue; }
        const Expr& a = args[j++];
        if (a.kind == ExprKind::HoleExpr) out.push_back(std::nullopt);
        else out.push_back(clone_array_value(eval_expr(a, env, globals)));
    }
    return out;
}

static void eval_block(const Stmt& block, ValueEnv* parent, GlobalInfo& globals) {
    ValueEnv env; env.parent = parent;
    for (auto& stmt : block.stmts) eval_stmt(stmt, env, globals);
}

static void eval_stmt(const Stmt& stmt, ValueEnv& env, GlobalInfo& globals) {
    switch (stmt.kind) {
        case StmtKind::VarDecl:
            env.vars[stmt.name] = clone_array_value(eval_expr(*stmt.expr, env, globals));
            return;
        case StmtKind::AssignStmt: {
            ClegValue value = clone_array_value(eval_expr(*stmt.expr, env, globals));
            if (stmt.indices.empty()) {
                set_value(env, stmt.name, std::move(value));
                return;
            }
            // `target` points into the value stored in `env` (lookup_value_ptr never copies), so the
            // final slot below is mutated in place.
            ClegValue* target = lookup_value_ptr(env, stmt.name);
            // Unreachable in a program that has passed typecheck_cleg.
            if (!target) throw std::runtime_error("cleg: undeclared variable '" + stmt.name + "'");
            for (size_t i = 0; i + 1 < stmt.indices.size(); i++) {
                double idx_value = eval_expr(stmt.indices[i], env, globals).number;
                size_t idx = validate_array_index(idx_value, target->arr_v.size());
                target = &target->arr_v[idx];
            }
            double last_idx_value = eval_expr(stmt.indices.back(), env, globals).number;
            size_t last_idx = validate_array_index(last_idx_value, target->arr_v.size());
            target->arr_v[last_idx] = std::move(value);
            return;
        }
        case StmtKind::IfStmt: {
            ClegValue cond = eval_expr(*stmt.cond, env, globals);
            if (cond.boolean) eval_block(*stmt.then_stmt, &env, globals);
            else if (stmt.else_stmt) {
                if (stmt.else_stmt->kind == StmtKind::Block) eval_block(*stmt.else_stmt, &env, globals);
                else eval_stmt(*stmt.else_stmt, env, globals);
            }
            return;
        }
        case StmtKind::ForStmt: {
            ValueEnv loop_env; loop_env.parent = &env;
            if (stmt.for_init) eval_stmt(*stmt.for_init, loop_env, globals);
            while (!stmt.cond || eval_expr(*stmt.cond, loop_env, globals).boolean) {
                try {
                    eval_block(*stmt.body, &loop_env, globals);
                } catch (BreakSignal&) {
                    break;
                } catch (ContinueSignal&) {
                    // fall through to for_update below
                }
                if (stmt.for_update) eval_stmt(*stmt.for_update, loop_env, globals);
            }
            return;
        }
        case StmtKind::WhileStmt: {
            while (eval_expr(*stmt.cond, env, globals).boolean) {
                try {
                    eval_block(*stmt.body, &env, globals);
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
            throw ReturnSignal{eval_expr(*stmt.expr, env, globals)};
        case StmtKind::ExprStmt:
            eval_expr(*stmt.expr, env, globals);
            return;
        case StmtKind::Block:
            eval_block(stmt, &env, globals);
            return;
    }
}

static ClegValue eval_expr(const Expr& expr, ValueEnv& env, GlobalInfo& globals) {
    switch (expr.kind) {
        case ExprKind::NumberLit: return make_number(expr.number_value);
        case ExprKind::StringLit: return make_string(expr.string_value);
        case ExprKind::BoolLit: return make_bool(expr.bool_value);
        case ExprKind::Identifier: {
            ClegValue* v = lookup_value_ptr(env, expr.string_value);
            if (v) return *v;
            // Not a variable - a bare reference to one of program's own top-level functions, used as
            // a function-pointer value (check_expr already confirmed this resolves and is func-typed).
            const FunctionDecl* fn = globals.funcs.at(expr.string_value);
            ClegValue fv; fv.kind = CTKind::Func; fv.func_name = expr.string_value;
            for (auto& p : fn->params) { fv.func_params.push_back(p.type); fv.func_bound_args.push_back(std::nullopt); }
            fv.func_return_type = fn->return_type;
            return fv;
        }
        case ExprKind::ArrayLit: {
            std::vector<ClegValue> values;
            for (auto& e : expr.elements) values.push_back(eval_expr(e, env, globals));
            // typecheck_cleg already rejected an empty or mixed-element-type literal, so the first
            // value's own type is always the array's element type.
            ClegValue v; v.kind = CTKind::Array; v.elem = cleg_value_type(values[0]); v.arr_v = std::move(values);
            return v;
        }
        case ExprKind::SetLit: {
            std::vector<ClegValue> values;
            for (auto& e : expr.elements) values.push_back(eval_expr(e, env, globals));
            // Computed in its own statement, before std::move(values): sibling-argument evaluation
            // order is unspecified (same hazard as game/cleg_parser.cpp's parse_expr chain).
            ClegType elem = cleg_value_type(values[0]);
            return make_cleg_set(std::move(elem), std::move(values));
        }
        case ExprKind::CallExpr: {
            if (std::any_of(expr.elements.begin(), expr.elements.end(),
                             [](const Expr& a) { return a.kind == ExprKind::HoleExpr; })) {
                // Partial application - check_expr already ruled out a builtin callee.
                ClegValue* var_value = lookup_value_ptr(env, expr.string_value);
                const std::string& name = var_value ? var_value->func_name : expr.string_value;
                const FunctionDecl* fn = globals.funcs.at(name);
                std::vector<std::optional<ClegValue>> starting_bound_args = var_value
                    ? var_value->func_bound_args
                    : std::vector<std::optional<ClegValue>>(fn->params.size());
                auto bound_args = merge_bound_args(starting_bound_args, expr.elements, env, globals);
                ClegValue fv; fv.kind = CTKind::Func; fv.func_name = name; fv.func_bound_args = bound_args;
                for (size_t i = 0; i < fn->params.size(); i++) if (!bound_args[i]) fv.func_params.push_back(fn->params[i].type);
                fv.func_return_type = fn->return_type;
                return fv;
            }
            std::vector<ClegValue> args;
            for (auto& a : expr.elements) args.push_back(eval_expr(a, env, globals));
            auto& builtins = builtin_functions();
            auto bit = builtins.find(expr.string_value);
            if (bit != builtins.end()) return bit->second.call(args, globals);
            // A local func-typed variable shadows a same-named top-level function (see check_expr).
            ClegValue* var_value = lookup_value_ptr(env, expr.string_value);
            if (var_value) return call_user_function(*globals.funcs.at(var_value->func_name), fill_holes(var_value->func_bound_args, args), globals);
            return call_user_function(*globals.funcs.at(expr.string_value), args, globals);
        }
        case ExprKind::BinaryExpr: {
            // `&&`/`||` short-circuit here, before ever reaching binary_operator_overloads() below -
            // see logical_overload's own doc comment (game/cleg_base.cpp).
            if (expr.op == "&&" || expr.op == "||") {
                bool l = eval_expr(*expr.left, env, globals).boolean;
                if (expr.op == "&&") return l ? eval_expr(*expr.right, env, globals) : make_bool(false);
                return l ? make_bool(true) : eval_expr(*expr.right, env, globals);
            }
            ClegValue l = eval_expr(*expr.left, env, globals);
            ClegValue r = eval_expr(*expr.right, env, globals);
            ClegType lt = cleg_value_type(l), rt = cleg_value_type(r);
            for (auto& overload : binary_operator_overloads().at(expr.op)) {
                auto m = overload.match(lt, rt);
                if (m) return m->eval(l, r);
            }
            // Unreachable in a program that has passed typecheck_cleg.
            throw std::runtime_error("cleg: operator '" + expr.op + "' has no overload for these operand types at runtime");
        }
        case ExprKind::UnaryExpr:
            if (expr.op == "-") return make_number(-eval_expr(*expr.operand, env, globals).number);
            return make_bool(!eval_expr(*expr.operand, env, globals).boolean);
        case ExprKind::NilExpr: {
            ClegValue v; v.kind = CTKind::Array; v.elem = expr.nil_type;
            return v;
        }
        case ExprKind::IndexExpr: {
            ClegValue arr = eval_expr(*expr.left, env, globals);
            double idx_value = eval_expr(*expr.right, env, globals).number;
            size_t idx = validate_array_index(idx_value, arr.arr_v.size());
            return arr.arr_v[idx];
        }
    }
    throw std::runtime_error("cleg: eval_expr: unexpected ExprKind");
}

static ClegValue call_user_function(const FunctionDecl& fn, const std::vector<ClegValue>& args, GlobalInfo& globals) {
    ValueEnv env;
    for (size_t i = 0; i < fn.params.size(); i++) env.vars[fn.params[i].name] = clone_array_value(args[i]);
    try {
        eval_block(fn.body, &env, globals);
    } catch (ReturnSignal& r) {
        return std::move(r.value);
    }
    throw std::runtime_error("cleg: function '" + fn.name + "' fell off its own end without a 'return'");
}

// Mirrors shared/clegEval.ts's runClegProgram().
static ClegValue run_cleg_program(const ClegProgram& program, uint32_t seed) {
    typecheck_cleg(program);
    UserFuncTable funcs;
    for (auto& fn : program.functions) funcs[fn.name] = &fn;
    GlobalInfo globals{std::move(funcs), ClegRandom(seed)};

    ValueEnv env;
    std::optional<ClegValue> result;
    for (auto& stmt : program.stmts) {
        if (stmt.kind == StmtKind::ExprStmt) result = eval_expr(*stmt.expr, env, globals);
        else eval_stmt(stmt, env, globals);
    }
    return std::move(*result); // typecheck_cleg already required the last top-level statement to be an ExprStmt
}

// ── Public API ───────────────────────────────────────────────────────────────

BoardConfig build_board_from_cleg(const ClegProgram& program, uint32_t seed) {
    typecheck_cleg_as_board(program);
    ClegValue result = run_cleg_program(program, seed);
    return std::move(*result.egr_v);
}
