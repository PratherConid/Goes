/**
 * cleg's builtin-function table (BUILTIN_FUNCTIONS) and its tree-walking evaluator, plus the
 * runCleg/buildBoardFromCleg-style entry points every consumer outside cleg's own files actually
 * uses - see shared/clegBase.ts's own top comment for the cleg language itself (grammar, semantics,
 * the four-file split this is one part of).
 *
 * A BuiltinFunction (see its own doc comment below) bundles a `checkCall` (static type-checking)
 * alongside its `call` (runtime evaluation) in one object, by original design (see BuiltinFunction's
 * own doc comment) - so this file, alongside every builtin's actual runtime behavior, also owns
 * `checkCall`, and shared/clegCheck.ts imports BUILTIN_FUNCTIONS from here for that half. This is a
 * real (non-type-only) circular import - shared/clegCheck.ts's own typecheckCleg is imported back
 * into this file (typecheckClegAsBoard/runClegProgram re-typecheck before running) - safe here
 * because every use on both sides happens inside a function body, never at module-top-level
 * evaluation time, so by the time either is actually called both modules have already finished
 * initializing (see shared/types.ts's own top comment for this codebase's general policy on
 * avoiding circular imports, and why a deferred-use cycle like this one doesn't run into the hazard
 * that policy is about).
 */

import {
    BoardArgType, numArg, csvArg, parseBoardArgToken,
    makeBoardEdge, makeBoardTriangle, makeBoardQuad, Embedding,
    type BoardArgEntry, type BoardConfig, type BoardEdge, type BoardTriangle, type BoardQuad,
    type Selector, type SelectorType, type SelectedVals, type BoardModifier,
} from './types.js';
import {
    PrescribedBoard, PrescribedBoardMap, PrescribedBoardFns, product, applyModifiers,
    make, nodeInducedSubgraph, edgeInducedSubgraph,
} from './boardConfig.js';
import {
    randomlyRemove, parseSelector, parseNodeSelector, parseEdgeSelector, parseTriangleSelector,
    parseQuadSelector, selectNode, selectEdge, selectTriangle, selectQuad,
} from './selector.js';
import { zeroAdj } from './topology.js';
import {
    type ClegType, type ClegValue, SET_ELEM_KINDS, typeEquals, typeToString, clegValueType,
    clegSetKey, makeClegSet, BINARY_OPERATOR_OVERLOADS, type MultiSelector,
    type FunctionDecl, type Block, type Stmt, type Expr, type ClegProgram,
} from './clegBase.js';
import { parseCleg } from './clegParser.js';
import { typecheckCleg } from './clegCheck.js';

// ── Predefined (board-construction) functions ─────────────────────────────────

function argTypeToClegType(t: BoardArgType): ClegType {
    switch (t) {
        case BoardArgType.Number: return { kind: 'number' };
        case BoardArgType.CommaSeparatedNumbers: return { kind: 'array', elem: { kind: 'number' } };
        case BoardArgType.ZeroOneList: return { kind: 'string' };
    }
}

function valueToBoardArgEntry(argType: BoardArgType, val: ClegValue): BoardArgEntry {
    switch (argType) {
        case BoardArgType.Number:
            if (val.kind !== 'number') throw new Error(`cleg: expected a number argument, got ${typeToString(clegValueType(val))}`);
            return numArg(val.value);
        case BoardArgType.CommaSeparatedNumbers:
            if (val.kind !== 'array' || val.elem.kind !== 'number')
                throw new Error(`cleg: expected a number[] argument, got ${typeToString(clegValueType(val))}`);
            return csvArg(val.value.map(v => (v as { kind: 'number'; value: number }).value));
        case BoardArgType.ZeroOneList:
            if (val.kind !== 'string') throw new Error(`cleg: expected a string argument, got ${typeToString(clegValueType(val))}`);
            // Reuses shared/types.ts's own ZeroOneList validation (rejects anything but a string of
            // 0/1 characters) rather than re-implementing it here.
            return parseBoardArgToken(BoardArgType.ZeroOneList, val.value);
    }
}

export interface FunctionSignature { params: ClegType[]; returnType: ClegType; }

/**
 * A callable built into the language itself, as opposed to one of `program`'s own function
 * declarations (which always have a fixed FunctionSignature - see shared/clegCheck.ts's own
 * FuncTable). Covers two rather different kinds of builtin under one shape:
 *   - the fixed-signature per-prescribed-board constructors (`menger`, `rect`, `cublat`, ...) -
 *     `checkCall` here is just fixedSignature(...)'s own arg-count/arg-type check against a fixed
 *     ClegType[]/ClegType, built once per PrescribedBoardMap entry below.
 *   - the small set of generic functions (currently just `len`) whose result type can depend on the
 *     actual argument types at a call site, so a fixed ClegType[]/ClegType can't describe them -
 *     `checkCall` is hand-written per function instead.
 * Both live in one flat BUILTIN_FUNCTIONS table, checked before `program`'s own functions in
 * checkExpr/evalExpr's CallExpr cases, and reserved against user redeclaration in typecheckCleg -
 * from a cleg program's own point of view there's no distinction between the two kinds.
 */
export interface BuiltinFunction {
    /** Validates `argTypes` (throwing descriptively on a bad arg count/type) and returns the call's
     * result type. */
    checkCall: (callee: string, argTypes: ClegType[]) => ClegType;
    /** `funcs` is only ever needed by a builtin that itself calls a `func`-typed argument back
     * (e.g. subHcublatB's own `cond`, via callUserFunction) - every other builtin's own `call` simply
     * ignores it, which TypeScript allows (a function declared with fewer parameters than a call
     * signature requires is still a valid implementation of it). */
    call: (args: ClegValue[], funcs: UserFuncTable) => ClegValue;
}

/** Builds a BuiltinFunction's own checkCall from a fixed ClegType[] -> ClegType signature - shares
 * the arg-count/arg-type checking logic with shared/clegCheck.ts's own user-function CallExpr case
 * (see checkExpr there) rather than duplicating it. */
function fixedSignature(params: ClegType[], returnType: ClegType): BuiltinFunction['checkCall'] {
    return (callee, argTypes) => {
        if (argTypes.length !== params.length)
            throw new Error(`cleg: '${callee}' expects ${params.length} argument(s), got ${argTypes.length}`);
        argTypes.forEach((t, i) => {
            if (!typeEquals(t, params[i]))
                throw new Error(`cleg: '${callee}' argument ${i + 1}: expected ${typeToString(params[i])}, got ${typeToString(t)}`);
        });
        return returnType;
    };
}

export const BUILTIN_FUNCTIONS: Record<string, BuiltinFunction> = {};

// One builtin per shared/boardConfig.ts's own PrescribedBoardMap/PrescribedBoardFns entry, named
// after PrescribedBoardMap's own cleg-name field (e.g. "mengerB", "rectB" - already carrying its own
// trailing "B", so no name-mangling happens here) - built generically from that existing table
// (rather than one hand-written cleg function per board type) so this list never drifts out of sync
// with it. The "B" suffix (baked into PrescribedBoardMap itself) keeps every one of these names
// clear of TYPE_KEYWORDS by construction (rather than special-casing the one existing collision,
// "tri" vs. the `tri` triangle-value type - a future board name could collide too, e.g. "mod" or
// "egr").
for (const [pbKey, [argTypes, clegName]] of
    Object.entries(PrescribedBoardMap) as [string, [BoardArgType[], string, string, string]][]) {
    const pb = Number(pbKey) as PrescribedBoard;
    BUILTIN_FUNCTIONS[clegName] = {
        checkCall: fixedSignature(argTypes.map(argTypeToClegType), { kind: 'egr' }),
        call: (args: ClegValue[]): ClegValue =>
            ({ kind: 'egr', value: PrescribedBoardFns[pb](...argTypes.map((t, i) => valueToBoardArgEntry(t, args[i]))) }),
    };
}

// `len`: an array's or set's length, as a `number` - its one argument may be an array or set of
// any element type, which a fixedSignature(...) can't express (there's no "any" ClegType), hence
// the hand-written checkCall/call pair here instead.
BUILTIN_FUNCTIONS['len'] = {
    checkCall(callee, argTypes) {
        if (argTypes.length !== 1)
            throw new Error(`cleg: '${callee}' expects 1 argument(s), got ${argTypes.length}`);
        if (argTypes[0].kind !== 'array' && argTypes[0].kind !== 'set')
            throw new Error(
                `cleg: '${callee}' argument 1: expected an array or set, got ${typeToString(argTypes[0])}`);
        return { kind: 'number' };
    },
    call(args) {
        const v = args[0] as { kind: 'array' | 'set'; elem: ClegType; value: ClegValue[] };
        return { kind: 'number', value: v.value.length };
    },
};

// `has(x, e)`: whether `x` (a `T[]` or `T{}`) contains `e` (a `T`), as a `bool` - like `len`, its
// result depends on the actual argument types (here, argument 2's own required type, taken from
// argument 1's element type), hence the hand-written checkCall/call pair. `T` is restricted to
// SET_ELEM_KINDS (number/string/bool/edge/tri/quad) for BOTH `T[]` and `T{}` - even though an
// array's own element type is normally unrestricted, nothing outside SET_ELEM_KINDS has a defined
// equality in this language (e.g. `egr`: "no natural equality/hashing for a whole board", see
// shared/clegBase.ts's own top comment on SET_ELEM_KINDS), so `has` can't be given a well-defined
// meaning for one either. Compares by clegSetKey, the same equality every set operation already uses.
function hasCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'array' && argTypes[0].kind !== 'set')
        throw new Error(
            `cleg: '${callee}' argument 1: expected an array or set, got ${typeToString(argTypes[0])}`);
    const elem = (argTypes[0] as { elem: ClegType }).elem;
    if (!SET_ELEM_KINDS.has(elem.kind))
        throw new Error(
            `cleg: '${callee}' argument 1: element type ${typeToString(elem)} has no defined equality - only ` +
            `number/string/bool/edge/tri/quad elements are supported`);
    if (!typeEquals(argTypes[1], elem))
        throw new Error(
            `cleg: '${callee}' argument 2: expected ${typeToString(elem)} (the element type of argument 1), got ` +
            `${typeToString(argTypes[1])}`);
    return { kind: 'bool' };
}
BUILTIN_FUNCTIONS['has'] = {
    checkCall: hasCheckCall,
    call(args) {
        const container = args[0] as { kind: 'array' | 'set'; elem: ClegType; value: ClegValue[] };
        const key = clegSetKey(args[1]);
        return { kind: 'bool', value: container.value.some(v => clegSetKey(v) === key) };
    },
};

// `randRmN`/`randRmP`: both `(T{}, number) -> T{}`, differing only in how the second argument
// becomes a removal count - share this one checkCall rather than duplicating its arg-count/
// arg-type checks. Like `len`, their result type depends on the actual argument type (here, the
// whole input set type passes through unchanged), so they need a hand-written checkCall rather
// than fixedSignature(...).
function randRmCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 1: expected a set, got ${typeToString(argTypes[0])}`);
    if (argTypes[1].kind !== 'number')
        throw new Error(`cleg: '${callee}' argument 2: expected number, got ${typeToString(argTypes[1])}`);
    return argTypes[0];
}

// Randomly (uniformly) removes `count` elements from a set of any element type, mirroring
// shared/selector.ts's own `(rrmn <num> SEL)` selector semantics - reuses that file's own
// randomlyRemove() rather than reimplementing the same partial shuffle.
BUILTIN_FUNCTIONS['randRmN'] = {
    checkCall: randRmCheckCall,
    call(args) {
        const s = args[0] as { kind: 'set'; elem: ClegType; value: ClegValue[] };
        const count = (args[1] as { value: number }).value;
        if (!Number.isInteger(count) || count < 0)
            throw new Error(`cleg: 'randRmN' count must be a nonnegative integer, got ${count}`);
        return { kind: 'set', elem: s.elem, value: randomlyRemove(s.value, count) };
    },
};

// Randomly removes a fixed portion of a set - `frac` (a nonnegative float, not necessarily <= 1)
// times the set's own size, rounded down - mirroring shared/selector.ts's own `(rrmp <num> SEL)`
// exactly (including that a `frac` big enough to exceed the set's size just empties it, via
// randomlyRemove's own clamp, same as rrmp's own behavior).
BUILTIN_FUNCTIONS['randRmP'] = {
    checkCall: randRmCheckCall,
    call(args) {
        const s = args[0] as { kind: 'set'; elem: ClegType; value: ClegValue[] };
        const frac = (args[1] as { value: number }).value;
        if (!Number.isFinite(frac) || frac < 0)
            throw new Error(`cleg: 'randRmP' portion must be a nonnegative number, got ${frac}`);
        return { kind: 'set', elem: s.elem, value: randomlyRemove(s.value, Math.floor(frac * s.value.length)) };
    },
};

// The ClegType 'kind' a set of SelectorType `k` is made of - 'node' selections are plain numbers
// (node indices), the other three match their own SelectorType name exactly. Shared by
// resolveSelectorArg below (which needs to validate a `set`-typed selector argument against an
// already-known wantKind) and Selector's own 'raw' variant (shared/types.ts), which this builds.
const SELECTOR_SET_ELEM_KIND: Record<SelectorType, ClegType['kind']> = {
    node: 'number', edge: 'edge', tri: 'tri', quad: 'quad',
};

// Unwraps a `set` ClegValue's own elements into the matching SelectedVals branch - `kind` must
// already be known to match the set's own elem type (checked by resolveSelectorArg's caller before
// this runs). 'node' collects into a real Set<number> (numbers have genuine equality); the other
// three stay plain arrays, matching SelectedVals' own doc comment on why.
function setValueToSelectedVals(kind: SelectorType, values: ClegValue[]): SelectedVals {
    switch (kind) {
        case 'node': return { kind: 'node', value: new Set(values.map(v => (v as { value: number }).value)) };
        case 'edge': return { kind: 'edge', value: values.map(v => (v as { value: BoardEdge }).value) };
        case 'tri': return { kind: 'tri', value: values.map(v => (v as { value: BoardTriangle }).value) };
        case 'quad': return { kind: 'quad', value: values.map(v => (v as { value: BoardQuad }).value) };
    }
}

// Resolves a nis/eis/triangleForm/quadForm-style "selector-like" argument into a real Selector -
// a `sel` value (its actual kind checked against `wantKind` at runtime, since 'sel' carries no kind
// at the type level - see ClegType's own 'sel' doc comment), a `string` (parsed via `parseFn`,
// following shared/selector.ts's own grammar exactly), or a `set` (of the ClegType matching
// `wantKind` - see SELECTOR_SET_ELEM_KIND - wrapped directly into a `raw` Selector, no parsing
// involved). Shared by every builtin that accepts this shape, so the "string, sel, or set - kind-
// checked" logic exists in exactly one place.
function resolveSelectorArg(
    callee: string, arg: ClegValue, wantKind: SelectorType, parseFn: (s: string) => Selector,
): Selector {
    if (arg.kind === 'string') return parseFn(arg.value);
    if (arg.kind === 'set') {
        if (arg.elem.kind !== SELECTOR_SET_ELEM_KIND[wantKind])
            throw new Error(
                `cleg: '${callee}' expects a ${wantKind} selector, got a set of ${typeToString(arg.elem)}`);
        return { op: 'raw', type: wantKind, items: setValueToSelectedVals(wantKind, arg.value) };
    }
    const sel = arg as { kind: 'sel'; selType: SelectorType; value: Selector };
    if (sel.selType !== wantKind)
        throw new Error(`cleg: '${callee}' expects a ${wantKind} selector, got a '${sel.selType}' selector`);
    return sel.value;
}

const NUMBER_TYPE: ClegType = { kind: 'number' };
// Exported for shared/clegCheck.ts's own typecheckCleg, which needs a harmless placeholder
// ClegType to pass as checkStmt's own `returnType` param while checking a TopStmt (never actually
// read there, since a TopStmt is never a ReturnStmt).
export const EGR_TYPE: ClegType = { kind: 'egr' };

// `abs(x)`/`sqrt(x)`: fixed-signature `number -> number`, built with fixedSignature(...) like
// mkEdge/mkTri/mkQuad below. `sqrt` throws at evaluation time for a negative `x` (not statically
// knowable from `number`'s type alone) rather than returning NaN, matching every other cleg
// evaluation-time validity check (e.g. requireRepeatCount's own nonnegative-integer requirement).
BUILTIN_FUNCTIONS['abs'] = {
    checkCall: fixedSignature([NUMBER_TYPE], NUMBER_TYPE),
    call: ([x]) => ({ kind: 'number', value: Math.abs((x as { value: number }).value) }),
};
BUILTIN_FUNCTIONS['sqrt'] = {
    checkCall: fixedSignature([NUMBER_TYPE], NUMBER_TYPE),
    call: ([x]) => {
        const v = (x as { value: number }).value;
        if (v < 0) throw new Error(`cleg: 'sqrt' argument must be nonnegative, got ${v}`);
        return { kind: 'number', value: Math.sqrt(v) };
    },
};

// `mkEdge`/`mkTri`/`mkQuad`: build an edge/tri/quad from node indices, canonicalized exactly as
// shared/types.ts's own makeBoardEdge/makeBoardTriangle/makeBoardQuad do (mkQuad's arguments must
// already be in cycle order, same requirement as makeBoardQuad's own - see its doc comment). All
// three are fixed-signature (number, ..., number) -> edge/tri/quad, so they're built with
// fixedSignature(...) like the board constructors above rather than needing a hand-written checkCall.
BUILTIN_FUNCTIONS['mkEdge'] = {
    checkCall: fixedSignature([NUMBER_TYPE, NUMBER_TYPE], { kind: 'edge' }),
    call: ([a, b]) => ({
        kind: 'edge',
        value: makeBoardEdge((a as { value: number }).value, (b as { value: number }).value),
    }),
};
BUILTIN_FUNCTIONS['mkTri'] = {
    checkCall: fixedSignature([NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE], { kind: 'tri' }),
    call: ([a, b, c]) => ({
        kind: 'tri',
        value: makeBoardTriangle(
            (a as { value: number }).value, (b as { value: number }).value, (c as { value: number }).value),
    }),
};
BUILTIN_FUNCTIONS['mkQuad'] = {
    checkCall: fixedSignature([NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE, NUMBER_TYPE], { kind: 'quad' }),
    call: ([a, b, c, d]) => ({
        kind: 'quad',
        value: makeBoardQuad(
            (a as { value: number }).value, (b as { value: number }).value,
            (c as { value: number }).value, (d as { value: number }).value),
    }),
};

// `prod(a, b)`: the graph (tensor) product of two boards - shared/boardConfig.ts's own product(),
// fixed-signature (egr, egr) -> egr like mkEdge/mkTri/mkQuad above.
BUILTIN_FUNCTIONS['prod'] = {
    checkCall: fixedSignature([EGR_TYPE, EGR_TYPE], { kind: 'egr' }),
    call: ([a, b]) => ({
        kind: 'egr',
        value: product((a as { value: BoardConfig }).value, (b as { value: BoardConfig }).value),
    }),
};

// `mkSel(X)`: builds a selector from X - a `string` (parsed via selector.ts's own context-free
// parseSelector, whichever kind it turns out to be bottom-up - see that file's own top comment) or a
// `set` (of number/edge/tri/quad, wrapped into a `raw` Selector, its own kind read off the set's own
// element type), resolved via resolveAnyKindSelectorArg below exactly like msBase's own selector
// argument - there's no separate `kind` argument anymore, since a Selector already self-describes its
// own kind, so reading it off X directly replaces the old design where mkSel had to be told which
// kind to parse X as. For "every object of some kind K", pass the text "(all K)" directly rather than
// omitting an argument - there's no longer a second, optional argument to omit either. Hand-written
// checkCall (rather than fixedSignature(...)) since X's own accepted type isn't just one fixed
// ClegType, and `sel` isn't parameterized by kind at the type level (see ClegType's own 'sel' doc
// comment).
function mkSelCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 1)
        throw new Error(`cleg: '${callee}' expects 1 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'string' && argTypes[0].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 1: expected string or set, got ${typeToString(argTypes[0])}`);
    return { kind: 'sel' };
}
BUILTIN_FUNCTIONS['mkSel'] = {
    checkCall: mkSelCheckCall,
    call: ([arg]) => {
        const sel = resolveAnyKindSelectorArg('mkSel', arg);
        return { kind: 'sel', selType: sel.type, value: sel };
    },
};

const MOD_TYPE: ClegType = { kind: 'mod' };

// `rectify`/`truncate`/`globalCentralize`/`quadOctarize`: zero-argument BoardModifier constructors,
// one per shared/types.ts's own like-named BoardModifier kind - build the value directly
// (`{ kind: 'X' }`) rather than calling shared/boardConfig.ts's own
// rectify()/truncate()/globalCentralize()/quadOctarize() (those APPLY a modifier to a board
// immediately; these instead build the modifier value itself, to be applied later - see
// shared/clegBase.ts's own top comment on the `mod` type).
BUILTIN_FUNCTIONS['rectify'] = {
    checkCall: fixedSignature([], MOD_TYPE),
    call: () => ({ kind: 'mod', value: { kind: 'Rectify' } }),
};
BUILTIN_FUNCTIONS['truncate'] = {
    checkCall: fixedSignature([], MOD_TYPE),
    call: () => ({ kind: 'mod', value: { kind: 'Truncate' } }),
};
BUILTIN_FUNCTIONS['globalCentralize'] = {
    checkCall: fixedSignature([], MOD_TYPE),
    call: () => ({ kind: 'mod', value: { kind: 'GlobalCentralize' } }),
};
BUILTIN_FUNCTIONS['quadOctarize'] = {
    checkCall: fixedSignature([], MOD_TYPE),
    call: () => ({ kind: 'mod', value: { kind: 'QuadOctarize' } }),
};

// `edgeSplit`/`mergeClose`/`scale`: one-number-argument BoardModifier constructors, same
// "build the value, don't apply it" rationale as rectify/globalCentralize/quadOctarize above.
BUILTIN_FUNCTIONS['edgeSplit'] = {
    checkCall: fixedSignature([NUMBER_TYPE], MOD_TYPE),
    call: ([n]) => ({ kind: 'mod', value: { kind: 'EdgeSplit', splitN: (n as { value: number }).value } }),
};
BUILTIN_FUNCTIONS['mergeClose'] = {
    checkCall: fixedSignature([NUMBER_TYPE], MOD_TYPE),
    call: ([d]) => ({ kind: 'mod', value: { kind: 'MergeClose', dist: (d as { value: number }).value } }),
};
BUILTIN_FUNCTIONS['scale'] = {
    checkCall: fixedSignature([NUMBER_TYPE], MOD_TYPE),
    call: ([f]) => ({ kind: 'mod', value: { kind: 'Scale', factor: (f as { value: number }).value } }),
};

// `nis(X)`/`eis(X)`: build a NodeInducedSubgraph/EdgeInducedSubgraph BoardModifier - `X` (a `sel` or
// `string`, resolved via resolveSelectorArg above) becomes that modifier's own `sel: Selector`
// field. Same family as triangleForm/quadForm just below (construct the value, don't apply it), but
// simpler - no `w`, and the selector is mandatory rather than optional (NodeInducedSubgraph/
// EdgeInducedSubgraph's own `sel` field isn't `?`). Unlike a `number{}`/`edge{}` set (nis/eis's own
// earlier, since-removed third accepted shape), there's no Selector grammar production for "exactly
// this literal set of nodes/edges" (see shared/clegBase.ts's own top comment history), so a real
// NodeInducedSubgraph/EdgeInducedSubgraph modifier value can only ever hold a genuine Selector.
function inducedSubgraphModCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 1)
        throw new Error(`cleg: '${callee}' expects 1 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'sel' && argTypes[0].kind !== 'string' && argTypes[0].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 1: expected sel, string, or set, got ${typeToString(argTypes[0])}`);
    return MOD_TYPE;
}
BUILTIN_FUNCTIONS['nis'] = {
    checkCall: inducedSubgraphModCheckCall,
    call: ([arg]) => (
        { kind: 'mod', value: { kind: 'NodeInducedSubgraph', sel: resolveSelectorArg('nis', arg, 'node', parseNodeSelector) } }
    ),
};
BUILTIN_FUNCTIONS['eis'] = {
    checkCall: inducedSubgraphModCheckCall,
    call: ([arg]) => (
        { kind: 'mod', value: { kind: 'EdgeInducedSubgraph', sel: resolveSelectorArg('eis', arg, 'edge', parseEdgeSelector) } }
    ),
};

const EDGE_TYPE: ClegType = { kind: 'edge' };
const TRI_TYPE: ClegType = { kind: 'tri' };
const QUAD_TYPE: ClegType = { kind: 'quad' };

// `selectNode(X, bc)`/`selectEdge(X, bc)`/`selectTriangle(X, bc)`/`selectQuad(X, bc)`: evaluates a
// selector (`X`, a `sel`, `string`, or `set` - resolved via resolveSelectorArg above, same convention as
// nis/eis/triangleForm/quadForm) against a real board `bc`, returning the exact set of
// nodes/edges/triangles/quads it selects (shared/selector.ts's own selectNode/selectEdge/
// selectTriangle/selectQuad do the actual work) - unlike nis/eis (which build a
// NodeInducedSubgraph/EdgeInducedSubgraph BoardModifier to apply LATER via modify(...)), this
// evaluates the selector immediately against a real board and hands back the result as an ordinary
// cleg set, so a program can inspect/combine/count (len) it directly. One builtin per selector kind,
// rather than a single overloaded name, because `sel`'s own ClegType carries no kind at the type
// level (see shared/clegBase.ts's own top comment) - checkCall only ever sees argument TYPES, never
// their runtime values, so there's no way for one `select(X, bc)` to know ahead of time which of
// number{}/edge{}/tri{}/quad{} it should return.
function selectSetCheckCall(elemType: ClegType): BuiltinFunction['checkCall'] {
    return (callee, argTypes) => {
        if (argTypes.length !== 2)
            throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
        if (argTypes[0].kind !== 'sel' && argTypes[0].kind !== 'string' && argTypes[0].kind !== 'set')
            throw new Error(`cleg: '${callee}' argument 1: expected sel, string, or set, got ${typeToString(argTypes[0])}`);
        if (argTypes[1].kind !== 'egr')
            throw new Error(`cleg: '${callee}' argument 2: expected egr, got ${typeToString(argTypes[1])}`);
        return { kind: 'set', elem: elemType };
    };
}
BUILTIN_FUNCTIONS['selectNode'] = {
    checkCall: selectSetCheckCall(NUMBER_TYPE),
    call: ([arg, egrVal]) => {
        const sel = resolveSelectorArg('selectNode', arg, 'node', parseNodeSelector);
        const bc = (egrVal as { value: BoardConfig }).value;
        const nodes = [...selectNode(bc.adj, bc.emb.pos, sel)].map((n): ClegValue => ({ kind: 'number', value: n }));
        return makeClegSet(NUMBER_TYPE, nodes);
    },
};
BUILTIN_FUNCTIONS['selectEdge'] = {
    checkCall: selectSetCheckCall(EDGE_TYPE),
    call: ([arg, egrVal]) => {
        const sel = resolveSelectorArg('selectEdge', arg, 'edge', parseEdgeSelector);
        const bc = (egrVal as { value: BoardConfig }).value;
        const edges = selectEdge(bc.adj, bc.emb.pos, sel).map((e): ClegValue => ({ kind: 'edge', value: e }));
        return makeClegSet(EDGE_TYPE, edges);
    },
};
BUILTIN_FUNCTIONS['selectTriangle'] = {
    checkCall: selectSetCheckCall(TRI_TYPE),
    call: ([arg, egrVal]) => {
        const sel = resolveSelectorArg('selectTriangle', arg, 'tri', parseTriangleSelector);
        const bc = (egrVal as { value: BoardConfig }).value;
        const tris = selectTriangle(bc.adj, bc.emb.pos, sel).map((t): ClegValue => ({ kind: 'tri', value: t }));
        return makeClegSet(TRI_TYPE, tris);
    },
};
BUILTIN_FUNCTIONS['selectQuad'] = {
    checkCall: selectSetCheckCall(QUAD_TYPE),
    call: ([arg, egrVal]) => {
        const sel = resolveSelectorArg('selectQuad', arg, 'quad', parseQuadSelector);
        const bc = (egrVal as { value: BoardConfig }).value;
        const quads = selectQuad(bc.adj, bc.emb.pos, sel).map((q): ClegValue => ({ kind: 'quad', value: q }));
        return makeClegSet(QUAD_TYPE, quads);
    },
};

// `triangleForm(w, [selArg])`/`quadForm(w, [selArg])`: builds a TriangleForm/QuadForm
// BoardModifier - `selArg` (a `sel`, `string`, or `set`, resolved via resolveSelectorArg) restricts which
// triangles/quads get replaced, mirroring TriangleForm/QuadForm's own optional `sel?: Selector`
// field exactly - omitted, every one found gets replaced. Variable-arity (1 or 2 args) rather than
// fixedSignature(...), to mirror that optionality exactly.
function formModCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 1 && argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 1 or 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'number')
        throw new Error(`cleg: '${callee}' argument 1: expected number, got ${typeToString(argTypes[0])}`);
    if (argTypes.length === 2 && argTypes[1].kind !== 'sel' && argTypes[1].kind !== 'string' && argTypes[1].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 2: expected sel, string, or set, got ${typeToString(argTypes[1])}`);
    return MOD_TYPE;
}
BUILTIN_FUNCTIONS['triangleForm'] = {
    checkCall: formModCheckCall,
    call: (args) => {
        const w = (args[0] as { value: number }).value;
        if (args.length === 1) return { kind: 'mod', value: { kind: 'TriangleForm', w } };
        const sel = resolveSelectorArg('triangleForm', args[1], 'tri', parseTriangleSelector);
        return { kind: 'mod', value: { kind: 'TriangleForm', w, sel } };
    },
};
BUILTIN_FUNCTIONS['quadForm'] = {
    checkCall: formModCheckCall,
    call: (args) => {
        const w = (args[0] as { value: number }).value;
        if (args.length === 1) return { kind: 'mod', value: { kind: 'QuadForm', w } };
        const sel = resolveSelectorArg('quadForm', args[1], 'quad', parseQuadSelector);
        return { kind: 'mod', value: { kind: 'QuadForm', w, sel } };
    },
};

// `form(w, ...sels)`: builds a Form BoardModifier - `w` (the shared lattice width) followed by one
// or more selector arguments, each a `sel` (typically built via `mkSel(...)`), a bare `string`
// (parsed via selector.ts's own context-free parseSelector - kind inferred bottom-up from the text
// itself, exactly like mkSel's own string case, so `mkSel` is no longer needed just to wrap one), or
// a `set` of number/edge/tri/quad (kind read off the set's own element type) - resolved via
// resolveAnyKindSelectorArg below, the same resolution mkSel/msBase already use. Mirrors genericForm's
// own (bc, w, sels) signature (genericForm itself accepts an empty `sels` as a no-op; `form` requires
// at least one, below, since a cleg call with none would be a pointless no-op board program). None of
// `sel`/`string`/`set` carries a tri-or-quad kind at the type level, so a non-tri/quad argument
// type-checks here but is rejected at runtime by genericForm itself - the same check any hand-built
// Selector needs, not something `form` repeats.
function formCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length < 2)
        throw new Error(`cleg: '${callee}' expects at least 2 argument(s) (w, and >= 1 sel), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'number')
        throw new Error(`cleg: '${callee}' argument 1: expected number, got ${typeToString(argTypes[0])}`);
    for (let i = 1; i < argTypes.length; i++)
        if (argTypes[i].kind !== 'sel' && argTypes[i].kind !== 'string' && argTypes[i].kind !== 'set')
            throw new Error(`cleg: '${callee}' argument ${i + 1}: expected sel, string, or set, got ${typeToString(argTypes[i])}`);
    return MOD_TYPE;
}
BUILTIN_FUNCTIONS['form'] = {
    checkCall: formCheckCall,
    call: (args) => {
        const w = (args[0] as { value: number }).value;
        const sels = args.slice(1).map(a => resolveAnyKindSelectorArg('form', a));
        return { kind: 'mod', value: { kind: 'Form', w, sels } };
    },
};

// `triCentralize([selArg])`/`quadCentralize([selArg])`: builds a TriCentralize/QuadCentralize
// BoardModifier - `selArg` (a `sel`, `string`, or `set`, resolved via resolveSelectorArg) restricts
// which triangles/quads get centralized, mirroring TriCentralize/QuadCentralize's own optional
// `sel?: Selector` field exactly - omitted, every one found gets centralized. Variable-arity (0 or 1
// args) rather than fixedSignature(...), to mirror that optionality exactly.
function centralizeModCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 0 && argTypes.length !== 1)
        throw new Error(`cleg: '${callee}' expects 0 or 1 argument(s), got ${argTypes.length}`);
    if (argTypes.length === 1 && argTypes[0].kind !== 'sel' && argTypes[0].kind !== 'string' && argTypes[0].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 1: expected sel, string, or set, got ${typeToString(argTypes[0])}`);
    return MOD_TYPE;
}
BUILTIN_FUNCTIONS['triCentralize'] = {
    checkCall: centralizeModCheckCall,
    call: (args) => {
        if (args.length === 0) return { kind: 'mod', value: { kind: 'TriCentralize' } };
        const sel = resolveSelectorArg('triCentralize', args[0], 'tri', parseTriangleSelector);
        return { kind: 'mod', value: { kind: 'TriCentralize', sel } };
    },
};
BUILTIN_FUNCTIONS['quadCentralize'] = {
    checkCall: centralizeModCheckCall,
    call: (args) => {
        if (args.length === 0) return { kind: 'mod', value: { kind: 'QuadCentralize' } };
        const sel = resolveSelectorArg('quadCentralize', args[0], 'quad', parseQuadSelector);
        return { kind: 'mod', value: { kind: 'QuadCentralize', sel } };
    },
};

// `centralize(...sels)`: builds a Centralize BoardModifier - one or more selector arguments (each a
// `sel`, bare `string`, or `set` - resolved via resolveAnyKindSelectorArg, same as form/mkSel/msBase),
// mirroring genericCentralize's own (bc, sels) signature (genericCentralize itself accepts an empty
// `sels` as a no-op; `centralize` requires at least one, below, since a cleg call with none would be
// a pointless no-op board program). None of `sel`/`string`/`set` carries a tri-or-quad kind at the
// type level, so a non-tri/quad argument type-checks here but is rejected at runtime by
// genericCentralize itself - the same check any hand-built Selector needs, not something
// `centralize` repeats.
function centralizeCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length < 1)
        throw new Error(`cleg: '${callee}' expects at least 1 argument(s), got ${argTypes.length}`);
    for (let i = 0; i < argTypes.length; i++)
        if (argTypes[i].kind !== 'sel' && argTypes[i].kind !== 'string' && argTypes[i].kind !== 'set')
            throw new Error(`cleg: '${callee}' argument ${i + 1}: expected sel, string, or set, got ${typeToString(argTypes[i])}`);
    return MOD_TYPE;
}
BUILTIN_FUNCTIONS['centralize'] = {
    checkCall: centralizeCheckCall,
    call: (args) => {
        const sels = args.map(a => resolveAnyKindSelectorArg('centralize', a));
        return { kind: 'mod', value: { kind: 'Centralize', sels } };
    },
};

// `modify(mods, bc)`: applies every modifier in `mods`, in order, to `bc` - shared/boardConfig.ts's
// own applyModifiers(), the one builtin that actually turns a list of `mod` values into a
// transformed board (every rectify/edgeSplit/.../form/nis/eis builtin above only constructs an
// opaque `mod` value, never applies one). `mods` is a plain array (`mod[]`), not a set - modifiers
// are meaningfully ordered and can repeat (e.g. `[scale(2), scale(2)]` is not the same as one
// `scale(4)`), neither of which a set would preserve.
BUILTIN_FUNCTIONS['modify'] = {
    checkCall: fixedSignature([{ kind: 'array', elem: MOD_TYPE }, EGR_TYPE], EGR_TYPE),
    call: ([modsVal, egrVal]) => {
        const mods = (modsVal as { value: ClegValue[] }).value.map(v => (v as { value: BoardModifier }).value);
        const bc = (egrVal as { value: BoardConfig }).value;
        return { kind: 'egr', value: applyModifiers(bc, mods) };
    },
};

// ── multiProd: N-ary Cartesian board product, restricted by a MultiSelector ────

const MSEL_TYPE: ClegType = { kind: 'msel' };

// Inverse of SELECTOR_SET_ELEM_KIND above (ClegType elem kind -> SelectorType) - used only by
// resolveAnyKindSelectorArg below, which (unlike resolveSelectorArg) has no fixed wantKind of its
// own to check a set's element type against, so it has to recover a SelectorType FROM the set's own
// element kind instead.
const SELECTOR_TYPE_BY_SET_ELEM_KIND: Partial<Record<ClegType['kind'], SelectorType>> = {
    number: 'node', edge: 'edge', tri: 'tri', quad: 'quad',
};

// Resolves a selector argument whose own kind isn't fixed ahead of the call (mkSel/form/msBase) into
// a real Selector - a `sel` value (used directly, whatever SelectorType it is), a `string` (parsed via
// selector.ts's own context-free parseSelector, whichever kind the text itself turns out to be -
// unlike resolveSelectorArg's own string case, which parses against one fixed wantKind), or a `set`
// (of number/edge/tri/quad, wrapped into a `raw` Selector the same way resolveSelectorArg's own
// `set` case does, its own kind read off the set's own element type).
function resolveAnyKindSelectorArg(callee: string, arg: ClegValue): Selector {
    if (arg.kind === 'sel') return arg.value;
    if (arg.kind === 'string') return parseSelector((arg as { value: string }).value);
    if (arg.kind === 'set') {
        const wantKind = SELECTOR_TYPE_BY_SET_ELEM_KIND[arg.elem.kind];
        if (!wantKind)
            throw new Error(
                `cleg: '${callee}': a selector set must be a set of number/edge/tri/quad, got a set of ${typeToString(arg.elem)}`);
        return { op: 'raw', type: wantKind, items: setValueToSelectedVals(wantKind, arg.value) };
    }
    throw new Error(`cleg: '${callee}': expected sel, string, or set, got ${typeToString(clegValueType(arg))}`);
}

function msBaseCheckCall(callee: string, argTypes: ClegType[]): ClegType {
    if (argTypes.length !== 2)
        throw new Error(`cleg: '${callee}' expects 2 argument(s), got ${argTypes.length}`);
    if (argTypes[0].kind !== 'number')
        throw new Error(`cleg: '${callee}' argument 1: expected number, got ${typeToString(argTypes[0])}`);
    if (argTypes[1].kind !== 'sel' && argTypes[1].kind !== 'string' && argTypes[1].kind !== 'set')
        throw new Error(`cleg: '${callee}' argument 2: expected sel, string, or set, got ${typeToString(argTypes[1])}`);
    return MSEL_TYPE;
}
// `msAll()`: every node of the full product, unrestricted - see MultiSelector's own 'all' doc
// comment.
BUILTIN_FUNCTIONS['msAll'] = {
    checkCall: fixedSignature([], MSEL_TYPE),
    call: () => ({ kind: 'msel', value: { op: 'all' } }),
};

// `msBase(number, X)`: "every full-product node whose `number`-th coordinate is kept by X, every
// other coordinate unrestricted" - see MultiSelector's own doc comment for what X may be. X is a
// `sel`, a bare `string` (parsed via resolveAnyKindSelectorArg below - kind inferred bottom-up from
// the text itself, same as form/mkSel), or a `set`.
BUILTIN_FUNCTIONS['msBase'] = {
    checkCall: msBaseCheckCall,
    call: ([numberVal, arg]) => {
        const number = (numberVal as { value: number }).value;
        if (!Number.isInteger(number) || number < 0)
            throw new Error(`cleg: msBase: number must be a nonnegative integer, got ${number}`);
        const sel = resolveAnyKindSelectorArg('msBase', arg);
        return { kind: 'msel', value: { op: 'base', number, sel } };
    },
};

// `msUnion(items)`/`msInter(items)`: fixed-signature (msel[] -> msel) - a plain array, not a set,
// mirroring modify's own mods array (see its own doc comment) and the underlying Selector grammar's
// own (union SEL...)/(inter SEL...), which are similarly plain lists.
BUILTIN_FUNCTIONS['msUnion'] = {
    checkCall: fixedSignature([{ kind: 'array', elem: MSEL_TYPE }], MSEL_TYPE),
    call: ([itemsVal]) => ({
        kind: 'msel',
        value: {
            op: 'union',
            items: (itemsVal as { value: ClegValue[] }).value.map(v => (v as { value: MultiSelector }).value),
        },
    }),
};
BUILTIN_FUNCTIONS['msInter'] = {
    checkCall: fixedSignature([{ kind: 'array', elem: MSEL_TYPE }], MSEL_TYPE),
    call: ([itemsVal]) => ({
        kind: 'msel',
        value: {
            op: 'inter',
            items: (itemsVal as { value: ClegValue[] }).value.map(v => (v as { value: MultiSelector }).value),
        },
    }),
};
// `msDiff(a, b)`: fixed-signature (msel, msel -> msel).
BUILTIN_FUNCTIONS['msDiff'] = {
    checkCall: fixedSignature([MSEL_TYPE, MSEL_TYPE], MSEL_TYPE),
    call: ([a, b]) => ({
        kind: 'msel',
        value: {
            op: 'diff',
            a: (a as { value: MultiSelector }).value,
            b: (b as { value: MultiSelector }).value,
        },
    }),
};

// Fixed once per multiProd call, from `boards`' own ORIGINAL (unrestricted) sizes: `Ns[k]` is
// boards[k]'s own node count, and `stride[k]` lets any tuple of per-board node indices flatten
// to/from one "original" flat index into the full (never fully materialized)
// Ns[0] x Ns[1] x ... x Ns[N-1] product space - the one shared universe every MultiSelector
// combinator (msUnion/msInter/msDiff) has to combine its own operands against. This has to be fixed
// UP FRONT, before any msBase/msUnion/msInter/msDiff runs: two differently-restricted intermediate
// boards (e.g. one msBase restricting board 0, another restricting board 1) would otherwise have no
// common index space to combine against at all.
interface FullProductIndex { Ns: number[]; stride: number[]; total: number; }

function makeFullProductIndex(boards: BoardConfig[]): FullProductIndex {
    const Ns = boards.map(b => b.N);
    const stride = new Array<number>(Ns.length);
    stride[Ns.length - 1] = 1;
    for (let k = Ns.length - 2; k >= 0; k--) stride[k] = stride[k + 1] * Ns[k + 1];
    const total = Ns.reduce((p, n) => p * n, 1);
    return { Ns, stride, total };
}

function fullIndexOf(fpi: FullProductIndex, tuple: number[]): number {
    return tuple.reduce((sum, n, k) => sum + n * fpi.stride[k], 0);
}

function tupleOfFullIndex(fpi: FullProductIndex, idx: number): number[] {
    return fpi.Ns.map((n, k) => Math.floor(idx / fpi.stride[k]) % n);
}

// Restricts `board` (always boards[msel.number], see evalMultiSelector's own 'base' case) to just
// the nodes `sel` keeps - nodeInducedSubgraph directly for a node selector, or (mirroring
// edgeInducedSubgraph's own "which nodes survive" rule) the nodes touched by at least one selected
// edge for an edge selector. Returns the restricted board AND which of `board`'s own ORIGINAL node
// indices survived, in the same ascending order nodeInducedSubgraph/edgeInducedSubgraph themselves
// compact to - evalMultiSelector's own full-product index bookkeeping needs that mapping to place
// the restricted board's own local nodes back into the fixed full index space (FullProductIndex).
// Throws for any SelectorType other than node/edge - unlike msBase itself (which accepts any
// SelectorType at the data-structure level - see MultiSelector's own doc comment), multiProd's own
// evaluation requires node or edge specifically, since there's no other sensible way to turn a
// tri/quad selection into "which nodes of this one factor board survive".
function restrictBoardBySelector(board: BoardConfig, sel: Selector): { bc: BoardConfig; survivors: number[] } {
    if (sel.type === 'node') {
        const kept = selectNode(board.adj, board.emb.pos, sel);
        return { bc: nodeInducedSubgraph(board, kept), survivors: [...kept].sort((a, b) => a - b) };
    }
    if (sel.type === 'edge') {
        const edges = selectEdge(board.adj, board.emb.pos, sel);
        const kept = new Set(edges.flatMap(e => [e.n1, e.n2]));
        return { bc: edgeInducedSubgraph(board, edges), survivors: [...kept].sort((a, b) => a - b) };
    }
    throw new Error(
        `cleg: multiProd: msBase's own selector must be a node or edge selector, got a '${sel.type}' selector`);
}

// Every original flat index, 0..fpi.total-1 - the universal set 'all' and 'inter' (with zero
// operands) both denote (see MultiSelector's own doc comment on why those two coincide).
function universalOriginalIndices(fpi: FullProductIndex): Set<number> {
    const all = new Set<number>();
    for (let i = 0; i < fpi.total; i++) all.add(i);
    return all;
}

// Builds a real BoardConfig for an arbitrary subset of the full product's node space, given only the
// kept ORIGINAL flat indices (`keptOriginal`) - decomposes each back into its own per-board tuple
// (via `fpi`) to compute adjacency (Cartesian product rule: two kept nodes are adjacent iff they
// differ in EXACTLY one coordinate k, adjacent there in boards[k]) and embedding (per-board
// positions concatenated) directly, never materializing the full product's own (possibly enormous)
// N x N adjacency matrix. Compacts to a fresh 0..K-1 range in ascending original-index order (same
// convention as nodeInducedSubgraph/edgeInducedSubgraph) - returns both the new BoardConfig and
// origIndex (new local index -> kept original flat index), so a further msUnion/msInter/msDiff can
// keep combining against the very same fixed full-product index space. Used by every
// evalMultiSelector case except 'base' (which instead reuses the real product() function directly -
// see its own comment on why that still ends up with the exact same adjacency/embedding).
function buildFromOriginalIndices(
    boards: BoardConfig[], fpi: FullProductIndex, keptOriginal: Set<number>,
): { bc: BoardConfig; origIndex: number[] } {
    const origIndex = [...keptOriginal].sort((a, b) => a - b);
    const tuples = origIndex.map(idx => tupleOfFullIndex(fpi, idx));
    const embDim = boards.reduce((s, b) => s + b.emb.embDim, 0);
    const pos = tuples.map(tuple => tuple.flatMap((n, k) => boards[k].emb.pos[n]));
    const K = origIndex.length;
    const adj = zeroAdj(K);
    for (let a = 0; a < K; a++) {
        for (let b = a + 1; b < K; b++) {
            let diffCoord = -1;
            let tooManyDiffs = false;
            for (let k = 0; k < boards.length; k++) {
                if (tuples[a][k] !== tuples[b][k]) {
                    if (diffCoord !== -1) { tooManyDiffs = true; break; }
                    diffCoord = k;
                }
            }
            if (!tooManyDiffs && diffCoord >= 0 && boards[diffCoord].adj[tuples[a][diffCoord]][tuples[b][diffCoord]]) {
                adj[a][b] = 1;
                adj[b][a] = 1;
            }
        }
    }
    return { bc: make(new Embedding(embDim, pos), adj), origIndex };
}

/**
 * Evaluates a MultiSelector against `boards`/`fpi` into a real BoardConfig plus origIndex (see
 * buildFromOriginalIndices) - every combinator ultimately reduces to a set operation over
 * origIndex's own shared "which of the full product's original flat indices survive" universe (see
 * FullProductIndex's own doc comment on why that has to be fixed up front, from `boards`' own
 * unrestricted sizes, rather than derived along the way).
 *
 * 'base' restricts boards[number] (see restrictBoardBySelector) and folds the real product()
 * pairwise, left to right, across every factor (boards[number] replaced by its restriction) -
 * mathematically identical adjacency/embedding to a direct N-ary construction, since product()'s own
 * row-major node indexing composes correctly under folding (`product(product(A,B),C)`'s own index
 * `(a*NB+b)*NC+c` already equals the direct 3-ary flattening `a*NB*NC+b*NC+c`). The resulting local
 * node indices are then translated back into the fixed full-product index space one at a time
 * (decompose via the RESTRICTED factors' own sizes, substitute the restricted coordinate's own local
 * index for its ORIGINAL boards[number] index via `survivors`, then flatten via `fpi`).
 *
 * 'union'/'inter'/'diff' recursively evaluate their own operands first (discarding each one's own
 * `bc`, since only its origIndex set matters for combining), combine via ordinary Set operations,
 * then materialize a fresh BoardConfig for exactly the combined set via buildFromOriginalIndices.
 * 'inter' with zero operands is the universal set (every original index) - the usual absorbing-
 * element identity for an empty intersection fold, matching Selector's own `(inter)`.
 */
function evalMultiSelector(
    boards: BoardConfig[], fpi: FullProductIndex, msel: MultiSelector,
): { bc: BoardConfig; origIndex: number[] } {
    switch (msel.op) {
        case 'all':
            return buildFromOriginalIndices(boards, fpi, universalOriginalIndices(fpi));
        case 'base': {
            const { bc: restricted, survivors } = restrictBoardBySelector(boards[msel.number], msel.sel);
            const factorBoards = boards.map((b, i) => i === msel.number ? restricted : b);
            const bc = factorBoards.reduce((acc, b) => product(acc, b));
            const localNs = factorBoards.map(b => b.N);
            const localStride = new Array<number>(localNs.length);
            localStride[localNs.length - 1] = 1;
            for (let k = localNs.length - 2; k >= 0; k--) localStride[k] = localStride[k + 1] * localNs[k + 1];
            const origIndex = new Array<number>(bc.N);
            for (let local = 0; local < bc.N; local++) {
                const tuple = localNs.map((n, k) => Math.floor(local / localStride[k]) % n);
                tuple[msel.number] = survivors[tuple[msel.number]];
                origIndex[local] = fullIndexOf(fpi, tuple);
            }
            return { bc, origIndex };
        }
        case 'union': {
            const kept = new Set<number>();
            for (const item of msel.items)
                for (const idx of evalMultiSelector(boards, fpi, item).origIndex) kept.add(idx);
            return buildFromOriginalIndices(boards, fpi, kept);
        }
        case 'inter': {
            if (msel.items.length === 0) return buildFromOriginalIndices(boards, fpi, universalOriginalIndices(fpi));
            let kept = new Set(evalMultiSelector(boards, fpi, msel.items[0]).origIndex);
            for (let i = 1; i < msel.items.length; i++) {
                const next = new Set(evalMultiSelector(boards, fpi, msel.items[i]).origIndex);
                kept = new Set([...kept].filter(idx => next.has(idx)));
            }
            return buildFromOriginalIndices(boards, fpi, kept);
        }
        case 'diff': {
            const a = new Set(evalMultiSelector(boards, fpi, msel.a).origIndex);
            const b = new Set(evalMultiSelector(boards, fpi, msel.b).origIndex);
            return buildFromOriginalIndices(boards, fpi, new Set([...a].filter(idx => !b.has(idx))));
        }
    }
}

// `multiProd(boards, msel)`: the N-ary Cartesian product of `boards` (an egr[]), restricted to
// exactly the subgraph `msel` denotes - see evalMultiSelector's own doc comment for the full
// algorithm. `boards` must be non-empty - an N-ary product of zero factors has no principled
// definition here.
BUILTIN_FUNCTIONS['multiProd'] = {
    checkCall: fixedSignature([{ kind: 'array', elem: EGR_TYPE }, MSEL_TYPE], EGR_TYPE),
    call: ([boardsVal, mselVal]) => {
        const boards = (boardsVal as { value: ClegValue[] }).value.map(v => (v as { value: BoardConfig }).value);
        if (boards.length === 0) throw new Error(`cleg: multiProd: boards must be non-empty`);
        const fpi = makeFullProductIndex(boards);
        const msel = (mselVal as { value: MultiSelector }).value;
        return { kind: 'egr', value: evalMultiSelector(boards, fpi, msel).bc };
    },
};

// `subHcublatB(bounds, cond)`: a "sub-region" of an N-dimensional hypercubical lattice - `bounds` is
// an N-length array of `[lo, hi]` pairs (inclusive bounds, one pair per dimension, describing the
// bounding hyperrectangle - not necessarily integers themselves: `lo` is rounded UP and `hi` rounded
// DOWN to the nearest integer lattice point before use, i.e. the actual integer range is
// `[Math.ceil(lo), Math.floor(hi)]`, so a non-integer bound just trims the lattice down to the
// integer points genuinely inside `[lo, hi]` rather than being rejected); `cond` decides which lattice points inside it actually
// become nodes, called once per candidate point (as that point's own N coordinates, a `number[]`)
// via callUserFunction - the one builtin so far that needs to call back into a `func`-typed argument,
// which is why `funcs` is threaded through BuiltinFunction's own `call` signature. Surviving nodes
// keep the plain grid adjacency (connected iff their coordinates differ by exactly 1 in exactly one
// dimension) and their own lattice coordinates, re-centered (see below), as their N-dim embedding
// position - same convention, and the same full-lattice-index/stride bookkeeping to avoid an
// O(survivors^2) adjacency scan, as
// shared/boardConfig.ts's own hypercuboidBoard, just over an explicit per-dimension [lo, hi] rather
// than always starting at 0 - unlike hypercuboidBoard, the re-centering (see the end of `call`
// below) is computed from the SURVIVING nodes' own bounding box, not from `bounds` itself, since
// `cond` may keep a shape nowhere near centered within the hyperrectangle it was given.
BUILTIN_FUNCTIONS['subHcublatB'] = {
    checkCall: fixedSignature(
        [
            { kind: 'array', elem: { kind: 'array', elem: NUMBER_TYPE } },
            { kind: 'func', params: [{ kind: 'array', elem: NUMBER_TYPE }], returnType: { kind: 'bool' } },
        ],
        EGR_TYPE,
    ),
    call([boundsVal, condVal], funcs) {
        const boundsArr = (boundsVal as { value: ClegValue[] }).value;
        const k = boundsArr.length;
        if (k === 0) throw new Error(`cleg: 'subHcublatB' bounds must be non-empty`);
        const lo = new Array<number>(k);
        const dims = new Array<number>(k);
        boundsArr.forEach((pairVal, i) => {
            const pair = (pairVal as { value: ClegValue[] }).value;
            if (pair.length !== 2)
                throw new Error(`cleg: 'subHcublatB' bounds[${i}] must have exactly 2 entries (lower, upper), got ${pair.length}`);
            // `lo`/`hi` need not themselves be integers - rounded to the nearest integer lattice
            // point INWARD (lo up, hi down) before use, so e.g. `[0.5, 2.5]` becomes the integer
            // range `[1, 2]`, not an error.
            const a = Math.ceil((pair[0] as { value: number }).value);
            const b = Math.floor((pair[1] as { value: number }).value);
            if (a > b)
                throw new Error(
                    `cleg: 'subHcublatB' bounds[${i}] has no integer lattice point in range after ` +
                    `rounding (lower up, upper down), got [${a}, ${b}]`);
            lo[i] = a;
            dims[i] = b - a + 1;
        });
        // `condVal` may be a plain top-level-function reference OR a partial application (e.g.
        // `goDeskCond(l, w, h, fw, fh, in, #)`) closing over everything but the one open `number[]`
        // position - fillHoles interleaves `pointArg` into whichever slot that is, rather than
        // assuming `fn`'s own full parameter list is just `[pointArg]` (it was, for a plain
        // reference, but not in general once `cond` can be a closure).
        const cond = condVal as { name: string; boundArgs: (ClegValue | null)[] };
        const fn = funcs[cond.name];

        const strides = new Array<number>(k);
        strides[0] = 1;
        for (let i = 1; i < k; i++) strides[i] = strides[i - 1] * dims[i - 1];
        const fullN = dims.reduce((p, d) => p * d, 1);
        const localCoordsOf = (n: number): number[] => {
            const coords = new Array<number>(k);
            for (let i = 0; i < k; i++) { coords[i] = n % dims[i]; n = Math.floor(n / dims[i]); }
            return coords;
        };

        // Only surviving (cond-kept) nodes get a board index (compacted, in ascending
        // full-lattice-index order) - boardIdxOf maps a full-lattice index to that compacted index,
        // absent for a point cond rejected.
        const boardIdxOf = new Map<number, number>();
        const survivingLocal: number[][] = [];
        const pos: number[][] = [];
        for (let n = 0; n < fullN; n++) {
            const local = localCoordsOf(n);
            const point = local.map((c, i) => c + lo[i]);
            const pointArg: ClegValue = {
                kind: 'array', elem: NUMBER_TYPE, value: point.map(v => ({ kind: 'number', value: v })),
            };
            const keep = (callUserFunction(fn, fillHoles(cond.boundArgs, [pointArg]), funcs) as { value: boolean }).value;
            if (!keep) continue;
            boardIdxOf.set(n, survivingLocal.length);
            survivingLocal.push(local);
            pos.push(point);
        }
        const N = survivingLocal.length;

        const adj = zeroAdj(N);
        for (let bi = 0; bi < N; bi++) {
            const local = survivingLocal[bi];
            for (let i = 0; i < k; i++)
                for (const delta of [1, -1]) {
                    const nc = local[i] + delta;
                    if (nc < 0 || nc >= dims[i]) continue;
                    const nlocal = local.slice();
                    nlocal[i] = nc;
                    const flat = nlocal.reduce((s, c, j) => s + c * strides[j], 0);
                    const nbi = boardIdxOf.get(flat);
                    if (nbi === undefined) continue;
                    adj[bi][nbi] = 1;
                }
        }

        // Re-center: subtract each dimension's own midpoint - (min + max) / 2, computed from the
        // SURVIVING nodes' own coordinates (not `bounds` itself) - so the shape sits roughly around
        // the origin regardless of where within `bounds` `cond` happened to keep it. No-op (and no
        // division-by-zero-shaped issue) when N === 0 - there's nothing to center.
        if (N > 0) {
            const mid = new Array<number>(k);
            for (let i = 0; i < k; i++) {
                let minC = pos[0][i];
                let maxC = pos[0][i];
                for (let j = 1; j < N; j++) {
                    if (pos[j][i] < minC) minC = pos[j][i];
                    if (pos[j][i] > maxC) maxC = pos[j][i];
                }
                mid[i] = (minC + maxC) / 2;
            }
            for (const p of pos) for (let i = 0; i < k; i++) p[i] -= mid[i];
        }
        return { kind: 'egr', value: make(new Embedding(k, pos), adj) };
    },
};

// ── Evaluation ───────────────────────────────────────────────────────────────

interface ValueEnv { vars: Map<string, ClegValue>; parent: ValueEnv | null; }
// Mirrors shared/clegCheck.ts's own lookupVarType `| undefined` convention - evalExpr's own
// Identifier/CallExpr cases need to tell "not a local variable" apart from "found" (falling back to
// a top-level function reference/call in the former case), unlike lookupValue/setValue below, which
// are always used where typecheckCleg already guarantees the variable exists.
function lookupValueOptional(env: ValueEnv, name: string): ClegValue | undefined {
    for (let e: ValueEnv | null = env; e; e = e.parent) { const v = e.vars.get(name); if (v) return v; }
    return undefined;
}
function lookupValue(env: ValueEnv, name: string): ClegValue {
    const v = lookupValueOptional(env, name);
    // Unreachable in a program that has passed typecheckCleg - every Identifier/AssignStmt there
    // already resolved to a declared variable.
    if (!v) throw new Error(`cleg: undeclared variable '${name}'`);
    return v;
}

/** Mutates `name`'s existing binding in place, in whichever env of the chain declared it - unlike
 * VarDecl's own `env.vars.set` (which always creates a fresh binding in the innermost scope), this
 * walks up to the declaring scope first so an assignment inside a nested block/if-branch is visible
 * to the enclosing scope that declared the variable, not just shadowed locally. */
function setValue(env: ValueEnv, name: string, value: ClegValue): void {
    for (let e: ValueEnv | null = env; e; e = e.parent) {
        if (e.vars.has(name)) { e.vars.set(name, value); return; }
    }
    // Unreachable in a program that has passed typecheckCleg - see lookupValue above.
    throw new Error(`cleg: undeclared variable '${name}'`);
}

/** Deep-clones an array value's own array structure (recursively, for a nested `T[][]`) so indexed
 * mutation (`arr[i] = x;`, see evalStmt's own AssignStmt case) can never be observed through another
 * variable that was previously assigned `= arr` or received it as a function argument - this
 * language's arrays are value types, not shared references (see shared/clegBase.ts's own top
 * comment). Every other ClegValue kind either can't be mutated in place at all, or (an array ELEMENT
 * that isn't itself an array - a number/egr/sel/set/... ) can only ever be wholesale REPLACED via an
 * indexed assignment, never mutated internally - so only the array *structure* itself needs a fresh
 * copy, not every value reachable from it; called at every site a value is bound to a (potentially
 * long-lived, aliasable) variable - VarDecl's init, a whole-value AssignStmt, and a function
 * argument's own param binding - a no-op passthrough for anything that isn't (or doesn't contain) an
 * array. */
function cloneArrayValue(v: ClegValue): ClegValue {
    if (v.kind !== 'array') return v;
    return { kind: 'array', elem: v.elem, value: v.value.map(cloneArrayValue) };
}

/** Validates that `idx` is a usable index into an array of `length` elements, returning it - shared
 * by evalExpr's own IndexExpr (read) case and evalStmt's own indexed AssignStmt (write) case, so
 * both report an out-of-bounds index the same way. */
function validateArrayIndex(idx: number, length: number): number {
    if (!Number.isInteger(idx) || idx < 0 || idx >= length)
        throw new Error(`cleg: array index ${idx} out of bounds for array of length ${length}`);
    return idx;
}

/** Interleaves `suppliedArgs` (in order) into `boundArgs`' own `null` ("still uninstantiated")
 * slots, producing the full argument list the original function actually needs - used by evalExpr's
 * own CallExpr case whenever it calls through a `func` value, whether that value is a plain
 * function-pointer reference (every slot `null`, so this is just `suppliedArgs` unchanged) or a
 * partial application (see ClegValue's own 'func' doc comment) - one shared interleaving rule for
 * both, rather than treating them as two different cases. */
function fillHoles(boundArgs: (ClegValue | null)[], suppliedArgs: ClegValue[]): ClegValue[] {
    let i = 0;
    return boundArgs.map(b => (b === null ? suppliedArgs[i++] : b));
}

/** Merges a partial-application CallExpr's own `args` (at least one is a HoleExpr) into `boundArgs`'
 * own currently-open (`null`) slots, in order - evaluating each non-hole argument now (once,
 * eagerly) and leaving each hole slot open, producing the NEW boundArgs for the resulting (possibly
 * still-partial) closure. Mirrors shared/clegCheck.ts's own checkPartialApplication "same merge
 * either way" reasoning: starting from a fresh all-`null` boundArgs (a bare top-level function name,
 * nothing bound yet) or an existing value's own boundArgs (a plain pointer, still all-`null`, or an
 * already-partial closure) is the exact same operation, just a different starting point. */
function mergeBoundArgs(boundArgs: (ClegValue | null)[], args: Expr[], env: ValueEnv, funcs: UserFuncTable): (ClegValue | null)[] {
    let j = 0;
    return boundArgs.map(b => {
        if (b !== null) return b;
        const a = args[j++];
        return a.kind === 'HoleExpr' ? null : cloneArrayValue(evalExpr(a, env, funcs));
    });
}

type UserFuncTable = Record<string, FunctionDecl>;

// Thrown to unwind out of nested blocks/if-statements on `return` - always caught by
// callUserFunction below, never escapes runCleg itself.
class ReturnSignal { constructor(public value: ClegValue) {} }
// Thrown by BreakStmt/ContinueStmt to unwind out of nested blocks/if-statements up to the innermost
// enclosing ForStmt/WhileStmt's own try/catch (see evalStmt's own ForStmt/WhileStmt cases) - never
// escapes past there in a program that has passed typecheckCleg (checkStmt's own `inLoop` check
// already rejected either one outside a loop). A ReturnSignal thrown from inside a loop body is
// neither of these, so it passes straight through both catches unchanged, all the way up to
// callUserFunction, exactly as if the loop weren't there.
class BreakSignal {}
class ContinueSignal {}

function evalBlock(block: Block, parent: ValueEnv, funcs: UserFuncTable): void {
    const env: ValueEnv = { vars: new Map(), parent };
    for (const stmt of block.stmts) evalStmt(stmt, env, funcs);
}

function evalStmt(stmt: Stmt, env: ValueEnv, funcs: UserFuncTable): void {
    switch (stmt.kind) {
        case 'VarDecl':
            env.vars.set(stmt.name, cloneArrayValue(evalExpr(stmt.init, env, funcs)));
            return;
        case 'AssignStmt': {
            const value = cloneArrayValue(evalExpr(stmt.value, env, funcs));
            if (stmt.indices.length === 0) {
                setValue(env, stmt.name, value);
                return;
            }
            // Walk down into the array named `stmt.name`, following every index but the last (each
            // of which - checkStmt's own AssignStmt case already guarantees - lands on another
            // array), then mutate the final slot in place. `target` is the SAME object stored in
            // `env` (lookupValue never copies), which is safe to mutate directly precisely because
            // cloneArrayValue already guarantees nothing else aliases it (see that function's own
            // doc comment).
            let target = lookupValue(env, stmt.name) as { kind: 'array'; value: ClegValue[] };
            for (let i = 0; i < stmt.indices.length - 1; i++) {
                const idxValue = (evalExpr(stmt.indices[i], env, funcs) as { kind: 'number'; value: number }).value;
                const idx = validateArrayIndex(idxValue, target.value.length);
                target = target.value[idx] as { kind: 'array'; value: ClegValue[] };
            }
            const lastIdxValue =
                (evalExpr(stmt.indices[stmt.indices.length - 1], env, funcs) as { kind: 'number'; value: number }).value;
            const lastIdx = validateArrayIndex(lastIdxValue, target.value.length);
            target.value[lastIdx] = value;
            return;
        }
        case 'IfStmt': {
            const cond = evalExpr(stmt.cond, env, funcs) as { kind: 'bool'; value: boolean };
            if (cond.value) evalBlock(stmt.then, env, funcs);
            else if (stmt.else_) stmt.else_.kind === 'Block' ? evalBlock(stmt.else_, env, funcs) : evalStmt(stmt.else_, env, funcs);
            return;
        }
        case 'ForStmt': {
            // One scope for the whole loop (init's own variable, if any, persists across every
            // iteration) - body gets its own further-nested scope each iteration via evalBlock,
            // same as any other BLOCK - see ForStmt's own doc comment. The try/catch around
            // evalBlock is BreakStmt/ContinueStmt's own unwind target (see BreakSignal/
            // ContinueSignal's own doc comment) - `continue` still runs `update` below before the
            // next `cond` check, exactly like real C++; `break` skips straight past the loop
            // entirely, never running `update` again.
            const loopEnv: ValueEnv = { vars: new Map(), parent: env };
            if (stmt.init) evalStmt(stmt.init, loopEnv, funcs);
            while (!stmt.cond || (evalExpr(stmt.cond, loopEnv, funcs) as { kind: 'bool'; value: boolean }).value) {
                try {
                    evalBlock(stmt.body, loopEnv, funcs);
                } catch (e) {
                    if (e instanceof BreakSignal) break;
                    if (!(e instanceof ContinueSignal)) throw e;
                }
                if (stmt.update) evalStmt(stmt.update, loopEnv, funcs);
            }
            return;
        }
        case 'WhileStmt': {
            // Same BreakStmt/ContinueStmt unwind target as ForStmt above - see its own comment.
            while ((evalExpr(stmt.cond, env, funcs) as { kind: 'bool'; value: boolean }).value) {
                try {
                    evalBlock(stmt.body, env, funcs);
                } catch (e) {
                    if (e instanceof BreakSignal) break;
                    if (!(e instanceof ContinueSignal)) throw e;
                }
            }
            return;
        }
        case 'BreakStmt':
            throw new BreakSignal();
        case 'ContinueStmt':
            throw new ContinueSignal();
        case 'ReturnStmt':
            throw new ReturnSignal(evalExpr(stmt.value, env, funcs));
        case 'ExprStmt':
            evalExpr(stmt.expr, env, funcs);
            return;
        case 'Block':
            evalBlock(stmt, env, funcs);
            return;
    }
}

function evalExpr(expr: Expr, env: ValueEnv, funcs: UserFuncTable): ClegValue {
    switch (expr.kind) {
        case 'NumberLit': return { kind: 'number', value: expr.value };
        case 'StringLit': return { kind: 'string', value: expr.value };
        case 'BoolLit': return { kind: 'bool', value: expr.value };
        case 'Identifier': {
            const v = lookupValueOptional(env, expr.name);
            if (v) return v;
            // Not a variable - a bare reference to one of program's own top-level functions, used as
            // a function-pointer value (checkExpr already confirmed this resolves and is func-typed).
            const fn = funcs[expr.name];
            return {
                kind: 'func', params: fn.params.map(p => p.type), returnType: fn.returnType, name: expr.name,
                boundArgs: fn.params.map(() => null),
            };
        }
        case 'ArrayLit': {
            const values = expr.elements.map(e => evalExpr(e, env, funcs));
            // typecheckCleg already rejected an empty or mixed-element-type literal, so the first
            // value's own type is always the array's element type.
            return { kind: 'array', elem: clegValueType(values[0]), value: values };
        }
        case 'SetLit': {
            const values = expr.elements.map(e => evalExpr(e, env, funcs));
            // typecheckCleg already rejected an empty, mixed-element-type, or non-SET_ELEM_KINDS
            // literal, so the first value's own type is always the set's element type.
            return makeClegSet(clegValueType(values[0]), values);
        }
        case 'CallExpr': {
            if (expr.args.some(a => a.kind === 'HoleExpr')) {
                // Partial application - checkExpr already confirmed expr.callee names either a
                // top-level function or a local func-typed variable (never a builtin) whenever any
                // arg is '#'. mergeBoundArgs evaluates each non-hole argument now (once, eagerly),
                // exactly like an ordinary call's own arguments - cloneArrayValue matters here for
                // the same reason it does at any other site a value is bound into a (potentially
                // long-lived) slot: `boundArgs` itself, held inside the resulting closure, is exactly
                // such a slot. Starting from a variable's own boundArgs (rather than a fresh all-null
                // one) is what lets this further-apply an already-partial closure.
                const varValue = lookupValueOptional(env, expr.callee);
                const fv = varValue as { kind: 'func'; name: string; boundArgs: (ClegValue | null)[] } | undefined;
                const name = fv ? fv.name : expr.callee;
                const fn = funcs[name];
                const startingBoundArgs = fv ? fv.boundArgs : fn.params.map(() => null);
                const boundArgs = mergeBoundArgs(startingBoundArgs, expr.args, env, funcs);
                const holeParams = fn.params.filter((_, i) => boundArgs[i] === null).map(p => p.type);
                return { kind: 'func', params: holeParams, returnType: fn.returnType, name, boundArgs };
            }
            const args = expr.args.map(a => evalExpr(a, env, funcs));
            const builtin = BUILTIN_FUNCTIONS[expr.callee];
            if (builtin) return builtin.call(args, funcs);
            // A local variable of func type shadows a same-named top-level function - see checkExpr's
            // own CallExpr case, which already required this to resolve the same way. fillHoles
            // handles a plain (never-partially-applied) function value transparently, since its own
            // `boundArgs` is all `null`.
            const varValue = lookupValueOptional(env, expr.callee);
            if (varValue) {
                const fv = varValue as { kind: 'func'; name: string; boundArgs: (ClegValue | null)[] };
                return callUserFunction(funcs[fv.name], fillHoles(fv.boundArgs, args), funcs);
            }
            return callUserFunction(funcs[expr.callee], args, funcs);
        }
        case 'BinaryExpr': {
            // `&&`/`||` short-circuit here, before ever reaching BINARY_OPERATOR_OVERLOADS below -
            // see BinOp's own doc comment. checkExpr already required both operands to be bool, so
            // the short-circuited result is always just whichever side actually determines it: `&&`
            // returns false without evaluating `right` at all if `left` is already false, otherwise
            // `right`'s own value; `||` is the mirror image.
            if (expr.op === '&&' || expr.op === '||') {
                const l = (evalExpr(expr.left, env, funcs) as { kind: 'bool'; value: boolean }).value;
                if (expr.op === '&&') return l ? evalExpr(expr.right, env, funcs) : { kind: 'bool', value: false };
                return l ? { kind: 'bool', value: true } : evalExpr(expr.right, env, funcs);
            }
            const l = evalExpr(expr.left, env, funcs);
            const r = evalExpr(expr.right, env, funcs);
            for (const overload of BINARY_OPERATOR_OVERLOADS[expr.op]) {
                const m = overload.match(clegValueType(l), clegValueType(r));
                if (m) return m.eval(l, r);
            }
            // Unreachable in a program that has passed typecheckCleg.
            throw new Error(`cleg: operator '${expr.op}' has no overload for these operand types at runtime`);
        }
        case 'UnaryExpr': {
            if (expr.op === '-') {
                const v = (evalExpr(expr.operand, env, funcs) as { kind: 'number'; value: number }).value;
                return { kind: 'number', value: -v };
            }
            const v = (evalExpr(expr.operand, env, funcs) as { kind: 'bool'; value: boolean }).value;
            return { kind: 'bool', value: !v };
        }
        case 'NilExpr': return { kind: 'array', elem: expr.type, value: [] };
        case 'IndexExpr': {
            const arr = evalExpr(expr.array, env, funcs) as { kind: 'array'; elem: ClegType; value: ClegValue[] };
            const idxValue = (evalExpr(expr.index, env, funcs) as { kind: 'number'; value: number }).value;
            const idx = validateArrayIndex(idxValue, arr.value.length);
            return arr.value[idx];
        }
        // Unreachable - CallExpr's own case above always filters HoleExpr args out before recursing
        // into evalExpr for the rest; the parser never produces one anywhere else.
        case 'HoleExpr':
            throw new Error(`cleg: '#' is only valid as an argument in a partial-application call`);
    }
}

function callUserFunction(fn: FunctionDecl, args: ClegValue[], funcs: UserFuncTable): ClegValue {
    // cloneArrayValue here (not just at the call site's own VarDecl/AssignStmt) is what makes an
    // array argument a genuine value-copy rather than a reference to the caller's own array, exactly
    // like passing one to another variable - see cloneArrayValue's own doc comment.
    const env: ValueEnv = { vars: new Map(fn.params.map((p, i) => [p.name, cloneArrayValue(args[i])])), parent: null };
    try {
        evalBlock(fn.body, env, funcs);
    } catch (e) {
        if (e instanceof ReturnSignal) return e.value;
        throw e;
    }
    throw new Error(`cleg: function '${fn.name}' fell off its own end without a 'return'`);
}

/**
 * Type-checks, then runs an already-parsed `program`: every top-level TopStmt runs in turn, left to
 * right, in one scope shared across all of them (see ClegProgram's own doc comment - entirely
 * separate from any function's own env, so none of this is visible inside a function body) - not
 * just the last one, since an earlier statement can still throw before the last one is ever reached,
 * the usual "run for effect" statement-sequencing semantics. There is no `main` and no other
 * designated entry-point function; the program's own value is whatever its last top-level statement
 * (an ExprStmt - typecheckCleg already required it) evaluated to. Always re-typechecks even if the
 * caller already did (e.g. GameConfig.boardDescr may have been validated once already at edit time,
 * but could also have arrived as untrusted deserialized JSON) - cheap relative to actually
 * evaluating, and a program's `program.functions`/`program.stmts` AST could in principle have been
 * hand-built or tampered with since it was last checked.
 */
export function runClegProgram(program: ClegProgram): ClegValue {
    typecheckCleg(program);
    const funcs: UserFuncTable = {};
    for (const fn of program.functions) funcs[fn.name] = fn;

    // One env shared across every top-level TopStmt - see typecheckCleg's own matching env.
    const env: ValueEnv = { vars: new Map(), parent: null };
    let result: ClegValue | undefined;
    for (const stmt of program.stmts) {
        if (stmt.kind === 'ExprStmt') result = evalExpr(stmt.expr, env, funcs);
        else evalStmt(stmt, env, funcs);
    }
    return result!; // typecheckCleg already required the last top-level statement to be an ExprStmt
}

/** Parses `source`, then runs it via runClegProgram - see that function's own doc comment. */
export function runCleg(source: string): ClegValue {
    return runClegProgram(parseCleg(source));
}

/** Type-checks `program` (see typecheckCleg) and additionally requires its own result type to be
 * `egr` - the shape every GameConfig.boardDescr must satisfy. Throws (with a message naming the
 * actual result type) if `program` type-checks but doesn't produce an `egr`. */
export function typecheckClegAsBoard(program: ClegProgram): void {
    const t = typecheckCleg(program);
    if (t.kind !== 'egr') throw new Error(`cleg: a board description must produce an egr, got ${typeToString(t)}`);
}

/**
 * Type-checks `program` as a board description (typecheckClegAsBoard), runs it (runClegProgram),
 * and unwraps the resulting `egr`'s own BoardConfig - the one entry point every GameConfig ->
 * BoardConfig call site (src/renderer.ts, src/main.ts, server/src/onlineGameManager.ts) uses
 * instead of the old boardType/boardArgs + applyModifiers two-step.
 */
export function buildBoardFromCleg(program: ClegProgram): BoardConfig {
    typecheckClegAsBoard(program);
    return (runClegProgram(program) as { kind: 'egr'; value: BoardConfig }).value;
}
