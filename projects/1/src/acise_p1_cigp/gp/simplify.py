import inspect
import operator
import re

import numpy as np
import pandas as pd
from deap import gp
from sklearn.pipeline import Pipeline

from acise_p1_cigp.gp.engine import statements

CMP = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
NEG = {"lt": "ge", "ge": "lt", "le": "gt", "gt": "le"}
FLIP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}
SYMBOL = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}
TIGHTER = {"ge": max, "gt": max, "lt": min, "le": min}
LOOSER = {"ge": min, "gt": min, "lt": max, "le": max}


def _nest(tree):
    """
    Prefix node list to nested tuples `(name, *children)`; leaves are feature
    names (str), bool constants or float thresholds.

    Args:
        tree: DEAP tree.

    Returns:
        tuple | str | bool | float: nested expression.
    """
    nodes = iter(tree)

    def walk():
        node = next(nodes)
        if isinstance(node, gp.Primitive):
            return (node.name, *(walk() for _ in range(node.arity)))
        return node.value

    return walk()


def _fmt(e) -> str:
    """
    Nested expression to a DEAP expression string.

    Args:
        e: nested expression.

    Returns:
        str: string `gp.PrimitiveTree.from_string` parses.
    """
    if isinstance(e, tuple):
        return f"{e[0]}({', '.join(_fmt(a) for a in e[1:])})"
    return repr(e) if isinstance(e, float) else str(e)


def _merge(op: str, x, y, pick: dict):
    """
    Merges two comparisons on the same feature with the same operator.

    Args:
        op (str): "and_" or "or_" (only for the result's sake; `pick` decides).
        x: left child.
        y: right child.
        pick (dict): operator -> max or min, the threshold kept.

    Returns:
        tuple | None: merged comparison, or None when the children do not match.
    """
    if (
        isinstance(x, tuple)
        and isinstance(y, tuple)
        and x[0] == y[0]
        and x[0] in CMP
        and x[1] == y[1]
        and isinstance(x[1], str)
        and isinstance(x[2], float)
        and isinstance(y[2], float)
    ):
        return (x[0], x[1], pick[x[0]](x[2], y[2]))
    return None


def _node(e):
    """
    One rewrite at the root of `e`, children already rewritten. Inputs never hold
    NaN (imputed), so negating a comparison is exact.

    Args:
        e: nested expression.

    Returns:
        nested expression, equivalent to `e` on every input.
    """
    if not isinstance(e, tuple):
        return e
    op, *args = e
    if op in CMP:
        a, b = args
        if isinstance(a, float) and isinstance(b, float):
            return CMP[op](a, b)
        if a == b:
            return CMP[op](0.0, 0.0)
        if isinstance(a, float):
            return (FLIP[op], b, a)
        return e
    if op == "not_":
        (x,) = args
        if isinstance(x, bool):
            return not x
        if isinstance(x, tuple) and x[0] == "not_":
            return x[1]
        if isinstance(x, tuple) and x[0] in NEG:
            return (NEG[x[0]], *x[1:])
        return e
    if op == "if_then_else":
        c, a, b = args
        if isinstance(c, bool):
            return a if c else b
        if a == b:
            return a
        if a is True and b is False:
            return c
        if a is False and b is True:
            return ("not_", c)
        return e
    x, y = args
    absorb = op == "or_"
    if x is absorb or y is absorb:
        return absorb
    if x is (not absorb):
        return y
    if y is (not absorb):
        return x
    if x == y:
        return x
    return _merge(op, x, y, LOOSER if absorb else TIGHTER) or e


def _rewrite(e):
    """
    Applies `_node` bottom-up until nothing changes.

    Args:
        e: nested expression.

    Returns:
        simplified nested expression.
    """
    while True:
        new = _node((e[0], *(_rewrite(a) for a in e[1:]))) if isinstance(e, tuple) else e
        if new == e and type(new) is type(e):
            return new
        e = new


def simplify(tree, pset: gp.PrimitiveSetTyped) -> gp.PrimitiveTree:
    """
    Semantics-preserving rewrites: dead branches, constant conditions, double
    negation, `not_` folded into comparisons, constants moved to the right of
    comparisons, and same-feature comparisons merged.

    Args:
        tree: DEAP tree.
        pset (gp.PrimitiveSetTyped): primitive set the tree was built with.

    Returns:
        gp.PrimitiveTree: simplified tree, same predictions on every input.
    """
    return gp.PrimitiveTree.from_string(_fmt(_rewrite(_nest(tree))), pset)


def pct(x, q):
    """
    Percentile of `x` among the fit rows, exactly as the fold's uniform
    `QuantileTransformer` maps one column. Also printed into `to_python` output.

    Args:
        x: raw value.
        q: the column's `quantiles_` (fit-row knots, ascending).

    Returns:
        float: value in [0, 1].
    """
    q = np.asarray(q, dtype=float)
    if x <= q[0]:
        return 0.0
    if x >= q[-1]:
        return 1.0
    r = np.linspace(0, 1, len(q))
    return float(0.5 * (np.interp(x, q, r) - np.interp(-x, -q[::-1], -r[::-1])))


def _raw_threshold(op: str, t: float, q: np.ndarray):
    """
    `pct(x, q) op t` as a condition on raw `x`. `pct` jumps at tied knots and is
    linear in between, so the boundary is a knot or a linear inverse inside the
    first knot interval where the condition holds.

    Args:
        op (str): lt, le, gt or ge.
        t (float): scaled threshold.
        q (np.ndarray): the column's `quantiles_`.

    Returns:
        bool | tuple[str, float]: constant, or `(op, raw threshold)` on `x`.
    """
    if op in ("lt", "le"):  # negation of ge / gt
        out = _raw_threshold(NEG[op], t, q)
        return (not out) if isinstance(out, bool) else (NEG[out[0]], out[1])
    u = np.unique(q)
    k = next((i for i, c in enumerate(u) if CMP[op](pct(c, q), t)), None)
    if k is None:
        return False
    if k == 0:  # pct is 0 up to the first knot, so the condition holds everywhere
        return True
    lo, hi = u[k - 1], u[k]
    f1, f2 = pct(lo + (hi - lo) / 3, q), pct(lo + 2 * (hi - lo) / 3, q)
    left, right = 2 * f1 - f2, 2 * f2 - f1  # one-sided limits of the linear piece
    if left >= t:  # interior values are all above left
        return ("gt", float(lo))
    if t >= right:
        return ("ge", float(hi))
    return (op, float(lo + (t - left) / (right - left) * (hi - lo)))


def _cond(e, prep: Pipeline) -> str:
    """
    Nested expression to a Python expression over raw feature values. A threshold
    becomes a raw-unit threshold; a feature-vs-feature comparison compares the two
    within-fit percentiles (`pct`, quantile knots in `Q`).

    Args:
        e: simplified nested expression.
        prep (Pipeline): fold pipeline; its last step is the `QuantileTransformer`.

    Returns:
        str: Python expression on `build`.
    """
    if isinstance(e, bool):
        return str(e)
    op, *args = e
    if op == "not_":
        return f"not {_cond(args[0], prep)}"
    if op in ("and_", "or_"):
        return f"({_cond(args[0], prep)} {op[:-1]} {_cond(args[1], prep)})"
    if op == "if_then_else":
        c, a, b = (_cond(x, prep) for x in args)
        return f"({a} if {c} else {b})"
    qt = prep[-1]
    col = {str(f): i for i, f in enumerate(qt.feature_names_in_)}
    a, b = args
    if isinstance(b, float):
        out = _raw_threshold(op, b, qt.quantiles_[:, col[a]])
        if isinstance(out, bool):
            return str(out)
        return f'build["{a}"] {SYMBOL[out[0]]} {out[1]:.6g}'
    return f'pct(build["{a}"], Q["{a}"]) {SYMBOL[op]} pct(build["{b}"], Q["{b}"])'


def _block(e, prep: Pipeline, depth: int) -> list[str]:
    """
    Nested expression to statement lines: `if_then_else` becomes if/elif/else,
    anything else a `return`.

    Args:
        e: simplified nested expression.
        prep (Pipeline): fold pipeline.
        depth (int): indentation level.

    Returns:
        list[str]: source lines.
    """
    pad = "    " * depth
    if not (isinstance(e, tuple) and e[0] == "if_then_else"):
        return [f"{pad}return {_cond(e, prep)}"]
    _, c, a, b = e
    lines = [f"{pad}if {_cond(c, prep)}:", *_block(a, prep, depth + 1)]
    while isinstance(b, tuple) and b[0] == "if_then_else":
        _, c, a, b = b
        lines += [f"{pad}elif {_cond(c, prep)}:", *_block(a, prep, depth + 1)]
    return lines + [f"{pad}else:", *_block(b, prep, depth + 1)]


def to_python(
    tree, pset: gp.PrimitiveSetTyped, prep: Pipeline, name: str = "predict"
) -> str:
    """
    Simplified program as Python source over raw units, thresholds mapped back with
    the fold's scaler and printed with 6 significant digits. Feature-vs-feature
    comparisons have no raw-unit form; they come with `pct` and the knots `Q`.

    Args:
        tree: DEAP tree.
        pset (gp.PrimitiveSetTyped): primitive set the tree was built with.
        prep (Pipeline): fold pipeline (`Split.prep`).
        name (str): function name.

    Returns:
        str: `def <name>(build) -> bool`, True = predict failure.
    """
    t = simplify(tree, pset)
    head = [f"def {name}(build) -> bool:", f'    """{statements(t)} statements."""']
    src = "\n".join(head + _block(_rewrite(_nest(t)), prep, 1)) + "\n"
    used = dict.fromkeys(re.findall(r'Q\["([^"]+)"\]', src))
    if not used:
        return src
    qt = prep[-1]
    col = {str(f): i for i, f in enumerate(qt.feature_names_in_)}
    knots = {f: qt.quantiles_[:, col[f]].tolist() for f in used}
    return f"import numpy as np\n\nQ = {knots!r}\n\n\n{inspect.getsource(pct)}\n\n{src}"


def _comparisons(e):
    """
    Yields every feature-vs-threshold comparison in `e`.

    Args:
        e: simplified nested expression.

    Yields:
        tuple[str, str, float]: feature, operator, scaled threshold.
    """
    if not isinstance(e, tuple):
        return
    if e[0] in CMP and isinstance(e[1], str) and isinstance(e[2], float):
        yield e[1], e[0], e[2]
    for a in e[1:]:
        yield from _comparisons(a)


def motifs(trees) -> pd.DataFrame:
    """
    Recurring (feature, operator) pairs across programs, after simplification, so
    the feature is always on the left.

    Args:
        trees: DEAP trees, e.g. a hall of fame.

    Returns:
        pd.DataFrame: feature, op, count, thr_min, thr_median, thr_max (scaled:
            within-fit percentiles),
            sorted by count descending.
    """
    rows = [
        {"feature": f, "op": op, "thr": thr}
        for t in trees
        for f, op, thr in _comparisons(_rewrite(_nest(t)))
    ]
    df = pd.DataFrame(rows, columns=["feature", "op", "thr"])
    return (
        df.groupby(["feature", "op"])["thr"]
        .agg(count="size", thr_min="min", thr_median="median", thr_max="max")
        .reset_index()
        .sort_values("count", ascending=False, ignore_index=True)
    )
