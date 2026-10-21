import ast
import inspect
import textwrap
from collections.abc import Callable

import numpy as np
import pandas as pd

from acise_p1_cigp.metrics.metrics import confusion, mcc
from acise_p1_cigp.training.cv import Result, Split

Program = Callable[[dict, dict], bool]
Thresholds = dict[str, tuple[str, float | None]]

QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)


def grid(col: pd.Series) -> np.ndarray:
    """
    Candidate thresholds for one feature: its distinct 5%..95% quantiles, taken as
    observed values (no interpolation), so count features get whole numbers.

    Args:
        col (pd.Series): raw values on the rows being tuned on.

    Returns:
        np.ndarray: ascending distinct values.
    """

    return np.unique(col.quantile(QUANTILES, interpolation="lower").to_numpy())


def run(predict: Program, rows: list[dict], t: dict) -> np.ndarray:
    """
    Evaluate the program on every row.

    Args:
        predict (Program): program (b).
        rows (list[dict]): raw builds.
        t (dict): threshold values.

    Returns:
        np.ndarray: bool per row, True = will fail.
    """
    return np.fromiter((bool(predict(r, t)) for r in rows), bool, len(rows))


def _mcc(predict: Program, rows: list[dict], y, t: dict) -> float:
    return mcc(confusion(y, run(predict, rows, t)))


def tune(predict: Program, thresholds: Thresholds, X: pd.DataFrame, y, passes: int = 3) -> tuple[dict, float]:
    """
    Coordinate descent on MCC: sweep one threshold over its grid with the others
    fixed, keep strict improvements, stop after `passes` or a pass with no change.

    Args:
        predict (Program): program (b).
        thresholds (Thresholds): name -> (feature, default).
        X (pd.DataFrame): raw, imputed rows to tune on.
        y: labels, 1 = failed.
        passes (int): maximum number of sweeps.

    Returns:
        tuple[dict, float]: tuned values and their MCC on `X`.
    """
    rows = X.to_dict("records")
    grids = {name: grid(X[feat]) for name, (feat, _) in thresholds.items()}
    t = {name: float(np.median(grids[name])) if d is None else d for name, (_, d) in thresholds.items()}
    best = _mcc(predict, rows, y, t)
    for _ in range(passes):
        changed = False
        for name in thresholds:
            for v in grids[name]:
                s = _mcc(predict, rows, y, t | {name: float(v)})
                if s > best:
                    best, t, changed = s, t | {name: float(v)}, True
        if not changed:
            break
    return t, best


def _impute(split: Split, raw: pd.DataFrame) -> pd.DataFrame:
    return split.prep[0].transform(raw[split.X_fit.columns])


def fit_predict(split: Split, predict: Program, thresholds: Thresholds) -> Result:
    """
    `run_cv` adapter: `run_cv(..., partial(fit_predict, predict=..., thresholds=...),
    "flow_control")`. Tunes on the fit rows; keeps the tuned values or the dev
    defaults, whichever scores higher on val (tuned wins ties), as the GP picks its
    final program on val.

    Args:
        split (Split): one fold, with `raw_fit` and `raw_val`.
        predict (Program): program (b).
        thresholds (Thresholds): name -> (feature, default).

    Returns:
        Result: test predictions; payload `{"thresholds", "source", "fit_mcc", "val_mcc"}`.
    """
    if split.raw_fit is None or split.raw_val is None:
        raise ValueError("split has no raw rows; build it with training.cv.prepare")
    X_fit, X_val = _impute(split, split.raw_fit), _impute(split, split.raw_val)
    tuned, fit_mcc = tune(predict, thresholds, X_fit, split.y_fit)

    val_rows = X_val.to_dict("records")
    cands = {"tuned": tuned}
    if all(d is not None for _, d in thresholds.values()):
        cands["default"] = {name: d for name, (_, d) in thresholds.items()}
    val = {k: _mcc(predict, val_rows, split.y_val, t) for k, t in cands.items()}
    source = max(val, key=lambda k: val[k])
    t = cands[source]
    if source != "tuned":
        fit_mcc = _mcc(predict, X_fit.to_dict("records"), split.y_fit, t)

    test_rows = _impute(split, split.raw_test).to_dict("records")
    payload = {"thresholds": t, "source": source, "fit_mcc": fit_mcc, "val_mcc": val[source]}
    return Result(run(predict, test_rows, t), payload)


def count_statements(fn: Callable) -> dict[str, int]:
    """
    Statement count of a program, from its source.

    Args:
        fn (Callable): program (b).

    Returns:
        dict[str, int]: `if` = if/elif (and conditional expression) count, the
            assignment's rule; `nodes` = the engine's count (if + comparisons +
            and/or/not), comparable with GP program sizes.
    """
    n_if = n_cmp = n_bool = 0
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
        if isinstance(node, ast.If | ast.IfExp):
            n_if += 1
        elif isinstance(node, ast.Compare):
            n_cmp += len(node.ops)
        elif isinstance(node, ast.BoolOp):
            n_bool += len(node.values) - 1
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            n_bool += 1
    return {"if": n_if, "nodes": n_if + n_cmp + n_bool}
