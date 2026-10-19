import math
from typing import NamedTuple

import numpy as np
from sklearn import metrics as skm

from acise_p1_cigp.data.load import PASS_LABEL, PREV_LABEL_COL


class Counts(NamedTuple):
    tp: int
    fp: int
    tn: int
    fn: int


def confusion(y_true, y_pred) -> Counts:
    """
    Confusion counts for the fail class.

    Args:
        y_true: labels, 1/True = failed.
        y_pred: predictions, 1/True = will fail.

    Returns:
        Counts: tp, fp, tn, fn.
    """
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {t.shape}, y_pred {p.shape}")
    tp = int(np.count_nonzero(t & p))
    fp = int(np.count_nonzero(~t & p))
    fn = int(np.count_nonzero(t & ~p))
    return Counts(tp, fp, t.size - tp - fp - fn, fn)


def _div(num: float, den: float) -> float:
    return num / den if den else 0.0


def rates(c: Counts) -> tuple[float, float]:
    """
    (TPR, FPR), the NSGA-II objective pair (maximise TPR, minimise FPR).
    """
    return _div(c.tp, c.tp + c.fn), _div(c.fp, c.fp + c.tn)


def mcc(c: Counts) -> float:
    """
    Matthews correlation coefficient from counts (GP fitness); 0.0 when a margin is empty.
    """
    den = math.sqrt((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn))
    return _div(c.tp * c.tn - c.fp * c.fn, den)


def scores(y_true, y_pred) -> dict[str, float]:
    """
    Reported classification scores on the fail class (sklearn), plus the raw counts.

    Accuracy is reported but misleads under class imbalance: on T1 atlasdb an
    all-pass predictor already scores ~0.68 while catching no failure.

    Returns:
        dict: precision, recall, f1, mcc, accuracy, tpr, fpr, tp, fp, tn, fn.
    """
    c = confusion(y_true, y_pred)
    # sklearn has no type hints; pyright infers `zero_division: str` from its "warn" default
    precision, recall, f1, _ = skm.precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(skm.matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "tpr": float(recall),
        "fpr": _div(c.fp, c.fp + c.tn),  # sklearn has no FPR scorer
        **c._asdict(),
    }


def domain(y_true, y_pred) -> dict[str, float]:
    """
    CI cost view, in percent.

    Returns:
        dict: ``pct_builds_saved`` (share of builds predicted pass, so skippable) and
            ``pct_failures_caught`` (recall on the fail class).
    """
    c = confusion(y_true, y_pred)
    return {
        "pct_builds_saved": 100 * _div(c.tn + c.fn, sum(c)),
        "pct_failures_caught": 100 * rates(c)[0],
    }


def parrot(split) -> np.ndarray:
    """
    Predict the previous outcome. On T2 ``last_label`` is dropped (always pass), so
    this predicts all-pass.
    """
    raw = split.raw_test
    if PREV_LABEL_COL not in raw:
        return np.zeros(len(raw), dtype=bool)
    return (raw[PREV_LABEL_COL] != PASS_LABEL).to_numpy()


def majority(split) -> np.ndarray:
    """
    Constant majority class of the fit rows; a tie predicts pass.
    """
    return np.full(len(split.X_test), np.mean(split.y_fit) > 0.5)
