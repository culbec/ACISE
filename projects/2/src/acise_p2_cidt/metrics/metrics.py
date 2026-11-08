from typing import NamedTuple

import numpy as np
from sklearn import metrics as skm

from acise_p2_cidt.data.load import PREV_LABEL_COL


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


def mcc(y_true, y_pred) -> float:
    """
    Matthews correlation of the fail class, from the confusion counts. 0 when a
    margin is empty (one class only), as in sklearn, but without its warning.
    """
    tp, fp, tn, fn = confusion(y_true, y_pred)
    den = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return _div(tp * tn - fp * fn, den)


def scores(y_true, y_pred) -> dict[str, float]:
    """
    Reported classification scores on the fail class (sklearn), plus the raw counts.

    Accuracy is reported but misleads under class imbalance: on n8n master (~15%
    failures) an all-pass predictor already scores ~0.85 while catching no failure.

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
        "mcc": mcc(y_true, y_pred),
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
        "pct_failures_caught": 100 * _div(c.tp, c.tp + c.fn),
    }


def parrot(split) -> np.ndarray:
    """
    Predict the previous outcome. On T2 ``prev_failed`` is dropped (always pass), so
    this predicts all-pass.
    """
    raw = split.raw_test
    if PREV_LABEL_COL not in raw:
        return np.zeros(len(raw), dtype=bool)
    # `== 1`, not astype(bool): a NaN (no earlier run) must predict pass, not fail.
    return (raw[PREV_LABEL_COL] == 1).to_numpy()


def majority(split) -> np.ndarray:
    """
    Constant majority class of the fit rows; a tie predicts pass.
    """
    return np.full(len(split.X_test), np.mean(split.y_fit) > 0.5)
