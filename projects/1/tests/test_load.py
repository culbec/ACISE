import pandas as pd
import pytest

from acise_p1_cigp.data.load import PASS_LABEL, preprocess


def frame():
    return pd.DataFrame(
        {
            "project": "p",
            "build_nums": [3, 1, 2, 5, 4],
            "now_label": [1, 1, 0, 0, 1],
            "last_label": [1, 1, 1, 0, 0],
            "src_churn": [30, 10, 20, 50, 40],
            "line_added": [3, 1, -1, 5, 4],
            "build_duration": [9, 9, 9, 9, 9],
        }
    ).sort_values("build_nums").reset_index(drop=True)


def test_t1():
    d = preprocess(frame(), "T1")
    assert list(d.X.columns) == ["last_label", "src_churn", "line_added"]
    assert list(d.X.src_churn) == [10, 20, 30, 40, 50]  # chronological
    assert list(d.y) == [0, 1, 0, 0, 1]  # failed = (now_label != PASS)
    assert d.dev_mask.sum() == 1


def test_t2():
    d = preprocess(frame(), "T2")
    assert list(d.X.columns) == ["src_churn", "line_added"]
    assert list(d.X.src_churn) == [10, 20, 30]  # last_label == PASS rows only
    assert list(d.y) == [0, 1, 0]
    assert d.meta["n_failed"] == 1 and PASS_LABEL == 1


def test_bad_target():
    with pytest.raises(ValueError):
        preprocess(frame(), "T3")


def test_line_added_sentinel_is_nan():
    d = preprocess(frame(), "T1")
    assert d.X.line_added.isna().tolist() == [False, True, False, False, False]
