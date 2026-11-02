from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# T1: all runs. T2: runs whose previous completed run passed (first failures).
Target = Literal["T1", "T2"]

# Fixed by data/build.py for every repository, so they are constants, not schema.
ORDER_COL: str = "created_at"
LABEL_COL: str = "failed"
PREV_LABEL_COL: str = "prev_failed"  # 1 = the previous completed run failed

# Share of the earliest rows used for EDA and feature selection only.
DEV_FRACTION: float = 0.2


# Only what the ported modules import; `available_repos` and `load` are added in
# module 4 (modules/data-load.md).
@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame  # feature columns only, chronological, RangeIndex
    y: pd.Series  # failed = 1, passed = 0
    dev_mask: np.ndarray  # True on the first DEV_FRACTION of rows
    meta: dict  # schema.json content + target, n_rows, n_failed, n_dev
    order: pd.Series  # created_at per row, for plots and the retest cut
