# `tree/tune.py`

Status: done (step 6). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_tune.py`.

## Purpose

Connect `tree/cart.py` to the ported CV loop: choose hyperparameters, the pruning alpha, and the decision threshold on validation rows only, and give `run_cv` its `fit_predict` callable.

## API

```python
GRID: dict[str, list] = {
    "max_depth": [3, 4, 5, 6, 8],
    "min_samples_leaf": [5, 10, 20],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
}

@dataclass(frozen=True)
class Choice:
    params: dict            # grid params
    alpha: float            # pruning alpha
    threshold: float        # on P(fail)
    val_mcc: float

grid_search(split, grid: dict = GRID) -> tuple[Choice, pd.DataFrame]   # best, all rows
choose_alpha(tree: CART, split) -> float
choose_threshold(proba: np.ndarray, y: np.ndarray) -> float
fit_predict(split, params: dict | None = None) -> Result
    # params None: grid_search on split; then fit on fit rows, prune, threshold, predict test
```

## Design decisions

- **Everything is chosen on `split.X_val` / `split.y_val`**, never on test. Score: MCC (`sklearn.metrics.matthews_corrcoef`, the same function `metrics.scores` reports), as in project 1, because it stays meaningful under imbalance.
- **Order inside one grid point:** fit on the fit rows → walk the pruning path and keep the alpha with the best validation MCC (at threshold 0.5) → tune the threshold on the pruned tree's validation probabilities. Ties go to the larger alpha (smaller tree), then to the threshold nearer 0.5. Across grid points, ties go to the first point in grid order (smallest depth first).
- **Thresholds** tried: the distinct validation probabilities (at most ~leaf count, since a tree has few distinct outputs), plus 0.5. Prediction is `P(fail) > threshold`, as in `CART.predict`.
- **The final model is not refitted on fit + val.** Refitting would change the tree after alpha and threshold were chosen for it. This matches the project 1 protocol (val = the tail of train, used for choices).
- `Result.payload` is `{"choice", "tree", "grid"}` for the notebooks; the grid frame has one row per point (params, alpha, threshold, val_mcc, depth, n_leaves), and is `None` when `params` is given.
- Grid size 5 × 3 × 2 × 2 = 60 fits per fold, each well under a second on ~4k rows.
- **Tuning on the dev segment for the paper table** uses `dev_split` from `training/cv.py`, as in project 1.

## Verification

- A synthetic split where depth 2 is optimal: `grid_search` picks depth ≤ 3, and the chosen alpha prunes back a depth-8 tree.
- `choose_threshold` on hand-made probabilities picks the value that maximises MCC (hand-computed).
- `fit_predict` never reads `X_test` before predicting: a split-like object logs each `X_test` read, `CART.fit` is wrapped to log each fit, and the log must be exactly one fit, then one `X_test` read.
