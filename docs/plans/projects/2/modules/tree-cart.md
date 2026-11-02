# `tree/cart.py`

Status: todo (step 5). Spec: [implementation-backbone.md](../implementation-backbone.md). Tests: `projects/2/tests/test_cart.py`.
Reference: Breiman, Friedman, Olshen & Stone, *Classification and Regression Trees*, 1984.

## Purpose

The project's ML algorithm, written from scratch: a binary classification tree with class weights, cost-complexity pruning, probabilities, and readable explanations.

## API

```python
Criterion = Literal["gini", "entropy"]

class CART:
    def __init__(
        self,
        criterion: Criterion = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        class_weight: Literal["balanced"] | None = None,
        ccp_alpha: float = 0.0,
    ): ...

    def fit(self, X, y, sample_weight=None) -> "CART": ...
    def predict_proba(self, X) -> np.ndarray          # (n, 2): P(pass), P(fail)
    def predict(self, X, threshold: float = 0.5) -> np.ndarray   # bool, True = will fail
    def apply(self, X) -> np.ndarray                   # leaf id per row
    def cost_complexity_path(self) -> tuple[np.ndarray, np.ndarray]  # alphas, total leaf impurity
    def pruned(self, alpha: float) -> "CART"           # new tree, self unchanged
    def export_text(self, max_depth: int | None = None) -> str
    def explain(self, x) -> list[str]                  # path for one row, in raw units

    # fitted attributes
    feature_names_: list[str]
    n_features_: int
    tree_: Tree                                        # flat arrays, below
    feature_importances_: np.ndarray
    depth_: int
    n_leaves_: int

@dataclass(frozen=True)
class Tree:                     # node i; children -1 at leaves
    feature: np.ndarray         # int, -1 at leaves
    threshold: np.ndarray       # float, go left when x <= threshold
    left: np.ndarray            # int
    right: np.ndarray           # int
    value: np.ndarray           # (n_nodes, 2) weighted class totals
    n_samples: np.ndarray       # int, unweighted
    impurity: np.ndarray        # float
```

## Design decisions

- **Splits (Breiman Ch. 2).** Binary `x_j <= t`. For each node and feature: sort the node's values once (`argsort`), sweep cumulative weighted class totals, evaluate the impurity decrease at every position where the value changes, and respect `min_samples_leaf` on both sides. The threshold is the midpoint between the two neighbouring distinct values. This finds exactly the same best split as checking every threshold, in `O(n log n)` per feature per node. NumPy only; no Python loop over rows.
- **Impurity.** Gini `1 - Σ p_k²` or entropy `-Σ p_k log2 p_k` on weighted class proportions. Gain = `w_node·I(node) - w_left·I(left) - w_right·I(right)`, normalised by the total root weight so `min_impurity_decrease` means the same as in sklearn.
- **Class weights (Breiman's priors, Ch. 4).** `"balanced"` gives class *k* the weight `n / (2 · n_k)`, multiplied into `sample_weight`. All counts in `value` are weighted, so `predict_proba` is the weighted class share in the leaf.
- **Ties.** Features are scanned in column order and a new split replaces the best only if strictly better. This makes fits deterministic with no random state.
- **Growth.** Depth-first with an explicit stack (no recursion limit). A node becomes a leaf when it is pure, has `< min_samples_split` rows, reaches `max_depth`, or has no split with gain `> min_impurity_decrease`.
- **Pruning (Breiman Ch. 3, minimal cost-complexity).** Weakest-link pruning: for every internal node *t*, `g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)`. Repeatedly collapse the node with the smallest `g` to get the nested sequence of subtrees and their alphas. `R` is the weighted node impurity times the node's weight share, as in sklearn, not Breiman's misclassification cost: this gives a smoother path and lets the test compare our path with `sklearn.tree.DecisionTreeClassifier.cost_complexity_pruning_path`. The report states this deviation. `ccp_alpha > 0` in `fit` prunes right after growing.
- **Inputs.** `X` is a `pd.DataFrame` (names kept for `explain`) or a 2-D array. NaN or inf raises `ValueError`: imputation happens per fold in `training/cv.py`.
- **Importances.** Total weighted gain per feature over all splits, normalised to sum to 1 (0 vector for a single-leaf tree).
- **Explanations.** `explain(x)` returns one line per split on the path, `"fail_rate_last_10 = 0.30 > 0.25"`, then `"leaf: P(fail) = 0.62 from 41 runs"`. `export_text` prints the tree in the same wording, indented.
- **skops.** The fitted model holds only NumPy arrays, ints, floats, strings, and the `Tree` dataclass, so `skops.io.dump` works without custom hooks. Loading lists `acise_p2_cidt.tree.cart.CART` and `acise_p2_cidt.tree.cart.Tree` in `trusted=`.

## Verification

- **Agreement with sklearn.** On synthetic data with distinct values (`make_classification`, 500 rows, seeds 0–4) and on a fixed slice of the real table: same `max_depth`, `min_samples_leaf`, `criterion`, `class_weight`. Training predictions are identical and `tree_.impurity` at the root matches. Where sklearn's random feature order breaks a tie differently, the test compares total leaf impurity instead of structure, and documents why.
- **Pruning path** alphas match `cost_complexity_pruning_path` to `1e-10`.
- Hand-computed toy: 6 rows, 1 feature, known best threshold and Gini decrease.
- `min_samples_leaf` is never violated (check every leaf); depth never exceeds `max_depth`.
- `class_weight="balanced"` on a 9:1 toy moves the threshold toward the minority class.
- `explain` on a depth-2 tree returns the expected 3 lines.
- skops round-trip: `load(dump(model))` gives identical `predict_proba` on the test data.
- NaN input raises `ValueError`.
