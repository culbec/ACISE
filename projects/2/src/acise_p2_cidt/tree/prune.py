from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from acise_p2_cidt.tree.cart import Tree


def collapse(tree: Tree, cut: set[int]) -> Tree:
    """Copy of `tree` with every node in `cut` turned into a leaf, renumbered."""
    keep: list[int] = []
    stack = [0]
    while stack:
        i = stack.pop()
        keep.append(i)
        if tree.left[i] >= 0 and i not in cut:
            stack += [tree.right[i], tree.left[i]]
    old = np.array(keep)
    new_id = np.full(len(tree.feature), -1)
    new_id[old] = np.arange(len(old))
    leaf = (tree.left[old] < 0) | np.isin(old, list(cut))
    return replace(
        tree,
        feature=np.where(leaf, -1, tree.feature[old]),
        threshold=np.where(leaf, np.nan, tree.threshold[old]),
        left=np.where(leaf, -1, new_id[tree.left[old]]),
        right=np.where(leaf, -1, new_id[tree.right[old]]),
        value=tree.value[old].copy(),
        n_samples=tree.n_samples[old].copy(),
        impurity=tree.impurity[old].copy(),
    )


def pruning_steps(tree: Tree) -> tuple[list[float], list[float], list[int]]:
    """
    Weakest-link pruning sequence (Breiman Ch. 3, minimal cost-complexity).

    `R(t)` is the node impurity times its weight share of the root, as in sklearn,
    not Breiman's misclassification cost. Each step collapses the internal node with
    the smallest `g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)`; ties go to the
    lowest node id.

    Returns:
        tuple[list[float], list[float], list[int]]: alpha of each step (first 0),
            total leaf R after it, and the node collapsed (-1 for the first entry).
    """
    n = len(tree.feature)
    w = tree.value.sum(axis=1)
    r_node = tree.impurity * w / w[0]
    parent = np.full(n, -1)
    inner = tree.left >= 0
    parent[tree.left[inner]] = np.flatnonzero(inner)
    parent[tree.right[inner]] = np.flatnonzero(inner)

    r_branch, n_leaves = r_node.copy(), np.ones(n, dtype=int)
    for i in range(n - 1, -1, -1):  # children before parents (reverse pre-order)
        if tree.left[i] >= 0:
            r_branch[i] = r_branch[tree.left[i]] + r_branch[tree.right[i]]
            n_leaves[i] = n_leaves[tree.left[i]] + n_leaves[tree.right[i]]

    alphas, totals, nodes = [0.0], [float(r_branch[0])], [-1]
    candidate = inner.copy()
    while candidate[0]:
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(candidate, (r_node - r_branch) / (n_leaves - 1), np.inf)
        t = int(g.argmin())
        r_diff, leaves_lost = r_node[t] - r_branch[t], n_leaves[t] - 1
        stack = [t]
        while stack:  # the collapsed subtree's nodes stop being candidates
            i = stack.pop()
            candidate[i] = False
            if tree.left[i] >= 0:
                stack += [tree.left[i], tree.right[i]]
        r_branch[t], n_leaves[t] = r_node[t], 1
        a = parent[t]
        while a >= 0:
            r_branch[a] += r_diff
            n_leaves[a] -= leaves_lost
            a = parent[a]
        # A split never raises weighted impurity, so g >= 0; summing r_branch can
        # still round it to a tiny negative (e.g. -5e-20), which `pruned` rejects.
        alphas.append(max(float(g[t]), 0.0))
        totals.append(float(r_branch[0]))
        nodes.append(t)
    return alphas, totals, nodes
