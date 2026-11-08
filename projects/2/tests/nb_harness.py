"""
Runs a notebook's code cells in one namespace, without a kernel, and checks that
every displayed object carries a title.
"""

import ast
import pathlib
from collections.abc import Callable

import IPython.display as ipd
import nbformat
import plotly.graph_objects as go

from acise_p2_cidt.viz import plots

TEXT = (str, ipd.Markdown, ipd.HTML)


def check_figure(fig: go.Figure) -> go.Figure:
    """A figure must say what it shows."""
    if not fig.layout.title.text:
        raise AssertionError("figure without a title")
    return fig


def check_display(*objs, **_) -> None:
    """
    Stands in for `display`. Figures need their own title; any other non-text
    object needs a caption (str or Markdown) earlier in the same call.
    """
    captioned = False
    for o in objs:
        if isinstance(o, TEXT):
            captioned = captioned or bool(str(getattr(o, "data", o)).strip())
        elif isinstance(o, go.Figure):
            check_figure(o)
        elif o is not None and not captioned:
            raise AssertionError(f"{type(o).__name__} displayed without a caption")


def _run_cell(source: str, name: str, ns: dict) -> None:
    """Runs a cell; a value left by its last expression is checked as Jupyter would show it."""
    tree = ast.parse(source)
    last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    if last is not None:
        tree.body.pop()
    exec(compile(tree, name, "exec"), ns)  # noqa: S102
    if last is not None:
        value = eval(compile(ast.Expression(last.value), name, "eval"), ns)
        if not source.rstrip().endswith(";"):
            check_display(value)


def run_notebook(
    path: pathlib.Path, stubs: dict[str, Callable], until: int | None = None
) -> dict:
    """
    Executes every code cell in order. Before each cell tagged `live`, `stubs`
    replace the driver names it calls, so no cell reaches the network. `display`
    and `plots.save` are replaced by checks that write and render nothing.

    Args:
        path (pathlib.Path): the notebook.
        stubs (dict[str, Callable]): namespace name -> stub, e.g. {"collect": ...}.
        until (int | None): last cell index to run; None = all. Sections not yet
            edited for multi-repo follow it.

    Returns:
        dict: the namespace after the last cell.
    """
    nb = nbformat.read(path, as_version=4)
    ns: dict = {"display": check_display}
    saved = ipd.display, plots.save
    ipd.display, plots.save = check_display, lambda fig, name, **k: check_figure(fig)
    try:
        for i, cell in enumerate(nb.cells[: None if until is None else until + 1]):
            if cell.cell_type != "code":
                continue
            if "live" in cell.metadata.get("tags", []):
                ns.update(stubs)
            try:
                _run_cell(cell.source, f"{path.name}[{i}]", ns)
            except Exception as e:
                first = cell.source.splitlines()[0] if cell.source else ""
                raise AssertionError(f"{path.name} cell {i} ({first!r}): {e!r}") from e
    finally:
        ipd.display, plots.save = saved
    return ns
