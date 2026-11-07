"""
Runs a notebook's code cells in one namespace, without a kernel.
"""

import pathlib
from collections.abc import Callable

import nbformat


def run_notebook(
    path: pathlib.Path, stubs: dict[str, Callable], until: int | None = None
) -> dict:
    """
    Executes every code cell in order. Before each cell tagged `live`, `stubs`
    replace the driver names it calls, so no cell reaches the network.

    Args:
        path (pathlib.Path): the notebook.
        stubs (dict[str, Callable]): namespace name -> stub, e.g. {"collect": ...}.
        until (int | None): last cell index to run; None = all. Sections not yet
            edited for multi-repo follow it.

    Returns:
        dict: the namespace after the last cell.
    """
    nb = nbformat.read(path, as_version=4)
    ns: dict = {"display": lambda *a, **k: None}
    for i, cell in enumerate(nb.cells[: None if until is None else until + 1]):
        if cell.cell_type != "code":
            continue
        if "live" in cell.metadata.get("tags", []):
            ns.update(stubs)
        try:
            exec(compile(cell.source, f"{path.name}[{i}]", "exec"), ns)  # noqa: S102
        except Exception as e:
            first = cell.source.splitlines()[0] if cell.source else ""
            raise AssertionError(f"{path.name} cell {i} ({first!r}): {e!r}") from e
    return ns
