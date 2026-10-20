import logging

from rich.console import Console
from rich.logging import RichHandler

ROOT = "acise_p1_cigp"

log = Console()

_root = logging.getLogger(ROOT)
if not _root.handlers:
    _root.addHandler(
        RichHandler(console=log, show_path=False, rich_tracebacks=True, markup=False)
    )
    _root.setLevel(logging.INFO)
    _root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Logger under the package logger, so it shares its rich handler and level.

    Args:
        name (str): module name, usually `__name__`.

    Returns:
        logging.Logger: `name` itself if it is under the package, else a child of it.
    """
    return logging.getLogger(name if name.startswith(ROOT) else f"{ROOT}.{name}")


def set_level(level: int | str) -> None:
    """
    Sets the level of every package logger, e.g. `set_level("DEBUG")` in a notebook.

    Args:
        level (int | str): logging level.
    """
    _root.setLevel(level)
