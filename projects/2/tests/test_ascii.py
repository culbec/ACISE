"""Text we write is printable ASCII: code, notebook cell sources, config and docs."""

import json
import pathlib

import pytest

P2 = pathlib.Path(__file__).resolve().parents[1]
REPO = P2.parents[1]
SKIP = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".sh", ".ipynb", ".tex", ".bib"}


def _files() -> list[pathlib.Path]:
    roots = [*(P2 / d for d in ("src", "tests", "scripts", "config", "notebook",
                                "sandbox", "report")),
             REPO / "docs/specs/project-2", REPO / "docs/plans/projects/2"]  # fmt: skip
    out = [P2 / "pyproject.toml"]
    for r in roots:
        out += [p for p in r.rglob("*") if p.is_file() and p.suffix in SUFFIXES
                and not SKIP & set(p.parts) and "assets" not in p.relative_to(r).parts]  # fmt: skip
    return sorted(out)


def _text(p: pathlib.Path) -> str:
    if p.suffix == ".ipynb":  # cell sources only: outputs hold library and API text
        return "".join("".join(c["source"]) for c in json.loads(p.read_text())["cells"])
    return p.read_text()


@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(REPO)))
def test_ascii(path):
    for i, line in enumerate(_text(path).splitlines(), 1):
        bad = [c for c in line if ord(c) > 127]
        assert not bad, f"line {i}: {''.join(bad)!r} in {line.strip()[:80]!r}"
