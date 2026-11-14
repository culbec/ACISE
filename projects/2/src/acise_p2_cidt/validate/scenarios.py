"""
Sandbox scenarios (E4): a seeded plan of changes and the code that writes each one
into a checkout of the sandbox branch.

A scenario has a shape (diff size, tests touched, directories, workflow touched),
a breakage kind (none, a code defect caught by a test, a CI break) and an event
(push only, or push plus a pull request). Breakage is verified locally: a code
defect is kept only if the covering test file fails, and an unbroken change must
pass lint and the whole test suite.
"""

import ast
import itertools
import json
import os
import pathlib
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

SIZES = ("small", "large")
TESTS = ("src", "src+tests")
DIRS = ("one", "many")
WORKFLOW = ("nowf", "wf")
# source directories of the sandbox package; tests live in `tests/test_<module>.py`
SRC_DIRS = ("src/ledger/core", "src/ledger/text")
WORKFLOW_FILE = ".github/workflows/sandbox.yml"
CI_BREAKS = ("invalid_step", "failing_command", "missing_dependency")


@dataclass(frozen=True)
class Shape:
    size: str
    tests: str
    dirs: str
    workflow: str

    @property
    def name(self) -> str:
        return f"{self.size}-{self.tests.replace('+', '-')}-{self.dirs}-{self.workflow}"


SHAPES = [Shape(*s) for s in itertools.product(SIZES, TESTS, DIRS, WORKFLOW)]


@dataclass(frozen=True)
class Scenario:
    n: int
    phase: str  # "warmup" (not scored) or "scored"
    shape: Shape
    breakage: str  # "none", "code" or "ci"
    event: str  # "push", or "pr" (push plus a pull request)

    @property
    def branch(self) -> str:
        # not `ci-sandbox/...`: git cannot hold a branch and branches under its name
        return f"sandbox/{self.n}-{self.shape.name}"

    def to_dict(self) -> dict:
        return asdict(self) | {"shape": self.shape.name, "branch": self.branch}


def plan(seed: int = 0, warmup: int = 30, scored: int = 48,
         extra_break_rate: float = 0.25) -> list[Scenario]:  # fmt: skip
    """
    The scenario order: `warmup` unbroken changes, then `scored` scenarios in a
    seeded shuffle. Every shape appears at least once unbroken and once broken;
    the scored scenarios beyond those pairs break with `extra_break_rate`. A
    broken shape that touches the workflow breaks the CI or the code, at random;
    one that does not breaks the code.
    """
    if scored < 2 * len(SHAPES):
        raise ValueError(f"scored must be >= {2 * len(SHAPES)}: every shape both ways")
    rng = random.Random(seed)

    def kind(shape: Shape, broken: bool) -> str:
        if not broken:
            return "none"
        return rng.choice(["code", "ci"]) if shape.workflow == "wf" else "code"

    pairs = [(s, b) for s in SHAPES for b in (False, True)]
    pairs += [(rng.choice(SHAPES), rng.random() < extra_break_rate)
              for _ in range(scored - len(pairs))]  # fmt: skip
    rng.shuffle(pairs)
    out = [Scenario(i, "warmup", rng.choice(SHAPES), "none", rng.choice(["push", "pr"]))
           for i in range(warmup)]  # fmt: skip
    out += [Scenario(warmup + i, "scored", s, kind(s, b), rng.choice(["push", "pr"]))
            for i, (s, b) in enumerate(pairs)]  # fmt: skip
    return out


# =============================================================================
# Text edits by AST position
# =============================================================================


def _offset(text: str, line: int, col: int) -> int:
    starts = [0, *itertools.accumulate(len(x) for x in text.splitlines(True))]
    return starts[line - 1] + col


def _splice(text: str, node: ast.AST, new: str) -> str:
    a = _offset(text, node.lineno, node.col_offset)
    b = _offset(text, node.end_lineno, node.end_col_offset)
    return text[:a] + new + text[b:]


def _functions(tree: ast.AST) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _insert_line(text: str, rng: random.Random, line_of) -> str | None:
    """Inserts the line `line_of(indent)` before a random statement of a function body."""
    stmts = [s for f in _functions(ast.parse(text)) for s in f.body[1:]]
    if not stmts:
        return None
    s = rng.choice(stmts)
    lines = text.splitlines(True)
    indent = re.match(r"\s*", lines[s.lineno - 1])[0]
    lines.insert(s.lineno - 1, line_of(indent))
    return "".join(lines)


WORDS = (
    "cache",
    "order",
    "stock",
    "price",
    "note",
    "check",
    "total",
    "batch",
    "line",
    "unit",
)


def _comment(text, rng):
    return _insert_line(text, rng, lambda i: f"{i}# {' '.join(rng.sample(WORDS, 3))}\n")


def _blank(text, rng):
    return _insert_line(text, rng, lambda i: "\n")


def _docstring(text, rng):
    docs = [f.body[0].value for f in _functions(ast.parse(text))
            if ast.get_docstring(f) is not None]  # fmt: skip
    if not docs:
        return None
    d = rng.choice(docs)
    end = _offset(text, d.end_lineno, d.end_col_offset) - 3  # before the closing quotes
    return text[:end] + f" See {rng.choice(WORDS)}." + text[end:]


def _rename(text, rng):
    """Renames a local variable everywhere in its function."""
    tree = ast.parse(text)
    cands = []
    for f in _functions(tree):
        params = {a.arg for a in ast.walk(f.args) if isinstance(a, ast.arg)}
        stores = {n.id for n in ast.walk(f) if isinstance(n, ast.Name)
                  and isinstance(n.ctx, ast.Store)} - params  # fmt: skip
        cands += [(f, v) for v in sorted(stores)]
    if not cands:
        return None
    f, v = rng.choice(cands)
    names = [n for n in ast.walk(f) if isinstance(n, ast.Name) and n.id == v]
    new = f"{v}_{rng.choice(WORDS)}"
    for n in sorted(names, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        text = _splice(text, n, new)
    return text


HARMLESS = {
    "comment": _comment,
    "docstring": _docstring,
    "rename": _rename,
    "blank": _blank,
}


def harmless(text: str, rng: random.Random) -> tuple[str, str]:
    """One edit that changes no behaviour: comment, docstring, local rename or blank line."""
    for name in rng.sample(sorted(HARMLESS), len(HARMLESS)):
        if (new := HARMLESS[name](text, rng)) is not None:
            return new, name
    raise ValueError("no place for a harmless edit")


# =============================================================================
# Mutations (code defects)
# =============================================================================

_FLIP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
         ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In}  # fmt: skip


def mutations(text: str) -> list[tuple[str, str]]:
    """
    Every single-point defect of a module: flipped comparison, off by one, swapped
    arguments, dropped branch. Returns (description, new text) pairs.
    """
    tree = ast.parse(text)
    out = []
    for n in ast.walk(tree):
        where = f"line {getattr(n, 'lineno', '?')}"
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _FLIP:
            m = ast.Compare(n.left, [_FLIP[type(n.ops[0])]()], n.comparators)
            out.append(
                (f"flipped comparison, {where}", _splice(text, n, ast.unparse(m)))
            )
        elif (isinstance(n, ast.Constant) and type(n.value) is int
              and n.col_offset is not None):  # fmt: skip
            for d in (1, -1):
                out.append((f"off by one ({n.value}{d:+d}), {where}",
                            _splice(text, n, str(n.value + d))))  # fmt: skip
        elif (isinstance(n, ast.Call) and len(n.args) >= 2
              and not any(isinstance(a, ast.Starred) for a in n.args[:2])):  # fmt: skip
            m = ast.Call(n.func, [n.args[1], n.args[0], *n.args[2:]], n.keywords)
            out.append(
                (f"swapped arguments, {where}", _splice(text, n, ast.unparse(m)))
            )
        elif isinstance(n, ast.If):
            out.append((f"dropped branch, {where}", _splice(text, n.test, "False")))
    # a negative literal is UnaryOp(-, Constant): its off-by-one is still valid code
    return [(d, t) for d, t in out if _parses(t)]


def _parses(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


# =============================================================================
# Workflow edits
# =============================================================================


def workflow_harmless(text: str, n: int) -> str:
    """Sets a top-level `env` value: a workflow change that breaks nothing."""
    text = re.sub(r"\nenv:\n  SCENARIO: \"\d+\"\n", "\n", text)
    return text.replace(
        "\npermissions:", f'\nenv:\n  SCENARIO: "{n}"\n\npermissions:', 1
    )


def workflow_break(text: str, kind: str) -> str:
    """A CI break: an action that does not exist, a failing command, or no dev deps."""
    if kind == "invalid_step":
        new = text.replace("actions/checkout@v4", "actions/checkout@v0-missing", 1)
    elif kind == "failing_command":
        new = text.replace(
            "run: uv run ruff check", "run: uv run ruff check --select ALL", 1
        )
    elif kind == "missing_dependency":
        new = text.replace("run: uv sync", "run: uv sync --no-dev")
    else:
        raise ValueError(f"unknown CI break {kind!r}")
    if new == text:
        raise ValueError(f"workflow has no place for {kind}")
    return new


# =============================================================================
# Local checks and the change itself
# =============================================================================


def _python(root: pathlib.Path, *args: str) -> bool:
    """Runs `python -m <args>` in the sandbox checkout; True when it exits 0."""
    env = os.environ | {"PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-m", *args], cwd=root, env=env,
                       capture_output=True, text=True, check=False)  # fmt: skip
    return r.returncode == 0


def tests_pass(root: pathlib.Path, files: list[str] | None = None) -> bool:
    # the slow invariant suite only lengthens the CI run; locally it is skipped
    return _python(
        root, "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow", *(files or [])
    )


def lint_pass(root: pathlib.Path) -> bool:
    return _python(root, "ruff", "check", "--no-cache", ".")


def _modules(root: pathlib.Path, d: str) -> list[pathlib.Path]:
    return sorted(p for p in (root / d).glob("*.py") if p.name != "__init__.py")


def _test_of(root: pathlib.Path, module: pathlib.Path) -> pathlib.Path:
    return root / "tests" / f"test_{module.stem}.py"


def _helper(
    n: int, module: pathlib.Path, large: bool, rng: random.Random
) -> tuple[str, str]:
    """A new function for `module` and its test: a small one, or a lookup table."""
    fn = f"code_{n}"
    if not large:
        k = rng.randint(2, 9)
        return (f"\n\ndef {fn}(x: int) -> int:\n    return x * {k}\n",
                f"\n\ndef test_{fn}():\n    assert {fn}(3) == {3 * k}\n")  # fmt: skip
    table = {w + str(i): rng.randint(1, 999) for i, w in enumerate(WORDS * 2)}
    body = "".join(f'        "{k}": {v},\n' for k, v in table.items())
    src = (f'\n\ndef {fn}(name: str) -> int:\n    """Code of a named entry; 0 if unknown."""\n'
           f"    table = {{\n{body}    }}\n    return table.get(name, 0)\n")  # fmt: skip
    keys = rng.sample(sorted(table), 3)
    checks = "".join(f'    assert {fn}("{k}") == {table[k]}\n' for k in keys)
    return src, f'\n\ndef test_{fn}():\n{checks}    assert {fn}("missing") == 0\n'


def apply(sc: Scenario, root: pathlib.Path, seed: int = 0) -> dict:
    """
    Writes scenario `sc` into the sandbox checkout `root` and checks its breakage
    locally. Returns what was changed (files, edits, the defect).
    """
    rng = random.Random(seed * 100_003 + sc.n)
    s = sc.shape
    dirs = list(SRC_DIRS) if s.dirs == "many" else [rng.choice(SRC_DIRS)]
    large = s.size == "large"
    edits, files = [], set()
    # one module per directory; large changes add a lookup-table function and
    # several harmless edits, small ones a single edit
    mods = [rng.choice(_modules(root, d)) for d in dirs]
    defect = None
    if sc.breakage == "code":
        defect = _mutate(rng.choice(mods), root, rng)
        files.add(defect["file"])
    for m in mods:
        text = m.read_text()
        # a small defect is the whole change in its own module
        hit = defect is not None and str(m.relative_to(root)) == defect["file"]
        n_edits = rng.randint(3, 5) if large else (0 if hit else 1)
        for _ in range(n_edits):
            text, kind = harmless(text, rng)
            edits.append(f"{kind} in {m.relative_to(root)}")
        if large or s.tests == "src+tests":
            src, test = _helper(sc.n, m, large, rng)
            text += src
            if s.tests == "src+tests":
                t = _test_of(root, m)
                t.write_text(t.read_text().replace(
                    f"from ledger.{m.parent.name}.{m.stem} import ",
                    f"from ledger.{m.parent.name}.{m.stem} import code_{sc.n}, ", 1) + test)  # fmt: skip
                files.add(str(t.relative_to(root)))
            edits.append(f"new function code_{sc.n} in {m.relative_to(root)}")
        if text != m.read_text():
            m.write_text(text)
            files.add(str(m.relative_to(root)))
    ci_break = None
    if s.workflow == "wf":
        wf = root / WORKFLOW_FILE
        text = workflow_harmless(wf.read_text(), sc.n)
        if sc.breakage == "ci":
            ci_break = rng.choice(CI_BREAKS)
            text = workflow_break(text, ci_break)
        wf.write_text(text)
        files.add(WORKFLOW_FILE)
    # the code must be fine unless the scenario says otherwise
    if sc.breakage != "code" and not (lint_pass(root) and tests_pass(root)):
        raise RuntimeError(f"scenario {sc.n}: unbroken code fails lint or tests")
    if sc.breakage == "code" and not lint_pass(root):
        raise RuntimeError(f"scenario {sc.n}: the defect also breaks lint")
    return {
        "files": sorted(files),
        "edits": edits,
        "defect": defect,
        "ci_break": ci_break,
    }


def _mutate(module: pathlib.Path, root: pathlib.Path, rng: random.Random) -> dict:
    """Applies a random mutation of `module` that its test file catches."""
    text = module.read_text()
    test = _test_of(root, module)
    # operator first, then a site: literals would otherwise make most defects off by one
    by_op: dict[str, list] = {}
    for desc, new in mutations(text):
        by_op.setdefault(desc.split(" (")[0].split(",")[0], []).append((desc, new))
    cands = []
    for op in rng.sample(sorted(by_op), len(by_op)):
        cands += rng.sample(by_op[op], len(by_op[op]))
    for desc, new in cands:
        module.write_text(new)
        if not tests_pass(root, [str(test.relative_to(root))]):
            return {"file": str(module.relative_to(root)), "mutation": desc,
                    "test": str(test.relative_to(root))}  # fmt: skip
    module.write_text(text)
    raise RuntimeError(f"no mutation of {module.name} is caught by {test.name}")


def save_plan(path: pathlib.Path, seed: int, scenarios: list[Scenario]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "scenarios": [s.to_dict() for s in scenarios]},
                               indent=1))  # fmt: skip
