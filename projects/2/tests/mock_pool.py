"""
Synthetic data with the shapes of the real pipeline (backbone §11.1). Each unit
extends it; so far it holds the mock targets and their registry.
"""

import json
import pathlib

from acise_p2_cidt.collect.targets import Target, slug

# two workflows of one repo (one restricted to a branch) + one other repo
TARGETS: list[Target] = [
    Target("acme/app", 11, None),
    Target("acme/app", 12, "main"),
    Target("other/lib", 21, None),
]
SOURCES: list[str] = [slug(t) for t in TARGETS]


def write_registry(path: pathlib.Path, enabled: bool = True) -> pathlib.Path:
    """`targets.json` holding the mock targets."""
    entries = [
        {"repo": t.repo, "workflow_id": t.workflow_id, "branch": t.branch,
         "enabled": enabled, "exclude_jobs": ["^Result$"]}
        for t in TARGETS
    ]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))
    return path
