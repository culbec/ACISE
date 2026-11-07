import argparse
import json

import pytest
from mock_pool import SOURCES, TARGETS, write_registry

from acise_p2_cidt.collect import targets
from acise_p2_cidt.collect.targets import (
    REGISTRY,
    Target,
    find_target,
    load_targets,
    slug,
)

OK = {"repo": "o/r", "workflow_id": 1, "branch": None, "enabled": True}


def registry(tmp_path, entries):
    p = tmp_path / "targets.json"
    p.write_text(json.dumps(entries))
    return p


def test_valid_file(tmp_path):
    cfgs = load_targets(write_registry(tmp_path / "t.json"))
    assert [c.target for c in cfgs] == TARGETS
    assert cfgs[0].exclude_jobs == ("^Result$",) and cfgs[0].enabled


def test_defaults(tmp_path):
    (c,) = load_targets(registry(tmp_path, [{"repo": "o/r", "workflow_id": 1,
                                             "enabled": True}]))  # fmt: skip
    assert c.target == Target("o/r", 1, None) and c.exclude_jobs == ()


@pytest.mark.parametrize(
    "change, match",
    [
        ({"enabled": "drop"}, "missing keys \\['enabled'\\]"),
        ({"enable": True}, "unknown keys \\['enable'\\]"),  # typo
        ({"repo": "no-slash"}, "owner/name"),
        ({"repo": "o/r/x"}, "owner/name"),
        ({"workflow_id": True}, "workflow_id"),  # bool is an int subclass
        ({"workflow_id": 0}, "workflow_id"),
        ({"workflow_id": "1"}, "workflow_id"),
        ({"branch": ""}, "branch"),
        ({"enabled": "yes"}, "enabled"),
        ({"exclude_jobs": ["(unclosed"]}, "bad exclude_jobs regex"),
        ({"exclude_jobs": "^Result$"}, "list of strings"),
    ],
)
def test_validation_errors(tmp_path, change, match):
    entry = {k: v for k, v in (OK | change).items() if v != "drop"}
    with pytest.raises(ValueError, match=match) as e:
        load_targets(registry(tmp_path, [OK, entry]))
    assert "entry 1" in str(e.value)


def test_not_a_list_or_object(tmp_path):
    with pytest.raises(ValueError, match="list"):
        load_targets(registry(tmp_path, OK))
    with pytest.raises(ValueError, match="entry 0: expected an object"):
        load_targets(registry(tmp_path, ["o/r"]))


def test_duplicate_slug(tmp_path):
    with pytest.raises(ValueError, match="entry 1 .*duplicate slug o__r__1"):
        load_targets(registry(tmp_path, [OK, OK | {"enabled": False}]))
    # same workflow on another branch is another target
    assert len(load_targets(registry(tmp_path, [OK, OK | {"branch": "main"}]))) == 2


def test_enabled_only_and_find_disabled(tmp_path):
    p = write_registry(tmp_path / "t.json", enabled=False)
    assert load_targets(p) == []
    assert len(load_targets(p, enabled_only=False)) == 3
    assert find_target(SOURCES[1], p).target == TARGETS[1]
    with pytest.raises(ValueError, match="no target"):
        find_target("acme__app__99", p)


def test_repo_registry_is_valid():
    cfgs = load_targets(REGISTRY, enabled_only=False)
    assert len(cfgs) == 14 and not any(c.enabled for c in cfgs)  # phase 1: all off
    assert find_target("n8n-io__n8n__33323319@master").target.branch == "master"


@pytest.mark.parametrize(
    "t, expected",
    [
        (Target("n8n-io/n8n", 33323319, "master"), "n8n-io__n8n__33323319@master"),
        (Target("astral-sh/uv", 82881044), "astral-sh__uv__82881044"),  # all branches
        (Target("o/r", 1, "release/1.x"), "o__r__1@release~1.x"),  # '/' not in a name
    ],
)
def test_slug(t, expected):
    assert slug(t) == expected


def test_targets_from_args(tmp_path, monkeypatch):
    p = write_registry(tmp_path / "t.json")
    monkeypatch.setattr(targets, "REGISTRY", p)
    ap = argparse.ArgumentParser()
    targets.add_target_args(ap)
    pick = lambda *argv: [
        c.target for c in targets.targets_from_args(ap.parse_args(argv))
    ]
    with pytest.raises(SystemExit):  # no built-in default target
        pick()
    with pytest.raises(SystemExit):
        pick("--repo", "o/r")  # no workflow
    assert pick("--targets", str(p)) == TARGETS
    assert pick("--repo", "o/r", "--workflow", "5") == [Target("o/r", 5)]
    assert pick("--target", SOURCES[0]) == [TARGETS[0]]
