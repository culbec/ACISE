"""
Tables, numbers and figures of the report, computed from the stored results only.

Writes `report/assets/`: `tables/*.tex` (booktabs), `numbers.tex` (one
`\\newcommand` per number the text cites) and `figures/*.png` (copies). Nothing in
the report's prose is typed by hand. Run from `projects/2` after the notebooks:

    uv run python scripts/report_assets.py
"""

import json
import pathlib
import shutil

import numpy as np
import pandas as pd

from acise_p2_cidt.analysis.features import load_features
from acise_p2_cidt.collect import gh_runs
from acise_p2_cidt.collect.targets import load_targets, slug
from acise_p2_cidt.data.load import load, pool_name
from acise_p2_cidt.stats import tests as st
from acise_p2_cidt.validate import live, sandbox
from acise_p2_cidt.viz import plots

OUT = pathlib.Path(__file__).resolve().parents[1] / "report" / "assets"
FIGURES = ["collection_fail_rate_per_source", "tree_cv_box_T2", "tree_importance_T2",
           "live_tree_icicle_T2", "live_retest_vs_cv_T2"]  # fmt: skip
CV_METHODS = ["cart", "cart_source_thr", "forest", "forest_source_thr", "source_rate",
              "parrot", "majority"]  # fmt: skip


# =============================================================================
# LaTeX output
# =============================================================================


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$"), ("~", r"\textasciitilde{}")):  # fmt: skip
        s = s.replace(a, b)
    return s


def tex_table(df: pd.DataFrame, align: str | None = None) -> str:
    """A booktabs tabular of `df` (index included), cells escaped."""
    df = df.reset_index()
    if align is None:  # numbers (and "a +/- b") right, text left
        num = df.apply(lambda c: c.astype(str).str.fullmatch(r"[-+<\d.\s/,]*").all())
        align = "".join("r" if n else "l" for n in num)
    rows = [" & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    rows += [" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    body = "\n".join(rows)
    return f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"


class Numbers:
    """Collects `\\newcommand`s; names are letters only, as LaTeX requires."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __setitem__(self, name: str, value) -> None:
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name!r}")
        if isinstance(value, float):
            value = f"{value:.3f}"
        elif isinstance(value, int):
            value = f"{value:,}".replace(",", "\\,")  # thin space, not escaped
        else:
            value = esc(value)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def fp(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def pm(s: pd.Series) -> str:
    return f"{s.mean():.3f} +/- {s.std():.3f}"


# =============================================================================
# Sections
# =============================================================================


def short_names(sources: list[str]) -> dict[str, str]:
    """`tokio`, `windmill-45664283`, `n8n@master`: the repo name, plus the workflow
    id when a repository has several workflows, plus the branch."""
    cfgs = {slug(c.target): c.target for c in load_targets(enabled_only=False)}
    repos = pd.Series([cfgs[s].repo for s in sources]).value_counts()
    out = {}
    for s in sources:
        t = cfgs[s]
        name = t.repo.split("/")[1]
        if t.branch:
            name += f"@{t.branch}"
        elif repos[t.repo] > 1:
            name += f"-{t.workflow_id}"
        out[s] = name
    return out


def data_section(sources, names, n: Numbers, tables: dict) -> dict:
    data = {t: load(sources, t) for t in ("T2", "T1")}
    rows = []
    for s in sources:
        m = json.loads((gh_runs.RESULTS_DIR / f"manifest_{s}.json").read_text())
        sc = json.loads((gh_runs.RESULTS_DIR / f"schema_{s}.json").read_text())
        row = {"source": names[s], "runs": m["n_runs"], "kept": sc["n_runs"],
               "infra dropped": sc["n_dropped"].get("infra_failure", 0)}  # fmt: skip
        for t, ds in data.items():
            y = ds.y[ds.source == s]
            row |= {f"{t} rows": len(y), f"{t} fail %": f"{100 * y.mean():.1f}"}
        rows.append(row)
    tables["sources"] = pd.DataFrame(rows).set_index("source")
    order = data["T1"].order
    n["WindowStart"], n["WindowEnd"] = str(order.min().date()), str(order.max().date())
    n["NSources"] = len(sources)
    n["RunsCollected"] = int(tables["sources"]["runs"].sum())
    n["RunsKept"] = int(tables["sources"]["kept"].sum())
    n["InfraDropped"] = int(tables["sources"]["infra dropped"].sum())
    for t, ds in data.items():
        k = t.replace("T1", "One").replace("T2", "Two")
        n[f"Rows{k}"] = len(ds.y)
        n[f"FailPct{k}"] = f"{100 * ds.y.mean():.1f}"
    return data


def model_section(dataset, n: Numbers, tables: dict) -> None:
    feats = {t: load_features(t, dataset) for t in ("T2", "T1")}
    width = max(map(len, feats.values()))
    tables["features"] = (
        pd.DataFrame({t: f + [""] * (width - len(f)) for t, f in feats.items()})
        .rename_axis("#")
        .set_axis(range(1, width + 1))
    )
    n["NFeatTwo"], n["NFeatOne"] = len(feats["T2"]), len(feats["T1"])
    side = json.loads(
        live.model_path(dataset, "T2", "cart").with_suffix(".json").read_text()
    )
    n["Cutoff"] = side["cutoff"][:10]
    n["DepthTwo"] = side["params"]["max_depth"]
    n["ThrTwo"] = side["threshold"]
    n["NTrainTwo"], n["NValTwo"] = side["n_train"], side["n_val"]


def cv_section(dataset, n: Numbers, tables: dict) -> dict:
    res = {
        t: pd.read_csv(live.MODEL_DIR / f"cv_results_{t}_{dataset}.csv")
        for t in ("T2", "T1")
    }
    rows = []
    for m in CV_METHODS:
        row = {"method": m}
        for t, r in res.items():
            p = r[(r["source"] == "all") & (r["method"] == m)]
            row[f"{t} MCC"] = pm(p["mcc"]) if len(p) else "-"
            row[f"{t} recall"] = f"{p['recall'].mean():.3f}" if len(p) else "-"
        rows.append(row)
    tables["cv"] = pd.DataFrame(rows).set_index("method")
    cells = []
    for t, r in res.items():
        k = t.replace("T1", "One").replace("T2", "Two")
        for m in ("cart", "forest", "source_rate", "parrot"):
            p = r[(r["source"] == "all") & (r["method"] == m)]["mcc"]
            n[f"CV{m.title().replace('_', '')}{k}"] = float(p.mean())
        table = st.per_cell(r, "mcc")
        # Parrot is constant on T2 (every T2 run follows a pass)
        pairs = [("cart", "cart_source_thr"), ("forest", "cart")]
        for a, b in pairs + ([("cart", "parrot")] if t == "T1" else []):
            c = st.compare(table, a, b)
            cells.append({"target": t, "comparison": f"{a} vs {b}",
                          "cells": len(table[[a, b]].dropna()),
                          "mean diff": f"{c['mean_diff']:+.4f}", "p": fp(c["p"]),
                          "A12": f"{c['A12']:.3f}"})  # fmt: skip
    tables["cells"] = pd.DataFrame(cells).set_index("target")
    return res


def transfer_section(dataset, names, cv2, n: Numbers, tables: dict) -> None:
    p = live.MODEL_DIR / f"e_results_T2_{dataset}.csv"
    e = pd.concat([cv2.query("method == 'cart' and source != 'all'"), pd.read_csv(p)])
    g = e.groupby(["source", "method"])["mcc"].mean().unstack("method")
    tables["transfer"] = (
        g.rename(
            index=names,
            columns={
                "cart": "E2 pooled",
                "cart_e1": "E1 own",
                "cart_e3": "E3 left out",
            },
        )
        .sort_index()
        .map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    )
    cells = st.per_cell(e, "mcc")
    for a, b, name in (
        ("cart", "cart_e1", "EtwoEone"),
        ("cart_e3", "cart_e1", "EthreeEone"),
    ):
        c = st.compare(cells, a, b)
        n[f"Diff{name}"] = f"{c['mean_diff']:+.4f}"
        n[f"P{name}"], n[f"A{name}"] = fp(c["p"]), c["A12"]
        n[f"Cells{name}"] = len(cells[[a, b]].dropna())


def retest_section(dataset, names, n: Numbers, tables: dict) -> None:
    rows, per = [], {}
    for t in ("T2", "T1"):
        k = t.replace("T1", "One").replace("T2", "Two")
        for m in ("cart", "forest"):
            r = pd.read_csv(
                f"{live.model_path(dataset, t, m).with_suffix('')}_retest.csv"
            )
            per[t, m] = (
                r[r["source"] != "all"].query("method == @m").set_index("source")
            )
            a = r[r["source"] == "all"].set_index("method")
            n[f"Retest{m.title()}{k}"] = float(a.loc[m, "mcc"])
            rows.append({"target": t, "model": m, "runs": int(a.loc[m, "n_test"]),
                         "failures": int(a.loc[m, "n_fail"]),
                         "MCC": f"{a.loc[m, 'mcc']:.3f}",
                         "precision": f"{a.loc[m, 'precision']:.3f}",
                         "recall": f"{a.loc[m, 'recall']:.3f}",
                         "Parrot MCC": f"{a.loc['parrot', 'mcc']:.3f}"})  # fmt: skip
        n[f"RetestRuns{k}"] = rows[-1]["runs"]
    tables["retest"] = pd.DataFrame(rows).set_index("target")
    c, f = per["T2", "cart"], per["T2", "forest"]
    tables["retest_sources"] = (
        pd.DataFrame(
            {
                "runs": c["n_test"],
                "failures": c["n_fail"],
                "CART MCC": c["mcc"].map("{:.3f}".format),
                "forest MCC": f["mcc"].map("{:.3f}".format),
            }
        )
        .rename(index=names)
        .sort_index()
    )


def sandbox_section(n: Numbers, tables: dict) -> None:
    """Reads the tables `validation.ipynb` saves from `sandbox.report`, if present."""
    d = sandbox.SANDBOX_DIR
    if not (d / "report_scores.csv").exists():
        n["SandboxDone"] = "no"
        return
    n["SandboxDone"] = "yes"
    s = pd.read_csv(d / "report_scores.csv", index_col="method")
    tables["sandbox_scores"] = s[
        ["n_test", "n_fail", "mcc", "precision", "recall"]
    ].round(3)
    tables["sandbox_breakage"] = pd.read_csv(d / "report_by_breakage.csv",
                                             index_col="breakage").round(3)  # fmt: skip
    tables["sandbox_confusion"] = pd.read_csv(
        d / "report_confusion.csv", index_col="actual"
    )
    imp = pd.read_csv(d / "report_imputed.csv", index_col=0)
    tables["sandbox_imputed"] = imp.round(2)
    model = s.index[0]
    n["SandboxRuns"], n["SandboxFails"] = (
        int(s.loc[model, "n_test"]),
        int(s.loc[model, "n_fail"]),
    )
    n["SandboxMCC"], n["SandboxParrotMCC"] = (
        float(s.loc[model, "mcc"]),
        float(s.loc["parrot", "mcc"]),
    )
    runs = pd.read_csv(d / "report_runs.csv")
    n["SandboxMaxP"] = float(runs["p_fail"].max())
    n["SandboxLeadMedian"] = int(runs["lead_secs"].median())
    preds = [json.loads(line) for line in (d / "predictions.jsonl").read_text().splitlines()]
    # no earlier run in the run's lineage: Parrot has nothing to repeat
    n["SandboxFirstPct"] = f"{100 * np.mean([p['prev_failed'] is None for p in preds]):.0f}"
    dropped = json.loads((d / "report_dropped.json").read_text())
    n["SandboxDropped"] = int(sum(dropped.values()))
    plan = json.loads((d / "scenarios.json").read_text())["scenarios"]
    scored = [p for p in plan if p["phase"] == "scored"]
    n["SandboxWarmup"] = len(plan) - len(scored)
    n["SandboxScored"] = len(scored)
    n["SandboxBreakPct"] = (
        f"{100 * np.mean([p['breakage'] != 'none' for p in scored]):.0f}"
    )


def main(out: pathlib.Path = OUT) -> pathlib.Path:
    sources = [slug(c.target) for c in load_targets()]
    dataset = pool_name(sources)
    names = short_names(sources)
    n, tables = Numbers(), {}
    n["Pool"] = dataset
    n["NRepos"] = len({c.target.repo for c in load_targets()})
    data_section(sources, names, n, tables)
    model_section(dataset, n, tables)
    res = cv_section(dataset, n, tables)
    transfer_section(dataset, names, res["T2"], n, tables)
    retest_section(dataset, names, n, tables)
    sandbox_section(n, tables)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for k, df in tables.items():
        (out / "tables" / f"{k}.tex").write_text(tex_table(df))
    (out / "numbers.tex").write_text(n.text())
    for f in FIGURES:
        if (p := plots.FIG_DIR / f"{f}.png").exists():
            shutil.copy(p, out / "figures" / p.name)
    return out


if __name__ == "__main__":
    print(main())
