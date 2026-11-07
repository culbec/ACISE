# Multiple Repository Analysis

Feature specification for the RF+DT CI failure prediction project (project 2). It states **what** the feature does and which assets it produces. **How** it is built lives in [plan.md](plan.md) and [implementation-backbone.md](implementation-backbone.md).

## 1. Purpose

Train and evaluate the CI failure model on runs from many workflows of many open-source repositories, on all branches and events those workflows run on. Today it uses one workflow of one repository, on one branch.

## 2. Problem

- GitHub keeps workflow runs for about 400 days. One workflow cannot give more history than that.
- Runs on the default branch after a merge rarely fail. Code was already checked before the merge, so the failure class is small. For several candidate workflows, the default branch has only a few failures a month.
- Most CI failures happen before the merge: on pull requests, feature branches and merge queues. These are the cases where a prediction is useful, and where new features and changes show their patterns.

More workflows and more branches give more runs and, above all, more failures. Pooling does not balance the classes. The pooled failure rate is a weighted mean of the sources' rates.

## 3. Concepts

| Term | Meaning |
|---|---|
| **Target** | One workflow of one repository. A repository can have several targets (one per workflow). A target covers all branches and all supported events unless it is restricted to one branch. |
| **Run** | One execution of a target's workflow. |
| **Supported events** | Events that carry a code change: `push`, `pull_request`, `merge_group`. Scheduled and manual runs carry no change and are left out. |
| **Lineage** | The sequence of runs that test the same line of work: one pull request (including its merge-queue runs), or one branch for push runs. |
| **Label** | The conclusion of a run's **first attempt**: `failure` or `success`. Other conclusions (cancelled, skipped) are left out. |
| **Infrastructure failure** | A failure caused by publishing, deployment, image pushes, secrets, runner setup, tool installs, or rate limits. No pre-run information can predict it. |
| **Code failure** | A failure of build, lint, type check or tests on the pushed change. Only these count as positive labels. |
| **Source** | The target a row of the pooled dataset comes from. |
| **Collection date C** | The day all targets were collected. Runs created after C are new data for live validation. |

## 4. Scope

**In scope**
- A registry of targets, each one enabled or disabled.
- Collecting all runs of each enabled target, on all branches, for the supported events.
- A label check per target, before any target is used for training.
- One dataset per target, then a pooled dataset over the enabled targets.
- Evaluation per source and pooled, transfer to an unseen repository, and a live sandbox.

**Out of scope**
- New model families. CART and the random forest stay as they are.
- Features that depend on one repository's layout, package names or job names (the portability rule of the main plan).
- Normalising features across repositories.
- Joining a workflow's history across renames.

## 5. Assets

| Asset | Kind | Content |
|---|---|---|
| Target registry | input, versioned | Per target: repository, workflow, optional branch restriction, enabled flag, jobs whose failures count as infrastructure. |
| Raw run cache | intermediate, not versioned | Everything fetched from the GitHub API and git for one target. It can be resumed and is never re-fetched without need. |
| Label check report | output, versioned | Per target: workflow role, failed-job breakdown, infrastructure share, volume, flaky rate, verdict. |
| Per-target dataset | output, not versioned | One row per run: pre-run features, label, ids, source, diagnostics. A schema file describes it (versioned). |
| Pooled dataset | in memory | The rows of all enabled targets, in global time order, with the source of each row. |
| Models | output | Final CART and forest trained on the pooled dataset, with the list of their sources. |
| Evaluation reports | output | Pooled and per-source results of every experiment in section 8. |
| Sandbox | external repository | A small codebase with tests and a CI workflow, broken on purpose for live validation. |

## 6. Functional requirements

- **FR1. Target registry.** Targets are read from one registry. It is validated when read: malformed or duplicate entries are rejected with a message naming the entry. Disabled targets are ignored by every step.
- **FR2. Several targets per repository.** Two workflows of one repository are two targets. Their data never overwrites each other. Data already collected for the current target stays usable.
- **FR3. All branches and events.** Collection covers every branch the workflow runs on, for the supported events, including pull requests from forks and merge-queue runs. A target can still be restricted to one branch.
- **FR4. Complete collection.** No run is lost to API result caps, however many runs a target has per day. An interrupted collection resumes where it stopped.
- **FR5. Label integrity.** A run whose first-attempt failures are all infrastructure failures is left out, not relabelled as a success, and counted. A run with at least one code failure keeps its failure label.
- **FR6. Per-target history.** History features are computed within one target only, never across targets.
- **FR7. Lineage-aware history.** "Previous run" means the latest run of the **same lineage** completed before the current run was created. Features that describe overall load and recent failure rates still cover the whole target. On a target restricted to one branch, the existing features equal today's.
- **FR8. Change description off the default branch.** For a pull request or merge-queue run, the change is the whole proposed change against its base. For a push run, it is the pushed commit. File and author history only use commits that existed before the run.
- **FR9. Run context.** Each row records the event and whether the change comes from a fork. Both are repository-agnostic.
- **FR10. Pooled dataset.** Rows of all enabled targets are ordered by creation time, so sources interleave. They are not grouped by source, and not shuffled. Each row keeps its source. Tables with different feature columns cannot be pooled.
- **FR11. Per-source reporting.** Every evaluation reports results per source as well as pooled, with baselines computed the same way.
- **FR12. Live use.** A pooled model can be retested on runs created after C for all its sources, and can predict a single new run of any of its sources.

## 7. Constraints

- **No leakage.** A feature only uses information available when the run was created: runs completed before it, commits reachable from its head.
- **Portability.** Every feature must be computable for any GitHub repository from the Actions API and git alone.
- **Time order.** The dev segment and the cross-validation folds follow creation time across the pooled data.
- **Privacy.** Authors are pseudonymised per repository. Raw identities and tokens are never stored next to the results.
- **Reproducibility.** For every target, the collection window, the API queries, the counts and the label-check verdict are recorded.
- **Budget.** Collection and build must fit the API rate limit and the available time. The collection window is chosen per study, the same for all targets.

## 8. Evaluation

| Id | Train | Test | Question |
|---|---|---|---|
| E1 | one source | the same source, time-ordered folds | how well does the model do per source on its own? |
| E2 | pooled sources | pooled, time-ordered folds, reported per source | does pooling help each source? |
| E3 | all sources but one repository | the held-out repository | does the model transfer to a repository it never saw? |
| E4 | final pooled model | live runs of the sandbox | does it behave as expected on new runs with a known cause? |

- E1 and E2 are compared per source with the project's paired statistical tests.
- E4 rules: the sandbox gets a warm-up of normal changes before scoring, because history features need earlier runs. Every change shape is used both broken and unbroken. Breakages are real code defects caught by tests. The scenario order is randomised. Predictions are made before each run ends. The result is reported next to the history baselines on the same runs.

## 9. Candidate targets

Each must pass the label check (plan, phase 3) before it is enabled.

| Repository | Workflow |
|---|---|
| n8n-io/n8n | [ci-pull-requests.yml](https://github.com/n8n-io/n8n/actions/workflows/ci-pull-requests.yml) |
| astral-sh/ruff | [ci.yaml](https://github.com/astral-sh/ruff/actions/workflows/ci.yaml) |
| astral-sh/uv | [ci.yml](https://github.com/astral-sh/uv/actions/workflows/ci.yml) |
| windmill-labs/windmill | [backend-check.yml](https://github.com/windmill-labs/windmill/actions/workflows/backend-check.yml), [frontend-check.yml](https://github.com/windmill-labs/windmill/actions/workflows/frontend-check.yml) |
| sharkdp/fd | [CICD.yml](https://github.com/sharkdp/fd/actions/workflows/CICD.yml) |
| github/gh-aw | [ci.yml](https://github.com/github/gh-aw/actions/workflows/ci.yml) |
| Homebrew/brew | [tests.yml](https://github.com/Homebrew/brew/actions/workflows/tests.yml) |
| openclaw/openclaw | [ci.yml](https://github.com/openclaw/openclaw/actions/workflows/ci.yml) |
| tokio-rs/tokio | [ci.yml](https://github.com/tokio-rs/tokio/actions/workflows/ci.yml) |
| sveltejs/svelte | [ci.yml](https://github.com/sveltejs/svelte/actions/workflows/ci.yml) |
| nushell/nushell | [ci.yml](https://github.com/nushell/nushell/actions/workflows/ci.yml) |
| servo/servo | [main.yml](https://github.com/servo/servo/actions/workflows/main.yml) |

## 10. Acceptance criteria

1. Every candidate has a label-check verdict before any training on the pooled data.
2. Collection of an enabled target holds runs from more than one branch and event when the workflow runs on them, with no gap caused by API caps.
3. Two targets of the same repository can be collected and built side by side.
4. A run that failed only in infrastructure jobs is excluded and counted. A run with one code failure is kept as a failure.
5. On a single-branch target, the lineage-aware build gives the same history features as the single-lineage build (checked on synthetic data); only the new columns are added.
6. The pooled dataset is in global creation order, every row has a source, and mismatched tables are refused.
7. E1 to E3 are reported per source and pooled, with baselines. E4 is reported with its scenario breakdown.
8. Existing behaviour and tests keep working, except where this spec changes a definition on purpose (plan, decisions).

## 11. Decisions

- Keep global time order and interleave sources (no shuffling, no grouping by source).
- Run a label check on every candidate before it is used.
- Evaluate with transfer (E3) and a live sandbox (E4), in addition to the retest on runs after C.
- Collect all branches and supported events of each workflow, not only the default branch.
