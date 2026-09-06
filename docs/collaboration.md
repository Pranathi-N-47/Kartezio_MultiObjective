# -------------------------------------------------------------------------
# Kartezio – Multi‑Objective Extension (NSGA‑II)
# Copyright (c) 2024‑2026 Inserm Transfert SA and co‑owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non‑commercial research only, and must retain this header.
# -------------------------------------------------------------------------

# Collaboration Guide – Multi‑Objective Kartezio

*(keep this file in the repository root as `COLLABORATION.md`)*

---

## 1. Project Overview

The repository contains the original **Kartezio** CGP framework and a new **multi‑objective (MOO) extension** based on NSGA‑II. The extension lives in a dedicated package `kartezio/moo/` so that the original code remains untouched.

### 1.1 Full GitHub Repository Structure (tentative - can be modified in future)

```
kartezio-multiobjective/
│
├── .git/
│
├── .venv/                              # Local virtual environment (Git-ignored)
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # Tests, linting and CI checks
│
├── LICENSE                             # Kartezio non-commercial research licence
├── README.md                            # Project overview + MOO extension
├── pyproject.toml                       # Package, build and dependency configuration
├── uv.lock                              # Reproducible dependency lock file
├── mkdocs.yml                            # Documentation website configuration
│
├── src/
│   └── kartezio/
│       │
│       ├── __init__.py                  # Public package API
│       │
│       ├── core/                        # EXISTING KARTEZIO
│       │   ├── __init__.py
│       │   ├── components.py
│       │   ├── fitness.py               # Existing performance/fitness system
│       │   ├── initialization.py
│       │   ├── reducer.py
│       │   └── ...
│       │
│       ├── evolution/                   # EXISTING KARTEZIO
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── decoder.py
│       │   ├── mcts.py
│       │   ├── population.py
│       │   └── strategy.py               # Existing 1+λ evolution
│       │
│       ├── mutation/                    # EXISTING KARTEZIO
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── ...
│       │
│       ├── primitives/                  # EXISTING KARTEZIO
│       │   ├── __init__.py
│       │   └── ...
│       │
│       ├── data/                        # EXISTING KARTEZIO
│       │   ├── __init__.py
│       │   └── ...
│       │
│       ├── vision/                     # EXISTING KARTEZIO
│       │   ├── __init__.py
│       │   └── ...
│       │
│       ├── utils/                       # EXISTING KARTEZIO
│       │   ├── __init__.py
│       │   ├── io.py
│       │   └── ...
│       │
│       └── moo/                         # NEW MULTI-OBJECTIVE EXTENSION
│           ├── __init__.py
│           │
│           ├── objectives.py             # Objective interface/adapters
│           │                                # Uses existing Kartezio performance
│           │                                # Adds complexity objectives
│           │
│           ├── complexity.py             # Complexity metrics
│           │                                # ProcessTime
│           │                                # ActiveNodeCount
│           │                                # OperationCount
│           │
│           ├── dominance.py              # Pareto dominance calculations
│           │
│           ├── population.py              # MOO population representation
│           │                                # Objective values
│           │                                # Pareto rank
│           │                                # Crowding distance
│           │
│           ├── strategy.py                # NSGA-II evolutionary strategy
│           │
│           ├── trainer.py                 # High-level MOO training interface
│           │
│           ├── pareto.py                  # Pareto-front extraction,
│           │                                # ranking and solution utilities
│           │
│           └── robustness.py              # Post-hoc robustness evaluation
│                                            # Noise, blur, brightness, contrast,
│                                            # JPEG compression, etc.
│
├── tests/
│   │
│   ├── core/                            # EXISTING KARTEZIO TESTS
│   │   └── ...
│   │
│   ├── evolution/
│   │   └── ...
│   │
│   ├── mutation/
│   │   └── ...
│   │
│   └── moo/                            # NEW EXTENSION TESTS
│       ├── __init__.py
│       ├── test_objectives.py
│       ├── test_complexity.py
│       ├── test_dominance.py
│       ├── test_population.py
│       ├── test_strategy.py
│       ├── test_pareto.py
│       ├── test_trainer.py
│       └── test_robustness.py
│
├── examples/
│   ├── basic_trainer.py                 # Existing/basic Kartezio usage
│   ├── moo_example.py                   # Basic MOO example
│   ├── deep_pcb_example.py              # DeepPCB demonstration
│   └── robustness_example.py             # Post-hoc robustness example
│
├── experiments/
│   ├── baseline_kartezio/
│   │   └── ...
│   │
│   ├── multiobjective/
│   │   └── ...
│   │
│   ├── robustness/
│   │   └── ...
│   │
│   └── deep_pcb/
│       └── ...
│
├── docs/                                # MkDocs documentation website
│   │
│   ├── index.md                         # Documentation homepage
│   │
│   ├── installation.md                  # Installation/setup
│   ├── quickstart.md                    # Quick-start tutorial
│   │
│   └── collaboration.md                 # Contribution/development guide
│
└── .gitignore
```

> **All new files must start with the licence header** (see the block at the top of this file).  Any modification of original Kartezio code is wrapped in a clear comment block indicating the reason for the change (see § 5.2 in the guide).

---

## 2. Git Branching Model

| Branch            | Purpose                                            |
|-------------------|----------------------------------------------------|
| `main`            | Stable released version (published on PyPI).       |
| `develop`         | Integration branch for the whole MOO effort. All feature PRs target this branch. |
| `feature/<area>`  | Work on a specific module (objectives, nsga2, tests, docs). |
| `release/vX.Y‑moo`| Temporary release branch used to tag the final version before merging to `main`. |
| `hotfix/<desc>`   | Emergency fix that must go to `main` immediately. |

### Typical workflow for a contributor
```bash
# 1️⃣ Update local develop
git checkout develop && git pull

# 2️⃣ Create feature branch
git checkout -b feature/<my‑area>

# … make changes, add licence headers, write tests …

# 3️⃣ Commit (see § 4) and push
git add .
git commit -m "<Name> - <short description>"
git push -u origin feature/<my‑area>

# 4️⃣ Open a Pull Request → target = develop
#    CI (tests + lint) runs automatically (see § 3).

# 5️⃣ When PR is approved → merge (fast‑forward) into develop.
#    Periodically rebase your branch on the latest develop to stay up‑to‑date.
```

---

## 3. Team Allocation (4 people ≈ 25 % each)

Each person works on a separate Git feature branch and a set of files that do not overlap with the other team members', so everyone can develop in parallel without blocking each other. The only shared file is `src/kartezio/moo/__init__.py`, which is updated in one small dedicated PR after all four feature branches are merged.

---

### Person A – Objectives & Complexity Metrics
**Branch:** `feature/objectives`

Person A is responsible for defining the new `ComplexityMetric` component type and implementing all built-in complexity metrics that the MOO extension will optimise alongside the existing performance (fitness) metrics.

**What to do:**

Person A must first study `src/kartezio/core/components.py` closely, particularly how the `@fundamental()` decorator, the `@register()` decorator, and the `Components` registry work. The existing `Fitness` base class in `src/kartezio/core/fitness.py` is the model to follow — `ComplexityMetric` should mirror the same pattern so that users can register their own custom complexity metrics using `@register(ComplexityMetric)`, exactly as they can register custom fitness metrics today.

Person A must then implement the following:

1. `src/kartezio/moo/objectives.py` — Define the abstract `ComplexityMetric` base class decorated with `@fundamental()`. It must expose a single abstract method `evaluate(individual, context) -> float` that every built-in and user-supplied metric must implement. Also define a thin `PerformanceObjective` adapter that wraps the existing `Fitness` classes so they can be treated as one of the MOO objectives without any change to the original code.

2. `src/kartezio/moo/complexity.py` — Implement the three built-in complexity metrics that inherit from `ComplexityMetric`:
   - `ActiveNodeCount`: counts the number of active (non-silent) nodes in the CGP graph of an individual. To compute this, Person A must call `decoder.parse_to_graphs(genotype)` and count the unique active nodes across all outputs.
   - `ProcessTime`: reads the per-individual wall-clock inference time that is already stored in `population.score.time` (populated in `src/kartezio/evolution/decoder.py` at line 189 via `population.set_time(i, t)`). It simply returns that stored value without re-running inference.
   - `OperationCount`: counts the total number of arithmetic/morphological operations performed by the active nodes (i.e., the sum of the arities of all active nodes).

3. `src/kartezio/moo/__init__.py` — Export `ComplexityMetric`, `PerformanceObjective`, `ActiveNodeCount`, `ProcessTime`, and `OperationCount` so users can import them with `from kartezio.moo import ComplexityMetric`.

**What to write:**

Every new file must start with the mandatory licence header (see § 5.1). Every class and method must have a docstring. Inline comments must be written in plain English and must explain *why* a design choice was made, not just *what* the code does.

Person A does **not** touch any existing file outside `src/kartezio/moo/`. If a bug is discovered in the original code during this work, it must be reported to the team rather than fixed unilaterally.

---

### Person B – NSGA-II Engine & Robustness Evaluation
**Branch:** `feature/nsga2`

Person B is responsible for the entire multi-objective evolutionary engine — Pareto dominance logic, the MOO population container, the NSGA-II selection strategy, the high-level training interface, the Pareto-front utilities — **and** for the post-hoc robustness evaluation module. Robustness evaluation is not an optional add-on; it is a mandatory step that runs automatically after evolution finishes and whose results are always returned to the user alongside the Pareto front.

**What to do:**

Person B must first read `src/kartezio/evolution/base.py` and `src/kartezio/evolution/strategy.py` to understand how the existing 1+λ evolution loop and the `step()` hook work. The existing `KartezioTrainer` class is the model that the new `KartezioMOOTrainer` wraps.

Person B must implement the following:

1. `src/kartezio/moo/dominance.py` — Implement fast non-dominated sorting (the standard NSGA-II procedure). Provide a `dominates(a, b)` function that returns `True` if individual `a` Pareto-dominates individual `b` across all objective values. Build on top of this a `non_dominated_sort(population)` function that assigns a Pareto rank to every individual, and a `crowding_distance(front)` function that computes the crowding distance for each individual within a single front.

2. `src/kartezio/moo/population.py` — Define `MOOPopulation`, a container that extends the existing `Population` with additional arrays: an `objectives` matrix (one column per objective), a `ranks` vector (Pareto rank per individual), a `crowding` vector (crowding distance per individual), and a `robustness` dict that maps each solution index to its per-perturbation robustness scores. This class must be initialised by the trainer and filled by the NSGA-II strategy each generation.

3. `src/kartezio/moo/strategy.py` — Implement `NSGAIIStrategy`. This class must expose a `step(population, objectives)` method that (a) evaluates all objective functions for every individual in the population, (b) calls `non_dominated_sort` and `crowding_distance` from `dominance.py`, (c) selects the next generation's survivors by preferring lower rank and, as a tie-breaker, higher crowding distance, and (d) returns the updated `MOOPopulation`. The `step()` method is called by the existing `evolve()` loop in `base.py` via the `hasattr(strategy, "step")` hook that is already present in the code — Person B must not modify `base.py` to make this work.

4. `src/kartezio/moo/robustness.py` — Implement the post-hoc robustness evaluation engine. This is a **core feature** of the project, not a testing utility. The module must:
   - Define a `RobustnessSuite` class that holds a configurable list of image perturbations. Each perturbation is a callable that takes a single image and returns a perturbed version of it. The built-in perturbations to include are: Gaussian noise (several σ levels), Gaussian blur (several kernel sizes), brightness shift (±20, ±40 intensity units), contrast scaling (0.5×, 2×), and JPEG compression (quality 30, 60).
   - Provide an `evaluate_pipeline(decoder, genotype, x_test, y_test, fitness_fn, suite)` function that runs inference on each test image under each perturbation, computes the fitness drop relative to the clean-image baseline, and returns a structured `RobustnessReport` object. The `RobustnessReport` holds: the mean and standard deviation of the fitness drop per perturbation type, an overall robustness score (mean fitness retention across all perturbations, in the range [0, 1]), and a per-image breakdown.
   - Provide an `evaluate_pareto_front(front, decoder, x_test, y_test, fitness_fn, suite=None)` function that calls `evaluate_pipeline` for every solution in the Pareto front and attaches the `RobustnessReport` to each solution in the `MOOPopulation.robustness` dict.
   - Provide a `summarise(front_with_robustness)` function that prints a human-readable table to stdout showing, for each solution in the Pareto front: its Pareto rank, each objective value, and its overall robustness score. This is the **primary way results are presented to the user** after training.
   - Provide a `to_dataframe(front_with_robustness)` function that returns a `pandas.DataFrame` containing the same information as `summarise`, suitable for saving to CSV or plotting.

5. `src/kartezio/moo/trainer.py` — Implement `KartezioMOOTrainer`. It accepts a list of objective instances, builds an `NSGAIIStrategy`, runs evolution, and exposes a `fit(n_generations, x_train, y_train, x_test, y_test, callbacks=None)` method. After the evolution loop completes, `fit()` must automatically call `robustness.evaluate_pareto_front(...)` and then call `robustness.summarise(...)` so the user sees the full results table on the console without having to do anything extra. The method returns the fully annotated `MOOPopulation` (Pareto front with objective values and robustness scores attached).

6. `src/kartezio/moo/pareto.py` — Utility functions for working with the Pareto front after evolution: `extract_front(population)` to pull out rank-0 individuals, `select_knee_point(front)` to pick the single solution that best balances all objectives, and `to_dataframe(front)` to produce a `pandas.DataFrame` suitable for saving to CSV or displaying in a notebook.

**What to write:**

Every new file starts with the licence header (§ 5.1). Any time Person B needs to interact with the original `decoder.py` or `population.py` code in a way that requires a change to those files, they must wrap the change in the `BEGIN/END KARTEZIO-MOO MODIFICATION` comment block (see § 5.2) and open a separate minimal PR for that change so the whole team can review it.

Person B does **not** modify `objectives.py` or `complexity.py` — those belong to Person A.

---

### Person C – Testing & CI
**Branch:** `feature/tests`

Person C is responsible for the entire automated test suite for the MOO extension and for setting up the continuous integration pipeline. This work can only begin in earnest once Persons A and B have their first working drafts merged into `develop`, but the test *skeleton* (empty test files with function stubs and docstrings) can be written immediately so that CI runs green from day one.

**What to do:**

Person C must implement:

1. `tests/moo/test_objectives.py` — Unit tests for every class in `objectives.py` and `complexity.py`. Tests must cover:
   - That `ComplexityMetric` cannot be instantiated directly (abstract class enforcement).
   - That `ActiveNodeCount.evaluate()` returns the correct integer count for a hand-crafted genotype with a known number of active nodes.
   - That `ProcessTime.evaluate()` reads the stored time value from the population rather than re-running inference.
   - That `OperationCount.evaluate()` returns the correct sum for a known graph.
   - That a user-defined subclass of `ComplexityMetric` can be registered with `@register(ComplexityMetric)` and then instantiated via `load_component`.

2. `tests/moo/test_dominance.py` — Unit tests for `dominance.py`. Tests must cover:
   - `dominates(a, b)` for all four cases: a dominates b, b dominates a, neither dominates the other, and both are identical.
   - `non_dominated_sort` assigns rank 0 to the true Pareto front and rank ≥ 1 to all others.
   - `crowding_distance` assigns infinity to the boundary solutions and a finite positive value to interior solutions.

3. `tests/moo/test_population.py` — Unit tests for `MOOPopulation`. Tests must cover shape and dtype of `objectives`, `ranks`, `crowding`, and `robustness` fields after initialisation and after a strategy step.

4. `tests/moo/test_strategy.py` — Unit tests for `NSGAIIStrategy.step()`. Tests must cover:
   - The returned population has the correct number of individuals.
   - The ranks are non-negative integers.
   - The solution with the lowest rank and highest crowding distance appears in the output.

5. `tests/moo/test_robustness.py` — Unit and integration tests for the robustness engine implemented by Person B. Tests must cover:
   - Each built-in perturbation (Gaussian noise, Gaussian blur, brightness shift, contrast scaling, JPEG compression) runs without raising an exception on a synthetic 64×64 image.
   - `evaluate_pipeline` returns a `RobustnessReport` whose overall score is a float in [0, 1].
   - Given a fixed random seed, the report is fully reproducible.
   - `evaluate_pareto_front` correctly attaches a `RobustnessReport` to every solution in the `MOOPopulation.robustness` dict.
   - `summarise` produces non-empty console output (captured with `capsys`) that includes a row for each solution.
   - `to_dataframe` returns a DataFrame with the expected column names and one row per Pareto-front solution.

6. `tests/moo/test_trainer.py` — End-to-end integration test for `KartezioMOOTrainer.fit()`. Using a tiny synthetic dataset (4 images, 64×64) and a very short evolution run (5 generations), it must:
   - Complete without raising an exception.
   - Return a `MOOPopulation` with at least one rank-0 solution.
   - Return a population whose `robustness` dict is non-empty (i.e., robustness evaluation ran automatically).
   - Confirm that the console output produced during `fit()` includes the robustness summary table.

7. `tests/moo/test_pareto.py` — Unit tests for `pareto.py` utility functions (`extract_front`, `select_knee_point`, `to_dataframe`).

8. `.github/workflows/ci.yml` — A GitHub Actions workflow that runs `pytest -q tests/` and `ruff check src/` on every push and pull request. The workflow must also install the project's dev dependencies using `uv`.

9. Updates to `pyproject.toml` — Add `pytest`, `pytest-cov`, and `ruff` under the `[dependency-groups]` dev section if they are not already present.

Person C is also responsible for running the **existing** core test suite after each merge into `develop` and flagging any regressions immediately.

**What to write:**

All new test files must start with the licence header (§ 5.1). Test function names must follow the pattern `test_<thing_being_tested>_<condition>`, e.g., `test_robustness_report_score_is_normalised`. Every test must have a one-line docstring stating what it verifies.

---

### Person D – Documentation & Examples
**Branch:** `feature/docs`

Person D is responsible for all user-facing written material: the MkDocs documentation website, the project README, and runnable example scripts. This work depends on the final API being stable, so Person D should start with the README update and the example skeleton, then fill in the full MkDocs content once Persons A and B have merged into `develop`.

**What to do:**

Person D must deliver:

1. `docs/index.md` — The documentation homepage. It must be designed to look and feel like the PyTorch or MkDocs-Material style pages. It should contain a short project description, a feature highlights list (including the robustness evaluation as a named feature), a "Quick Start" code snippet (using `KartezioMOOTrainer`), and links to the other documentation pages. **This is a web-page-style document rendered by MkDocs, not just a plain markdown file** — Person D must configure `mkdocs.yml` to use the Material theme so the output is a proper interactive site.

2. `docs/installation.md` — Step-by-step instructions for installing the package using `uv` or `pip`, including how to install the optional dev dependencies.

3. `docs/quickstart.md` — A narrative tutorial that walks a new user through: (a) loading a dataset, (b) configuring the objectives, (c) running `KartezioMOOTrainer.fit()`, (d) reading the console output that is automatically printed (the Pareto-front table with objective values and robustness scores), (e) saving the results to CSV with `robustness.to_dataframe()`, and (f) interpreting the robustness scores to choose a pipeline that is both accurate and computationally light.

4. `docs/moo_guide.md` — A deeper technical guide covering: the NSGA-II algorithm used internally; how to define a custom complexity metric; how to interpret Pareto ranks and crowding distances; and a dedicated section on the robustness evaluation module. That robustness section must explain what each built-in perturbation type represents, how the overall robustness score is calculated (mean fitness retention across all perturbations), how to add custom perturbations to the `RobustnessSuite`, and how to interpret the `RobustnessReport` printed to the console.

5. `docs/robustness_guide.md` — A standalone page dedicated entirely to robustness evaluation. It must explain the motivation (a pipeline that is accurate on clean images may degrade badly under real-world imaging artefacts), show the complete list of built-in perturbations and the severity levels used, explain the `RobustnessReport` fields in detail, provide a worked example with annotated console output, and show how to save and plot the per-perturbation breakdown.

6. `mkdocs.yml` — Configure the MkDocs site with the Material theme, the navigation structure (including the new `robustness_guide.md` page), and any necessary extensions (e.g., `pymdownx.highlight` for code blocks, `admonition` for alert boxes).

7. `README.md` — Add a "Multi-Objective Extension" section that briefly explains the motivation, lists the new objectives, mentions the automatic robustness evaluation as a headline feature, links to the documentation site, and provides a minimal usage example that shows the robustness summary being printed.

8. `examples/moo_example.py` — A self-contained, runnable script that demonstrates a full MOO training run from data loading through to the automatically printed Pareto-front and robustness summary table. It must then show how to save the results to CSV. It must be extensively commented so a user unfamiliar with CGP can follow along.

9. `examples/robustness_example.py` — A self-contained script that loads a pre-trained Pareto front, runs the robustness evaluation manually (i.e., calling `evaluate_pareto_front` directly rather than relying on the trainer to do it automatically), prints the summary table, plots per-perturbation fitness drops as a bar chart, and saves the chart as a PNG file.

**What to write:**

Documentation files do not need the Python licence header, but the example scripts (`moo_example.py` and `robustness_example.py`) are Python files and **must** include the licence header (§ 5.1). All prose in the documentation must be written in clear, jargon-light English aimed at a researcher who understands image segmentation but may not be familiar with evolutionary computation.

> **Parallelism guarantee:** each person works on a disjoint set of files, so merges never conflict except for `src/kartezio/moo/__init__.py`. That file is updated in one small dedicated PR (`feature/init-updates`) after all four feature branches have been merged into `develop`.

---

## 4. Commit‑Message Format
```
<Name> - <short description of change>

<optional longer description (wrap at 72 chars)>

References:
- <issue/feature ID if any>
- <link to related PR>
```
*Examples*:
```
Alice - add ComplexityMetric base class

Introduced a new fundamental component that registers under the
Components registry. Includes licence header and basic docstring.

References:
- #12 (MOO design doc)
```
```
Bob - implement NSGA‑II selection step

Added fast non‑dominated sorting, crowding distance calculation,
and integrated the step() hook into the existing evolution loop.

References:
- #15 (NSGA‑II implementation)
```
All commits must end with a blank line before the “References:” block.

---

## 5. Licence‑Compliance Comments

### 5.1 Licence header (copy‑paste into **every new .py file**)
```python
# -------------------------------------------------------------------------
# Kartezio – Multi‑Objective Extension (NSGA‑II)
# Copyright (c) 2024‑2026 Inserm Transfert SA and co‑owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non‑commercial research only, and must retain this header.
# -------------------------------------------------------------------------
```

### 5.2 Inline modification comment (when touching original code)
```python
# === BEGIN KARTEZIO‑MOO MODIFICATION ==============================
# File: src/kartezio/evolution/decoder.py
# Reason: expose per‑image inference time for ProcessTime metric.
# Changes: added `self.last_time` and populated `population.score.time`.
# === END KARTEZIO‑MOO MODIFICATION ==============================
```
Place the block **just above** the modified code segment.

---

## 6. Testing Strategy
| Test module | What it verifies | Example assertions |
|-------------|-----------------|--------------------|
| `tests/moo/test_objectives.py` | `ActiveNodeCount` returns correct node count; `ProcessTime` returns the stored wall‑clock time. | `assert metric.evaluate(ind, ctx) == len(active_nodes)` |
| `tests/moo/test_dominance.py` | Correct dominance relation (`dominates(a, b)`), crowding‑distance ordering. | `assert dominates(front[0], front[1])` |
| `tests/moo/test_population.py` | Population creation, fitness & time arrays shape, elitism handling. | `assert pop.score.fitness.shape == (pop.size,)` |
| `tests/moo/test_strategy.py` | `NSGAIIStrategy.step()` returns a new `Population` with proper rank & crowding. | `assert new_pop.ranks.max() <= old_pop.ranks.max() + 1` |
| `tests/moo/test_trainer.py` | End‑to‑end training on a tiny synthetic dataset (e.g., 2 images). Checks that a Pareto front of size ≥ 2 is produced. | `assert len(trainer.fit(...).front) >= 2` |
| Existing core tests | Must still pass unchanged after the extension. | `pytest -q` should report **0** failures. |
All tests are executed by the CI job:
```yaml
- name: Run pytest
  run: pytest -q tests/
```
---

## 7. Pull‑Request (PR) Workflow
| PR | When to open | What to include |
|----|--------------|-----------------|
| `feature/objectives` | After `objectives.py` and `__init__.py` are ready. | Code, licence headers, inline modification comment (if any). |
| `feature/nsga2` | After all new engine files compile and pass local tests. | Full implementation, unit‑tests for each class, CI passes. |
| `feature/tests` | After **all** new modules exist (objectives + nsga2). | Test suite covering every new file, updates to `pyproject.toml` (dev deps). |
| `feature/docs` | After code is merged into `develop`. | Docs referencing the new API; example script that runs on `develop`. |
| `feature/init‑updates` | *Optional* – if `kartezio/moo/__init__.py` needs both import lists. | Small PR that adds the missing re‑exports; reviewers from any team member. |
| `release/vX.Y‑moo` | After the four feature PRs are merged. | Minor bump of version in `pyproject.toml`, final integration test run, tag creation. |

**PR template** (copy‑paste into each PR description):
```markdown
## Summary
<One‑sentence description of the change.>

## Related Issue(s)
#12, #15 …

## Checklist
- [ ] Code follows repo style (black/ruff)
- [ ] Licence header added to every new file
- [ ] Inline modification comment added where original files were touched
- [ ] All unit tests pass locally (`pytest -q`)
- [ ] CI passes on the PR
```
---

## 8. How to Add Comments & Indicate Modifications
When a contributor **modifies** an existing Kartezio file (e.g., `decoder.py` or `trainer.py`), they must:
1. Insert the **BEGIN/END modification block** (see § 5.2) **right before** the changed code.
2. Keep the original code **intact** (do not delete lines; add new statements or wrap existing ones).
3. Write a short explanatory sentence for future reviewers.

*Example – adding a timer field to `PopulationScore` in `population.py`:*
```python
# === BEGIN KARTEZIO‑MOO MODIFICATION ==============================
# File: src/kartezio/evolution/population.py
# Reason: expose per‑individual inference time for the ProcessTime metric.
# Change: added `time: np.ndarray[float32]` to PopulationScore.
# === END KARTEZIO‑MOO MODIFICATION ==============================
class PopulationScore:
    fitness: np.ndarray[float32]
    time: np.ndarray[float32]   # <-- new column, initialized in Trainer
    raw: np.ndarray[float32]
```
---

## 9. Quick Reference Cheat‑Sheet
| Task | Command |
|------|---------|
| Create develop branch (once) | `git checkout -b develop origin/main && git push -u origin develop` |
| Start feature work | `git checkout develop && git pull && git checkout -b feature/<area>` |
| Rebase onto latest develop | `git fetch && git rebase origin/develop` |
| Commit (standard) | `git commit -m "Alice - add ComplexityMetric base class"` |
| Push feature branch | `git push -u origin feature/<area>` |
| Open PR → target `develop` | (GitHub UI) |
| Merge PR (after review) | `git checkout develop && git merge --ff-only feature/<area>` |
| Create release branch | `git checkout develop && git checkout -b release/v1.1-moo` |
| Tag release | `git tag -a v1.1-moo -m "Multi‑objective NSGA‑II extension"` |
| Push tag | `git push origin v1.1-moo` |
| Merge to main | `git checkout main && git merge release/v1.1-moo && git push` |

---

## 10. Final Remarks
* All new code resides under `kartezio/moo/` – this guarantees **no accidental alteration** of the original Kartezio implementation.
* The **licence header** and **modification comment** together satisfy the licence’s requirement that modified files carry a prominent notice.
* By following the branching model, each of the four contributors can work **completely in parallel** without stepping on each other’s toes.

Happy coding! 🚀
