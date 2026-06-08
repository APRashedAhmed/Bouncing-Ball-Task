# Bouncing-Ball-Task — Repo Re-orientation & CLAUDE.md Audit

**Date:** 2026-05-13
**Author:** Claude (Opus 4.7)
**Scope:** Audit of `CLAUDE.md` accuracy and assessment of repo state after ~1 year of inactivity.

## Summary

The repo has substantial **uncommitted work** centered on a new `controlled_bouncing_ball/` module with its own test suite, scripts, and a notebook. `CLAUDE.md` (itself untracked) describes this feature but has drifted in several places — wrong test paths, undocumented parameters, undocumented fixtures, undocumented markers, and zero coverage of the rest of the source tree (`human_bouncing_ball/`, `model_bouncing_ball/`, `utils/`, `notebooks/`, `scripts/`). One real bug was found in `tests/conftest.py`. The feature itself looks coherent but only one variant (`no_change`) is implemented despite the framework being designed for more.

---

## A. CLAUDE.md Audit

`CLAUDE.md` is **directionally correct** but drifted. It appears to have been written midway through the controlled-dataset feature and never revisited.

### A.1 Inaccurate claims (need fixing)

| Claim | Actual state | Location |
|---|---|---|
| `tests/test_no_change_variant.py` exists | Actual file is `tests/controlled_variants/test_no_change.py` | CLAUDE.md:40 |
| `pytest -m "not slow"` example | `slow` marker is declared in `pyproject.toml`, not `conftest.py`'s `pytest_configure` (works, but two registries diverging) | CLAUDE.md:32, pyproject.toml:24 |
| "No old parameters: implementation no longer uses `total_dataset_length` or `num_blocks`" | Plausible but unverified — should grep `controlled_bouncing_ball/` to confirm. | CLAUDE.md:105 |

### A.2 Missing items (should be added)

**Repo structure (not mentioned at all):**
- `notebooks/` — 20+ Jupyter notebooks driving dataset generation across V1–V3.2.3 (the 4.x series is current).
- `scripts/` — contains `validate_no_change.py` (standalone validation runner).
- `logs/` — present at top level.
- `src/bouncing_ball_task/human_bouncing_ball/` — trial generation for human experiments (`bounce.py`, `catch.py`, `straight.py`, `nonwall.py`, `dataset.py`, `defaults.py`).
- `src/bouncing_ball_task/model_bouncing_ball/` — 6 model variants (`cc_vc.py`, `cc_nvc.py`, `cc_rvc.py`, `ncc_vc.py`, `ncc_nvc.py`, `ncc_rvc.py`) + orchestrator + defaults. **Entirely undocumented.** The cc/ncc and vc/nvc/rvc naming is opaque without a legend.
- `src/bouncing_ball_task/utils/` — `taskutils`, `htaskutils`, `visualize`, `gif`, `pyutils`, `logutils`, `types`.
- `tests/README.md` — exists, has marker examples and runner instructions.
- `tests/controlled_variants/` — subdirectory holding `test_no_change.py` and `base_variant_tests.py`.
- `tests/run_validation.py` — standalone runner.

**Controlled dataset details (incomplete):**
- `ControlledDatasetParameters` has more fields than documented (`defaults.py:54-71`):
  - documented: `num_base_sequences`, `seed`, `duration`, `variable_length`
  - **undocumented**: `pccnvc_lower`, `pccnvc_upper`, `pccovc_lower`, `pccovc_upper` (hazard-rate / contingency bounds — these are central experimental knobs)
- `ControlledTaskParameters` exists at `defaults.py:20-51` and is undocumented entirely.
- `dict_trial_type_generation_funcs` currently registers **only `no_change`**. CLAUDE.md should be explicit about which variants are implemented vs. planned (`sudden_change`, `gradual_change` appear as commented-out placeholder markers in `conftest.py`).

**Fixtures (incomplete list):**
- documented: `controlled_dataset_small`, `controlled_dataset_with_videos`, `multiple_variants_dataset`, `temp_output_dir`
- **undocumented**: `session_temp_dir`, `default_controlled_params`, `default_task_params`, `reset_random_state`. The last is `autouse=True` and seeds NumPy to 42 before every test — important to know.

**Pytest markers (incomplete list):**
- documented: `controlled_dataset`, `slow`
- **undocumented**: `variant`, `no_change` (defined in `conftest.py`); `integration`, `unit`, `dataset_generation` (defined in `pyproject.toml`).

### A.3 Proposed CLAUDE.md additions

Beyond the corrections above, three new sections:
1. **Repo map** — bullet list of top-level dirs and `src/bouncing_ball_task/` submodules, one sentence each. Biggest single gap.
2. **Variant status table** — `no_change: implemented`; any planned variants (`sudden_change`, `gradual_change`) explicitly marked as not-yet-implemented so future agents don't waste time looking for them.
3. **Model-module legend** — what `cc` / `ncc` / `vc` / `nvc` / `rvc` mean. Without this, `model_bouncing_ball/` is illegible.

---

## B. Repo State Assessment

### B.1 What's in flight (uncommitted)

The bulk of the uncommitted work is one coherent feature: **the controlled-dataset module + its tests.**

**New module (~737 LOC, untracked):**
- `src/bouncing_ball_task/controlled_bouncing_ball/{__init__.py, dataset.py, no_change.py, defaults.py}`
- Public API: `generate_controlled_dataset()`, `generate_controlled_dataset_with_videos()`
- One variant implemented (`no_change`).

**New tests (untracked):**
- `tests/conftest.py`, `tests/test_controlled_dataset_endproduct.py`, `tests/test_change_vector_generation.py`
- `tests/controlled_variants/test_no_change.py`, `tests/controlled_variants/base_variant_tests.py`
- `tests/README.md`, `tests/run_validation.py`

**New auxiliary:**
- `scripts/validate_no_change.py`
- `notebooks/4.4-Generating-V322-Dataset.py`

**Modified, small (3 files):**
- `pyproject.toml`: added `[tool.pytest.ini_options]` block (testpaths, markers).
- `src/bouncing_ball_task/human_bouncing_ball/dataset.py`: signature change to expose `output_model_samples` from `generate_video_dataset()`, plus a `.title()` cosmetic removal. The signature change is consumed by the new controlled module and by `conftest.py` (which unpacks 6 values).
- `src/bouncing_ball_task/utils/htaskutils.py`: `.title()` cosmetic removal.

**Untracked file:** `CLAUDE.md` itself.

### B.2 Bugs / rough edges

1. **`tests/conftest.py:155-164` — broken fixtures.** `default_controlled_params` and `default_task_params` instantiate `ControlledDatasetParameters()` and `ControlledTaskParameters()`, but neither class is imported. Any test that requests these fixtures will `NameError`. Likely never exercised — easy to miss until someone tries. Fix: add `from bouncing_ball_task.controlled_bouncing_ball.defaults import ControlledDatasetParameters, ControlledTaskParameters`.
2. **Test discoverability.** `tests/controlled_variants/` has no `__init__.py` and `base_variant_tests.py` doesn't start with `test_` (intentional — it's a base class). Worth a sanity check that pytest discovery isn't surprised.
3. **`.title()` cosmetic removals** in `human_bouncing_ball/dataset.py` and `utils/htaskutils.py` are tiny but imply a partial pass to preserve raw trial-type strings. Worth grepping for other `.title()` call sites that may still need the same treatment.
4. **No verification that tests actually pass yet.** They have the right shape, but nothing in this audit has run `pytest`. That's the next single most valuable thing to do.

### B.3 Recommended work, in order

**Tier 1 — get the in-flight work onto a green baseline (low risk, high value):**
1. Fix the `conftest.py` import bug (one-line fix).
2. Run the full test suite (`pytest -v`). Triage failures.
3. Update `CLAUDE.md` per section A (corrections + repo map + variant status table + model module legend).
4. Commit the controlled-dataset feature as a coherent unit (module + tests + `CLAUDE.md` + the `human_bouncing_ball`/`htaskutils` edits it depends on).

**Tier 2 — usability refactors (need input on scope):**
5. Document the `model_bouncing_ball/` module — at minimum, what cc/ncc/vc/nvc/rvc mean and which file is which (currently zero docstrings on the variant files).
6. Decide the fate of unimplemented variants (`sudden_change`, `gradual_change`). Either build them or strip placeholder references so the framework doesn't look half-done.
7. Consider whether `human_bouncing_ball/` and `model_bouncing_ball/` can share more infrastructure — they each have their own `dataset.py` and `defaults.py` with similar shape. (Speculative; would want a closer read first.)

**Tier 3 — opportunistic cleanups:**
8. The single TODO in `utils/gif.py` ("Add a border between each frame") — close it or delete it.
9. Confirm/remove any `total_dataset_length` / `num_blocks` references if they survive anywhere; CLAUDE.md claims they're gone.
10. Add a top-level README pointer or expand the existing README if it's thin.

### B.4 Open questions

Answers to these determine how aggressive the Tier-2 refactor should be:
- Is the controlled-dataset feature finished as-is (just `no_change`), or were more variants the plan that got abandoned?
- Are the notebooks (especially 4.4) part of the canonical workflow, or sandbox artifacts?
- Is the dissertation reproducibility target frozen, or are you still generating new datasets?

---

## Critical files referenced

- `CLAUDE.md` (untracked, needs update)
- `tests/conftest.py:155-164` (bug)
- `src/bouncing_ball_task/controlled_bouncing_ball/defaults.py:20-71` (parameter classes)
- `src/bouncing_ball_task/controlled_bouncing_ball/dataset.py` (public API surface)
- `pyproject.toml:17-28` (pytest config)
- `src/bouncing_ball_task/human_bouncing_ball/dataset.py` (modified, signature change)

## Verification (once we execute)

- `pytest` from `Bouncing-Ball-Task/` — all collected tests pass.
- `pytest -m controlled_dataset` — runs only the controlled-dataset slice.
- `pytest tests/controlled_variants/test_no_change.py -v` — variant-specific tests pass.
- `python scripts/validate_no_change.py` — standalone validator runs clean.
- Sanity-check: updated `CLAUDE.md` repo map matches `ls src/bouncing_ball_task/`.
