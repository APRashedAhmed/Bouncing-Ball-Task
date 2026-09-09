"""Deterministic reproducibility tests for the seeding fix (audit P0-1/P0-2/P0-3).

These tests are seeded and deterministic by design; they join the controlled
green floor. They must NOT depend on the chronically-red statistical suite.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from bouncing_ball_task.bouncing_ball import BouncingBallTask
from bouncing_ball_task.utils import pyutils


# Minimal task parameters sufficient to construct a task that draws initial
# conditions. Batch size > 1 so resample_change_probabilities actually draws.
# sequence_mode="static" produces a fixed sequence and populates task.samples
# after one iteration pass; sample_mode="parameter" (default) keeps samples
# as lightweight parameter tuples.
MINIMAL_TASK_PARAMS = dict(
    size_frame=(256, 256),
    sequence_length=30,
    ball_radius=20,
    batch_size=4,
    sequence_mode="static",
    target_future_timestep=1,
)


def _make_task(**overrides):
    params = copy.deepcopy(MINIMAL_TASK_PARAMS)
    params.update(overrides)
    task = BouncingBallTask(**params)
    list(task)  # run one pass so task.samples is populated
    return task


def test_explicit_seed_is_deterministic():
    """Two tasks built with the same explicit seed produce identical samples."""
    task_a = _make_task(seed=12345)
    task_b = _make_task(seed=12345)
    assert np.array_equal(np.asarray(task_a.samples), np.asarray(task_b.samples))


def test_different_explicit_seeds_differ():
    """A different explicit seed produces different samples (sanity check)."""
    task_a = _make_task(seed=12345)
    task_b = _make_task(seed=54321)
    assert not np.array_equal(np.asarray(task_a.samples), np.asarray(task_b.samples))


def test_resolved_seed_is_the_explicit_seed():
    """An explicitly-seeded task records that exact seed on resolved_seed."""
    task = _make_task(seed=777)
    assert task.resolved_seed == 777


def test_resolved_seed_populated_when_seed_none():
    """With seed=None the task self-draws and reports the drawn integer."""
    task = _make_task(seed=None)
    assert isinstance(task.resolved_seed, int)


def test_seed_false_defers_and_resolves_to_none():
    """seed=False is the external-seeding escape hatch: the task does not seed
    and reports resolved_seed is None."""
    np.random.seed(2024)
    task = _make_task(seed=False)
    assert task.resolved_seed is None


def test_initial_rng_state_captured_pre_draw():
    """E1 option (a): initial_rng_state is the POST-seed, PRE-draw RNG state. For
    an explicit seed it must equal the state set_global_seed(seed) leaves before
    any draw — a post-draw capture would not (its position index/key advance)."""
    task = _make_task(seed=777)
    pyutils.set_global_seed(777)
    expected = np.random.get_state()
    rec = task.initial_rng_state
    assert rec[0] == expected[0]                  # 'MT19937'
    assert np.array_equal(rec[1], expected[1])    # 624-word state vector
    assert rec[2] == expected[2]                  # position index


# Pinned concrete first-sample for BouncingBallTask(seed=777) under the FIXED
# code (P0-1-MIG behavioral-break audit record, dynamic-12). This literal IS the
# fixed code's seed=777 first sample; the pre-fix code cannot reproduce it, so it
# is the durable, machine-checkable record of the P0-1 seed-order behavioral break.
PINNED_FIRST_SAMPLE_SEED_777 = np.array([  # pinned for P0-1 seed-order fix
    [210.88484 ,  40.168625, 255.      ,   0.      ,   0.      ],
    [211.12007 ,  40.39888 , 255.      ,   0.      ,   0.      ],
    [211.35532 ,  40.629135, 255.      ,   0.      ,   0.      ],
    [211.59055 ,  40.85939 , 255.      ,   0.      ,   0.      ],
    [211.82579 ,  41.089645, 255.      ,   0.      ,   0.      ],
    [212.06102 ,  41.3199  , 255.      ,   0.      ,   0.      ],
    [212.29626 ,  41.550156, 255.      ,   0.      ,   0.      ],
    [212.5315  ,  41.780415, 255.      ,   0.      ,   0.      ],
    [212.76674 ,  42.01067 , 255.      ,   0.      ,   0.      ],
    [213.00197 ,  42.240925, 255.      ,   0.      ,   0.      ],
    [213.23721 ,  42.47118 , 255.      ,   0.      ,   0.      ],
    [213.47244 ,  42.701435, 255.      ,   0.      ,   0.      ],
    [213.70769 ,  42.93169 , 255.      ,   0.      ,   0.      ],
    [213.94292 ,  43.161945, 255.      ,   0.      ,   0.      ],
    [214.17816 ,  43.3922  , 255.      ,   0.      ,   0.      ],
    [214.41339 ,  43.62246 , 255.      ,   0.      ,   0.      ],
    [214.64864 ,  43.852715, 255.      ,   0.      ,   0.      ],
    [214.88387 ,  44.08297 , 255.      ,   0.      ,   0.      ],
    [215.11911 ,  44.313225, 255.      ,   0.      ,   0.      ],
    [215.35434 ,  44.54348 , 255.      ,   0.      ,   0.      ],
    [215.58958 ,  44.773735, 255.      ,   0.      ,   0.      ],
    [215.82481 ,  45.00399 , 255.      ,   0.      ,   0.      ],
    [216.06006 ,  45.23425 , 255.      ,   0.      ,   0.      ],
    [216.29529 ,  45.464504, 255.      ,   0.      ,   0.      ],
    [216.53053 ,  45.69476 , 255.      ,   0.      ,   0.      ],
    [216.76576 ,  45.925014, 255.      ,   0.      ,   0.      ],
    [217.001   ,  46.15527 , 255.      ,   0.      ,   0.      ],
    [217.23624 ,  46.385525, 255.      ,   0.      ,   0.      ],
    [217.47148 ,  46.61578 , 255.      ,   0.      ,   0.      ],
], dtype=np.float32)


def test_fixed_seed_pins_concrete_first_sample():
    """Pin a concrete numerical output for a fixed seed (P0-1-MIG audit trail).

    The existing controlled-floor tests assert only *structural* invariants
    (cross-trial consistency, color/count correctness), so the P0-1 seed-order
    reorder — which changes the actual output for any given seed — leaves none
    of them failing and produces no commit-time record of the behavioral break.
    This test pins a real array value so the break IS recorded: the expected
    value in PINNED_FIRST_SAMPLE_SEED_777 is the output of the *fixed* code (seed
    applied before the first draw). It will not match any value produced by the
    pre-fix code, so this test is the machine-checkable behavioral-change marker
    for P0-1-MIG.

    PINNED_FIRST_SAMPLE_SEED_777 above is the real pinned literal (not a
    placeholder), so this is a live green guard. It is NOT a self-referential
    comparison of the just-computed value against itself: it asserts against a
    module-level constant, so it fails if the fixed code's output ever drifts.
    """
    task = _make_task(seed=777)
    first = np.asarray(task.samples)[0]
    np.testing.assert_array_equal(first, PINNED_FIRST_SAMPLE_SEED_777)


from bouncing_ball_task.color_controlled_bouncing_ball import dataset as cds
from bouncing_ball_task.color_controlled_bouncing_ball import defaults as cdefaults


def _controlled_params(seed):
    dataset_params = {
        key: getattr(cdefaults.ColorControlledDatasetParameters(), key)
        for key in cdefaults.ColorControlledDatasetParameters.keys
    }
    dataset_params["num_base_sequences"] = 3
    dataset_params["seed"] = seed
    task_params = {
        key: getattr(cdefaults.ColorControlledTaskParameters(), key)
        for key in cdefaults.ColorControlledTaskParameters.keys
    }
    return dataset_params, task_params


def test_controlled_seed_is_deterministic():
    """Same controlled seed -> identical samples across two full generations."""
    dp, tp = _controlled_params(seed=4242)
    _, samples_a, _, _, _, _ = cds.generate_color_controlled_dataset(dp, tp, shuffle=False)
    _, samples_b, _, _, _, _ = cds.generate_color_controlled_dataset(dp, tp, shuffle=False)
    assert np.array_equal(samples_a, samples_b)


def test_controlled_resolved_seed_recorded():
    """The metadata records the resolved seed actually used (P0-2 + P0-3)."""
    dp, tp = _controlled_params(seed=4242)
    _, _, _, _, _, meta = cds.generate_color_controlled_dataset(dp, tp, shuffle=False)
    assert meta["resolved_seed"] == 4242


def test_controlled_unseeded_records_real_seed_not_zero():
    """An unseeded (seed=None) controlled run records the *drawn* seed, never a
    false 0 (P0-3). The drawn seed is a real integer and is almost surely != 0."""
    dp, tp = _controlled_params(seed=None)
    _, _, _, _, _, meta = cds.generate_color_controlled_dataset(dp, tp, shuffle=False)
    assert isinstance(meta["resolved_seed"], int)
    # name must embed the resolved seed, not the literal 0
    assert str(meta["resolved_seed"]) in meta["name"]


def test_controlled_seed_false_name_has_no_false_zero():
    """seed=False escape hatch: the task resolves no seed (resolved_seed is None),
    so metadata records resolved_seed=None and the dataset name must NOT embed a
    false `0` (P0-3 / right-and-works-6). Caller seeds the global RNG first."""
    np.random.seed(2024)
    dp, tp = _controlled_params(seed=False)
    _, _, _, _, _, meta = cds.generate_color_controlled_dataset(dp, tp, shuffle=False)
    assert meta["resolved_seed"] is None
    # the old `get('seed', 0)` bug baked a literal 0 into the name; the fixed code
    # must reflect the real (un-resolved) escape-hatch value instead. NOTE: in
    # Python `False == 0`, so assert identity to False — `!= 0` would be False for
    # False and give a spurious failure.
    assert meta["seed"] is False
    assert meta["name"].endswith(str(meta["seed"]))


from bouncing_ball_task.human_bouncing_ball import dataset as hds
from bouncing_ball_task.human_bouncing_ball import defaults as hdefaults


def _human_params(seed):
    dataset_params = {
        key: getattr(hdefaults.HumanDatasetParameters(), key)
        for key in hdefaults.HumanDatasetParameters.keys
    }
    dataset_params["seed"] = seed
    dataset_params["total_videos"] = 12  # small but non-trivial
    task_params = {
        key: getattr(hdefaults.TaskParameters(), key)
        for key in hdefaults.TaskParameters.keys
    }
    return dataset_params, task_params


def _samples_equal(sa, sb):
    """Compare two lists of variable-length sample arrays for element-wise equality."""
    if len(sa) != len(sb):
        return False
    return all(np.array_equal(a, b) for a, b in zip(sa, sb))


def test_human_seed_is_deterministic():
    """Same human seed -> identical output_samples across two generations.

    Invariant guard; the distinguishing P0-2 check is test_human_pipeline_seeds_once."""
    dp, tp = _human_params(seed=99)
    out_a = hds.generate_video_dataset(
        dp, tp, hds.dict_trial_type_generation_funcs, _adjust_labels=False,
        validate=False,
    )
    out_b = hds.generate_video_dataset(
        dp, tp, hds.dict_trial_type_generation_funcs, _adjust_labels=False,
        validate=False,
    )
    assert _samples_equal(out_a[1], out_b[1])


@pytest.mark.slow
def test_human_seed_is_deterministic_production_path():
    """Determinism on the real production path (_adjust_labels=True, validate=True).

    Invariant guard; the distinguishing P0-2 check is test_human_pipeline_seeds_once.

    Runs at total_videos=60, which exercises the trial-type batch-of-3
    composition that the E4 set_color_parameters bug crashed on (clean at 30,
    crashed at >= 60). Post-E4 the path runs clean and deterministically. This
    test was previously @pytest.mark.skip-ped citing E4 (escalations.md)."""
    dp, tp = _human_params(seed=99)
    dp_large = dict(dp)
    dp_large["total_videos"] = 60  # exercises the previously-crashing E4 path
    out_a = hds.generate_video_dataset(
        dp_large, tp, hds.dict_trial_type_generation_funcs, _adjust_labels=True,
        validate=True,
    )
    out_b = hds.generate_video_dataset(
        dp_large, tp, hds.dict_trial_type_generation_funcs, _adjust_labels=True,
        validate=True,
    )
    assert _samples_equal(out_a[1], out_b[1])


from bouncing_ball_task.model_bouncing_ball import dataset as mds
from bouncing_ball_task.model_bouncing_ball import defaults as mdefaults
from bouncing_ball_task.model_bouncing_ball.ncc_nvc import generate_ncc_nvc_trials
from bouncing_ball_task.model_bouncing_ball.cc_nvc import generate_cc_nvc_trials
from bouncing_ball_task.model_bouncing_ball.ncc_vc import generate_ncc_vc_trials
from bouncing_ball_task.model_bouncing_ball.cc_vc import generate_cc_vc_trials
from bouncing_ball_task.model_bouncing_ball.ncc_rvc import generate_ncc_rvc_trials
from bouncing_ball_task.model_bouncing_ball.cc_rvc import generate_cc_rvc_trials

_MODEL_TRIAL_FUNCS = {
    "ncc_nvc": generate_ncc_nvc_trials,
    "cc_nvc": generate_cc_nvc_trials,
    "ncc_vc": generate_ncc_vc_trials,
    "cc_vc": generate_cc_vc_trials,
    "ncc_rvc": generate_ncc_rvc_trials,
    "cc_rvc": generate_cc_rvc_trials,
}


@pytest.mark.slow
def test_model_seed_is_deterministic():
    """Model pipeline inherits the human seeding path; same seed -> identical.

    Invariant guard; the distinguishing P0-2 check is test_human_pipeline_seeds_once.

    Calls hds.generate_video_dataset directly with model trial funcs and model
    defaults (the same seeding path generate_model_dataset_nongray uses) with
    _adjust_labels=False/validate=False, exercising the full seeding-inheritance
    path (seed flows from NongrayDatasetParameters through
    generate_video_parameters into per-trial BouncingBallTask constructions).
    Note: as of E5 (estimate-size decoupling) generate_model_dataset_nongray
    itself runs to completion with adjusted labels; its end-to-end determinism is
    covered by test_model_nongray_deterministic_separate_processes."""
    dataset_params = {
        key: getattr(mdefaults.NongrayDatasetParameters(), key)
        for key in mdefaults.NongrayDatasetParameters.keys
    }
    dataset_params["seed"] = 7
    dataset_params["total_videos"] = 12
    task_params = {
        key: getattr(mdefaults.TaskParameters(), key)
        for key in mdefaults.TaskParameters.keys
    }
    out_a = hds.generate_video_dataset(
        dataset_params, task_params, _MODEL_TRIAL_FUNCS,
        _adjust_labels=False, validate=False, defaults=mdefaults,
    )
    out_b = hds.generate_video_dataset(
        dataset_params, task_params, _MODEL_TRIAL_FUNCS,
        _adjust_labels=False, validate=False, defaults=mdefaults,
    )
    assert _samples_equal(out_a[1], out_b[1])


def test_human_pipeline_seeds_once(monkeypatch):
    """P0-2 validator that DISTINGUISHES fixed from unfixed code. With the
    task_parameters['seed']=False fix, the global RNG is seeded EXACTLY ONCE per
    generate_video_dataset (the single line-247 anchor in generate_video_parameters);
    per-trial-type and preset tasks defer. Without the fix each task self-seeds with
    seed=None, so set_global_seed is called many times (1 anchor + N per-trial-type
    tasks + 1 preset = N+2 for N active trial types). NOTE: the same-seed determinism
    tests cannot catch this regression because set_global_seed also seeds Python's
    `random`, making the unfixed per-trial reseeds deterministic — so this
    call-count assertion is the real P0-2-human guard.

    Concrete unfixed count for total_videos=12: 3 active trial types (catch,
    straight, bounce) + 1 preset task + 1 anchor = 5 calls. With the fix: 1."""
    import bouncing_ball_task.utils.pyutils as _pyu
    calls = {"n": 0}
    real = _pyu.set_global_seed

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_pyu, "set_global_seed", _counting)
    dp, tp = _human_params(seed=99)
    hds.generate_video_dataset(
        dp, tp, hds.dict_trial_type_generation_funcs,
        _adjust_labels=False, validate=False,
    )
    assert calls["n"] == 1, f"expected 1 seeding (single anchor), got {calls['n']}"


from bouncing_ball_task.utils import pyutils


def test_capture_provenance_structure():
    """Provenance dict carries git SHA, library versions, and RNG state."""
    prov = pyutils.capture_provenance()
    assert "git_sha" in prov
    assert prov["git_sha"] is None or isinstance(prov["git_sha"], str)
    assert "git_dirty" in prov
    assert prov["git_dirty"] is None or isinstance(prov["git_dirty"], bool)
    assert "library_versions" in prov
    assert "numpy" in prov["library_versions"]
    assert isinstance(prov["library_versions"]["numpy"], str)
    assert "numpy_rng_state" in prov
    assert prov["numpy_rng_state"][0] == "MT19937"


def test_capture_provenance_rng_state_roundtrips():
    """The captured RNG state can be restored to reproduce subsequent draws."""
    np.random.seed(123)
    prov = pyutils.capture_provenance()
    expected = np.random.random(5)
    np.random.set_state(prov["numpy_rng_state"])
    actual = np.random.random(5)
    assert np.array_equal(expected, actual)


def test_capture_provenance_marks_dirty_tree(tmp_path):
    """A dirty working tree must surface in provenance (dynamic-13): git_sha is
    suffixed '-dirty' and git_dirty is True, so a dataset built from
    uncommitted source is never attributed to a clean commit.

    This exercises capture_provenance's OWN return value against a throwaway
    repo via the repo_dir parameter."""
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.check_call(["git", "init", "-q"], cwd=repo)
    sp.check_call(["git", "config", "user.email", "t@t"], cwd=repo)
    sp.check_call(["git", "config", "user.name", "t"], cwd=repo)
    (repo / "f.txt").write_text("one")
    sp.check_call(["git", "add", "-A"], cwd=repo)
    # Conventional-commit subject: the machine may have a global commit-msg hook
    # (core.hooksPath) that rejects non-conventional subjects like "init".
    sp.check_call(["git", "commit", "-qm", "chore: init"], cwd=repo)
    (repo / "f.txt").write_text("two")  # uncommitted edit -> dirty tree

    prov = pyutils.capture_provenance(repo_dir=repo)
    assert prov["git_sha"].endswith("-dirty")
    assert prov["git_dirty"] is True


def test_capture_provenance_uses_supplied_rng_state():
    """E1 option (a): when an rng_state is supplied, capture_provenance records
    THAT state verbatim (the pre-draw state threaded by callers), not a freshly
    sampled live one. Advancing the global RNG after capturing `supplied` must
    not change what the helper records."""
    np.random.seed(555)
    supplied = np.random.get_state()
    np.random.random(10)  # advance the live global state so it differs
    prov = pyutils.capture_provenance(rng_state=supplied)
    assert prov["numpy_rng_state"][2] == supplied[2]
    assert np.array_equal(prov["numpy_rng_state"][1], supplied[1])


def test_controlled_metadata_has_provenance():
    dp, tp = _controlled_params(seed=4242)
    _, _, _, _, _, meta = cds.generate_color_controlled_dataset(dp, tp, shuffle=False)
    assert "provenance" in meta
    assert "git_sha" in meta["provenance"]
    assert "numpy" in meta["provenance"]["library_versions"]


def test_controlled_provenance_rng_state_is_pre_draw():
    """E1 option (a): the recorded provenance RNG state is the PRE-draw state
    (right after set_global_seed, before any initial-condition draw), so
    restoring it replays the trajectory. It must equal the state
    set_global_seed(seed) leaves before any draw; a POST-draw state would not."""
    dp, tp = _controlled_params(seed=4242)
    _, _, _, _, _, meta = cds.generate_color_controlled_dataset(dp, tp, shuffle=False)
    recorded = meta["provenance"]["numpy_rng_state"]
    pyutils.set_global_seed(4242)
    expected = np.random.get_state()
    assert recorded[0] == expected[0]                  # 'MT19937'
    assert np.array_equal(recorded[1], expected[1])    # 624-word state vector
    assert recorded[2] == expected[2]                  # position index


def test_human_metadata_has_provenance():
    dp, tp = _human_params(seed=99)
    out = hds.generate_video_dataset(
        dp, tp, hds.dict_trial_type_generation_funcs, _adjust_labels=False,
        validate=False,
    )
    meta = out[5]
    assert "provenance" in meta
    assert "git_sha" in meta["provenance"]
    assert "numpy" in meta["provenance"]["library_versions"]


def test_controlled_pipeline_seeds_once(monkeypatch):
    """E2 hygiene: the controlled pipeline must seed the global RNG exactly ONCE
    — at the base-task anchor — even with shuffle=True (the default). The two
    preset BouncingBallTask wrappers must pass seed=False so they do NOT reseed
    mid-pipeline; the shuffle's np.random.permutation then draws from the single
    seed anchor intentionally rather than relying on set_global_seed(None)
    coincidentally re-deriving a deterministic seed from the already-seeded
    Python `random`.

    STRUCTURAL guard (call-count spy), NOT an output-determinism test: output
    determinism holds with OR without the fix (set_global_seed seeds Python's
    random too), so only the seeds-once property distinguishes the fix. Cf.
    test_human_pipeline_seeds_once (the P0-2 analog).
    """
    from bouncing_ball_task.utils import pyutils

    calls = []
    real_set_global_seed = pyutils.set_global_seed

    def spy(value=None):
        calls.append(value)
        return real_set_global_seed(value)

    monkeypatch.setattr(pyutils, "set_global_seed", spy)

    dp, tp = _controlled_params(seed=4242)
    cds.generate_color_controlled_dataset(dp, tp, shuffle=True)

    assert calls == [4242], (
        f"expected exactly one global seeding (the base-task anchor 4242); got "
        f"{calls}. A preset BouncingBallTask construction is reseeding the global "
        f"RNG mid-pipeline — pass seed=False to the preset wrappers."
    )


from bouncing_ball_task.constants import DEFAULT_COLORS


def test_set_color_parameters_batch_of_three_no_crash():
    """E4 (C1): a batch of EXACTLY 3 in-set triples (len 3) must not raise
    AttributeError. Pre-fix this hit `len(initial_color) == 3` -> the ndim>1 arm
    -> `(list == list).all(axis=1)` -> 'bool' has no attribute 'all'.

    The batch MUST be passed the way the human production path supplies it — a
    Python list of per-trial RGB triples — so that `initial_color[0] ==
    valid_colors` is Python `list.__eq__` (returns a scalar bool -> the crash).
    An ndarray batch would broadcast element-wise and silently NOT reproduce the
    bug (verified: ndarray rows never crash pre-fix). The end-to-end production
    reproduction is the total_videos>=60 run in
    test_human_seed_is_deterministic_production_path."""
    task = _make_task(seed=4242)
    vc = np.asarray(DEFAULT_COLORS)                 # (3, 3): R, G, B reference
    # A Python list of exactly 3 in-set RGB triples (production-faithful shape).
    batch3 = [list(c) for c in DEFAULT_COLORS]      # [[255,0,0],[0,255,0],[0,0,255]]
    # Must not raise:
    _, out_vc, out_n = task.set_color_parameters(batch3, list(DEFAULT_COLORS), len(vc))
    assert np.array_equal(out_vc, vc)
    assert out_n == len(vc)


def test_set_color_parameters_noop_for_in_set_colors():
    """E4 G2 no-op invariant (C2): for color batches already in valid_colors the
    repaired arm fires NO prepend, so valid_colors/num_colors are returned
    UNCHANGED. For every working path pre-fix produced the same unchanged set
    (it skipped the arm for non-size-3 batches and crashed for size-3), so
    'unchanged' IS the byte-exact pre-fix result. Covers single triple (ndim=1)
    and batches (ndim=2, including size 3). For the ndim=1 case it also asserts
    the returned initial_color is normalised to 2-D (1, 3) — an intentional,
    documented shape change that no current production caller exercises."""
    task = _make_task(seed=4242)
    vc = np.asarray(DEFAULT_COLORS)
    n = len(vc)

    # ndim=2, batch of exactly 3 (the E4 crash shape) — Python list, as the
    # production path supplies it (an ndarray batch would not reproduce the bug).
    batch3_list = [list(c) for c in DEFAULT_COLORS]
    _, out_vc, out_n = task.set_color_parameters(batch3_list, list(DEFAULT_COLORS), n)
    assert np.array_equal(out_vc, vc) and out_n == n

    # ndim=2, a non-3 batch size (general no-op)
    _, out_vc5, out_n5 = task.set_color_parameters(vc[[0, 1, 2, 0, 1]], list(DEFAULT_COLORS), n)
    assert np.array_equal(out_vc5, vc) and out_n5 == n

    # ndim=1, a single in-set triple. The repaired arm normalises the returned
    # initial_color to 2-D (np.atleast_2d at the index block), so a single (3,)
    # triple comes back as a (1, 3) row. valid_colors/num_colors stay the
    # unchanged in-set set (no prepend fires). The shape change is asserted here
    # as intentional and documented; no current production caller passes ndim==1
    # (all production paths pass a 2-D batch -> identity, no shape change).
    out_ic1, out_vc1, out_n1 = task.set_color_parameters(vc[0], list(DEFAULT_COLORS), n)
    assert np.array_equal(out_vc1, vc) and out_n1 == n
    assert np.asarray(out_ic1).ndim == 2 and np.asarray(out_ic1).shape == (1, 3)


def _assert_default_color_set(task):
    """C2 byte-exact no-op proof on a PRODUCTION-PATH run: the returned task's
    color set must be the unchanged in-set DEFAULT_COLORS — proving NO prepend
    fired on this exact path (a fired prepend would make valid_colors (4, 3) /
    num_colors 4). generate_video_dataset and generate_color_controlled_dataset both
    return the BouncingBallTask as element [0]; it stores .valid_colors and
    .num_colors from set_color_parameters (bouncing_ball.py:289-292)."""
    vc = np.asarray(DEFAULT_COLORS)
    assert np.asarray(task.valid_colors).shape == (3, 3), \
        f"valid_colors shape changed: {np.asarray(task.valid_colors).shape} (a prepend fired)"
    assert np.array_equal(np.asarray(task.valid_colors), vc), \
        "valid_colors differ from DEFAULT_COLORS (a prepend fired on the production path)"
    assert task.num_colors == 3, f"num_colors changed to {task.num_colors} (a prepend fired)"


def test_color_paths_complete_and_deterministic_post_e4():
    """E4 (C2 path coverage): the three working paths each complete without the
    E4 crash, are deterministic at the named configs, AND each returns the
    byte-exact unchanged in-set color set (valid_colors == DEFAULT_COLORS,
    num_colors == 3) — directly proving no prepend fired on those exact
    production paths (the C2 no-op-vs-pre-fix invariant). The controlled path is
    a DISTINCT asserted signal here (not only via test_no_change.py)."""
    # (a) human total_videos=30 seed=99
    dp, tp = _human_params(seed=99)
    dp30 = dict(dp); dp30["total_videos"] = 30

    # C2 premise guard (converts the design-spec batch-size table prose into a
    # machine-checkable assertion): at n=30 seed=99 NO per-trial-type batch is
    # exactly 3, so the size-3 pre-fix arm was never entered and 'unchanged' is
    # provably the byte-exact pre-fix result. generate_video_parameters returns
    # dict_params keyed by trial type; len(params) is that trial type's batch
    # size (== task_parameters_type["batch_size"], dataset.py:130).
    dict_params30, _ = hds.generate_video_parameters(
        **dp30, dict_trial_type_generation_funcs=hds.dict_trial_type_generation_funcs)
    batch_sizes30 = [len(params) for params in dict_params30.values()]
    assert all(bs != 3 for bs in batch_sizes30), \
        f"a per-trial-type batch of 3 at n=30 seed=99 would enter the pre-fix arm: {batch_sizes30}"

    h_a = hds.generate_video_dataset(dp30, tp, hds.dict_trial_type_generation_funcs,
                                     _adjust_labels=False, validate=False)
    h_b = hds.generate_video_dataset(dp30, tp, hds.dict_trial_type_generation_funcs,
                                     _adjust_labels=False, validate=False)
    assert _samples_equal(h_a[1], h_b[1])
    _assert_default_color_set(h_a[0])   # no prepend fired on the human n=30 path

    # (b) model total_videos=12 seed=7 (same seeding path as the wrapper)
    mp = {k: getattr(mdefaults.NongrayDatasetParameters(), k)
          for k in mdefaults.NongrayDatasetParameters.keys}
    mp["seed"] = 7; mp["total_videos"] = 12
    mtp = {k: getattr(mdefaults.TaskParameters(), k)
           for k in mdefaults.TaskParameters.keys}
    m_a = hds.generate_video_dataset(mp, mtp, _MODEL_TRIAL_FUNCS,
                                     _adjust_labels=False, validate=False, defaults=mdefaults)
    m_b = hds.generate_video_dataset(mp, mtp, _MODEL_TRIAL_FUNCS,
                                     _adjust_labels=False, validate=False, defaults=mdefaults)
    assert _samples_equal(m_a[1], m_b[1])
    _assert_default_color_set(m_a[0])   # no prepend fired on the model path

    # (c) controlled num_base_sequences=3 seed=4242
    cdp, ctp = _controlled_params(seed=4242)
    c_a = cds.generate_color_controlled_dataset(cdp, ctp, shuffle=False)
    c_b = cds.generate_color_controlled_dataset(cdp, ctp, shuffle=False)
    assert np.array_equal(c_a[1], c_b[1])
    _assert_default_color_set(c_a[0])   # no prepend fired on the controlled path


def _hz_df():
    """Minimal df_data the estimator's groupby('Hazard Rate') needs."""
    return pd.DataFrame({
        "Hazard Rate": [0.1, 0.1, 0.5, 0.5],
        "PCCNVC_effective": [0.20, 0.30, 0.60, 0.40],
    })


def test_estimate_n_none_preserves_human_sizing(monkeypatch):
    """C7: estimate_n=None keeps the human sizing (total_dataset_length *= estimate_mult)."""
    seen = {}

    def spy(dataset_parameters, *args, **kwargs):
        seen["tdl"] = dataset_parameters["total_dataset_length"]
        return ("t", [], [], [], _hz_df(), {})

    monkeypatch.setattr(hds, "generate_video_dataset", spy)
    dp, tp = _human_params(seed=99)
    base = dp["total_dataset_length"]
    assert base is not None  # human params size by total_dataset_length
    hds.estimate_effective_hazard_rates(
        dp, tp, hds.dict_trial_type_generation_funcs, estimate_mult=3)
    assert seen["tdl"] == base * 3


def test_estimate_n_sizes_by_total_videos(monkeypatch):
    """C7/3a: estimate_n provided sizes by total_videos and leaves
    total_dataset_length untouched (so None *= mult never runs)."""
    seen = {}

    def spy(dataset_parameters, *args, **kwargs):
        seen["tdl"] = dataset_parameters["total_dataset_length"]
        seen["tv"] = dataset_parameters["total_videos"]
        return ("t", [], [], [], _hz_df(), {})

    monkeypatch.setattr(hds, "generate_video_dataset", spy)
    dp = {k: getattr(mdefaults.NongrayDatasetParameters(), k)
          for k in mdefaults.NongrayDatasetParameters.keys}   # total_dataset_length=None
    tp = {k: getattr(mdefaults.TaskParameters(), k)
          for k in mdefaults.TaskParameters.keys}
    hds.estimate_effective_hazard_rates(
        dp, tp, _MODEL_TRIAL_FUNCS, defaults=mdefaults, estimate_n=500)
    assert seen["tdl"] is None     # untouched
    assert seen["tv"] == 500


import subprocess
import sys
import textwrap


def test_model_nongray_invokes_label_adjustment(monkeypatch):
    """C5: generate_model_dataset_nongray completes and invokes
    adjust_dataset_labels (effective-hazard-rate labels are applied). Uses a
    small estimate_n (patched) and small total_videos for speed."""
    monkeypatch.setattr(mdefaults, "ESTIMATE_N", 24)  # small for the test

    calls = {"n": 0}
    real = hds.adjust_dataset_labels

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(hds, "adjust_dataset_labels", spy)

    dp = {k: getattr(mdefaults.NongrayDatasetParameters(), k)
          for k in mdefaults.NongrayDatasetParameters.keys}
    dp["seed"] = 7
    dp["total_videos"] = 12
    tp = {k: getattr(mdefaults.TaskParameters(), k)
          for k in mdefaults.TaskParameters.keys}

    out = mds.generate_model_dataset_nongray(dp, tp)
    assert out is not None
    assert calls["n"] == 1, f"expected adjust_dataset_labels once, got {calls['n']}"


_MODEL_DET_SNIPPET = textwrap.dedent('''
    import numpy as np, hashlib
    from bouncing_ball_task.model_bouncing_ball import dataset as mds
    from bouncing_ball_task.model_bouncing_ball import defaults as mdefaults
    mdefaults.ESTIMATE_N = 24  # small, for speed
    dp = {k: getattr(mdefaults.NongrayDatasetParameters(), k)
          for k in mdefaults.NongrayDatasetParameters.keys}
    dp["seed"] = 7
    dp["total_videos"] = 12
    tp = {k: getattr(mdefaults.TaskParameters(), k)
          for k in mdefaults.TaskParameters.keys}
    out = mds.generate_model_dataset_nongray(dp, tp)
    h = hashlib.sha256()
    for s in out[1]:
        h.update(np.asarray(s).tobytes())
    print("DIGEST=" + h.hexdigest())
''')


@pytest.mark.slow
def test_model_nongray_deterministic_separate_processes():
    """C6: two same-seed runs of generate_model_dataset_nongray in FRESH
    interpreters produce identical sample hashes. Separate processes defeat the
    set_global_seed(None) coincidental-determinism trap (pyutils.py:36-37) that
    a same-process output-equality check would false-pass."""
    def run():
        # The pipeline prints a summary to stdout; pick out the sentinel line.
        out = subprocess.check_output(
            [sys.executable, "-c", _MODEL_DET_SNIPPET], text=True)
        digests = [l[len("DIGEST="):] for l in out.splitlines()
                   if l.startswith("DIGEST=")]
        assert len(digests) == 1, out
        return digests[0]

    first = run()
    second = run()
    assert first == second and len(first) == 64
