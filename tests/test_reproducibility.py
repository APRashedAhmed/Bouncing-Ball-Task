"""Deterministic reproducibility tests for the seeding fix (audit P0-1/P0-2/P0-3).

These tests are seeded and deterministic by design; they join the controlled
green floor. They must NOT depend on the chronically-red statistical suite.
"""
import copy

import numpy as np
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
# code (P0-1-MIG behavioral-break audit record, dynamic-12). This is a
# PLACEHOLDER that the implementer MUST replace — the test is intentionally RED
# until the real post-fix array is pasted here (see Step note below).
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

    This test is intentionally RED until the implementer pastes the real
    post-fix array into PINNED_FIRST_SAMPLE_SEED_777 above (placeholder is an
    empty array, which never matches the real first sample). It is NOT a
    self-referential comparison of the just-computed value against itself.
    """
    task = _make_task(seed=777)
    first = np.asarray(task.samples)[0]
    np.testing.assert_array_equal(first, PINNED_FIRST_SAMPLE_SEED_777)


from bouncing_ball_task.controlled_bouncing_ball import dataset as cds
from bouncing_ball_task.controlled_bouncing_ball import defaults as cdefaults


def _controlled_params(seed):
    dataset_params = {
        key: getattr(cdefaults.ControlledDatasetParameters(), key)
        for key in cdefaults.ControlledDatasetParameters.keys
    }
    dataset_params["num_base_sequences"] = 3
    dataset_params["seed"] = seed
    task_params = {
        key: getattr(cdefaults.ControlledTaskParameters(), key)
        for key in cdefaults.ControlledTaskParameters.keys
    }
    return dataset_params, task_params


def test_controlled_seed_is_deterministic():
    """Same controlled seed -> identical samples across two full generations."""
    dp, tp = _controlled_params(seed=4242)
    _, samples_a, _, _, _, _ = cds.generate_controlled_dataset(dp, tp, shuffle=False)
    _, samples_b, _, _, _, _ = cds.generate_controlled_dataset(dp, tp, shuffle=False)
    assert np.array_equal(samples_a, samples_b)


def test_controlled_resolved_seed_recorded():
    """The metadata records the resolved seed actually used (P0-2 + P0-3)."""
    dp, tp = _controlled_params(seed=4242)
    _, _, _, _, _, meta = cds.generate_controlled_dataset(dp, tp, shuffle=False)
    assert meta["resolved_seed"] == 4242


def test_controlled_unseeded_records_real_seed_not_zero():
    """An unseeded (seed=None) controlled run records the *drawn* seed, never a
    false 0 (P0-3). The drawn seed is a real integer and is almost surely != 0."""
    dp, tp = _controlled_params(seed=None)
    _, _, _, _, _, meta = cds.generate_controlled_dataset(dp, tp, shuffle=False)
    assert isinstance(meta["resolved_seed"], int)
    # name must embed the resolved seed, not the literal 0
    assert str(meta["resolved_seed"]) in meta["name"]


def test_controlled_seed_false_name_has_no_false_zero():
    """seed=False escape hatch: the task resolves no seed (resolved_seed is None),
    so metadata records resolved_seed=None and the dataset name must NOT embed a
    false `0` (P0-3 / right-and-works-6). Caller seeds the global RNG first."""
    np.random.seed(2024)
    dp, tp = _controlled_params(seed=False)
    _, _, _, _, _, meta = cds.generate_controlled_dataset(dp, tp, shuffle=False)
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


@pytest.mark.skip(reason="blocked by pre-existing _adjust_labels=True/validate bug — see escalations.md E3")
@pytest.mark.slow
def test_human_seed_is_deterministic_production_path():
    """Determinism on the real production path (_adjust_labels=True, validate=True).

    Invariant guard; the distinguishing P0-2 check is test_human_pipeline_seeds_once.

    This test is skipped because _adjust_labels=True triggers
    estimate_effective_hazard_rates -> generate_video_dataset -> validate=True
    path, which raises AssertionError at dataset.py:144 (reverse-mode initial-color
    index inconsistency, a pre-existing bug independent of P0-2). Unblock by
    fixing E3 first, then remove the skip decorator."""
    dp, tp = _human_params(seed=99)
    dp_large = dict(dp)
    dp_large["total_videos"] = 30  # larger than the fast unit's 12
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

    Calls hds.generate_video_dataset directly with model trial type funcs and
    model defaults (same code path as generate_model_dataset_nongray) to bypass
    the generate_model_dataset_nongray wrapper, which hard-codes _adjust_labels=True
    (the default) and triggers estimate_effective_hazard_rates, which fails when
    total_dataset_length=None (a pre-existing bug unrelated to P0-2). Using
    _adjust_labels=False with model defaults and validate=False exercises the full
    seeding inheritance path (seed flows from NongrayDatasetParameters through
    generate_video_parameters into per-trial BouncingBallTask constructions)."""
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
