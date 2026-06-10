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
