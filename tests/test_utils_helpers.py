"""Direct unit tests for pure helper functions in the task utils modules.

Covers coverage-survey items (2026-09-10-S1-coverage-survey.md):
  #2  taskutils.last_visible_color                       (direct unit)
  #4  htaskutils.compute_dataset_size_video_based/_time_based  (sizing + guards)
  #8  human_bouncing_ball.dataset.compute_change_stats   (synthetic arithmetic)
  #9  htaskutils.group_trial_data                         (arity/broadcast contract)
  #12 htaskutils.compute_trial_color_and_stats            (balance characterization)

All items are fast pure-function units: no ``BouncingBallTask`` build, no
production human/model pipeline. Randomness is seeded explicitly with
``set_global_seed`` (the autouse ``reset_random_state`` fixture in conftest also
seeds 42 before every test).

Tests that document a *real* defect are marked ``xfail(strict=True)`` and assert
the INTENDED behaviour (so they flip green when the bug is fixed, and fail the
suite if the behaviour silently "improves" to something else). Tests that merely
pin down accepted quirks are plain passing characterization tests with a comment
saying so.
"""
import numpy as np
import pandas as pd
import pytest

from bouncing_ball_task.utils import htaskutils
from bouncing_ball_task.utils.taskutils import last_visible_color
from bouncing_ball_task.human_bouncing_ball.dataset import compute_change_stats
from bouncing_ball_task.utils.pyutils import set_global_seed, create_sequence_splits


# ---------------------------------------------------------------------------
# Item #2 -- taskutils.last_visible_color
# ---------------------------------------------------------------------------
#
# samples has shape (B, T, 5): feature 0 is the x-coordinate, feature 1 the
# y-coordinate, features 2:5 the RGB colour. The function reports, per row, the
# colour (and optionally index) of the LAST frame at which the ball is still
# "visible" (outside the masked x-band). The band definition differs by mode:
#   centroid: open interval  (mask_start - tol, mask_end + tol)
#   inner   : [mask_start + r - tol, mask_end - r + tol]
#   outer   : [mask_start - r - tol, mask_end + r + tol]
# with r = ball_radius. There is NO final ``else`` branch (see the unknown-mode
# characterization test below).

_MASK_START = 40
_MASK_END = 60
_BALL_RADIUS = 5
# x-trajectory: approaches from the left and ends inside the mask centre (50).
# Hand-computed last-visible frame per mode (tol=0, r=5, band edges 40/60):
#   centroid band (40,60)  open -> last x<=40 is frame 4 (x=38)
#   inner   band [45,55]        -> last x<45  is frame 5 (x=44)
#   outer   band [35,65]        -> last x<35  is frame 3 (x=34)
_X_TRAJ = [10, 20, 30, 34, 38, 44, 48, 52, 50, 50]
_EXPECTED_IDX = {"centroid": 4, "inner": 5, "outer": 3}


def _make_samples(x_traj):
    """Build a (1, T, 5) samples array with a unique RGB colour per frame."""
    t = len(x_traj)
    colors = np.array([[i, 0, 255 - i] for i in range(t)], dtype=float)
    s = np.zeros((1, t, 5), dtype=float)
    s[0, :, 0] = x_traj
    s[0, :, 2:] = colors
    return s, colors


@pytest.mark.parametrize("mode", ["centroid", "inner", "outer"])
def test_last_visible_color_crossing_index_and_color(mode):
    """#2: each mode returns the correct last-visible index/colour at the crossing."""
    s, colors = _make_samples(_X_TRAJ)
    color, idx = last_visible_color(
        s, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode=mode, tol=0, return_index=True,
    )
    exp = _EXPECTED_IDX[mode]
    assert int(idx[0]) == exp
    np.testing.assert_array_equal(color[0], colors[exp])


# Right-edge coverage (mirror of _X_TRAJ around the band centre 50): the ball
# approaches from the RIGHT and ends inside the mask centre, so the last-visible
# frame is now governed by the mask_end (right) band edge rather than mask_start.
# By symmetry (x' = 100 - x) the crossing frame indices match the left case:
#   centroid band (40,60) open -> last x>=60 is frame 4 (x=62)
#   inner   band [45,55]       -> last x>55  is frame 5 (x=56)
#   outer   band [35,65]       -> last x>65  is frame 3 (x=66)
_X_TRAJ_RIGHT = [90, 80, 70, 66, 62, 56, 52, 48, 50, 50]
_EXPECTED_IDX_RIGHT = {"centroid": 4, "inner": 5, "outer": 3}


@pytest.mark.parametrize("mode", ["centroid", "inner", "outer"])
def test_last_visible_color_crossing_index_and_color_right_edge(mode):
    """#2 (right mask edge): a trajectory crossing ``mask_end`` from the right
    returns the correct last-visible index/colour for each mode. Complements the
    left-edge test so a right-band-edge regression is caught at the unit level."""
    s, colors = _make_samples(_X_TRAJ_RIGHT)
    color, idx = last_visible_color(
        s, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode=mode, tol=0, return_index=True,
    )
    exp = _EXPECTED_IDX_RIGHT[mode]
    assert int(idx[0]) == exp
    np.testing.assert_array_equal(color[0], colors[exp])


def test_last_visible_color_tol_shifts_boundary_right_edge():
    """#2 (right mask edge): a positive ``tol`` widens the masked band at the
    mask_end side too, moving the last-visible frame EARLIER. Outer band with
    tol=5 -> [30,70]; last x>70 is frame 1 (x=80)."""
    s, _ = _make_samples(_X_TRAJ_RIGHT)
    _, idx0 = last_visible_color(
        s, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode="outer", tol=0, return_index=True,
    )
    _, idx5 = last_visible_color(
        s, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode="outer", tol=5, return_index=True,
    )
    assert int(idx0[0]) == 3
    assert int(idx5[0]) == 1
    assert int(idx5[0]) < int(idx0[0])


def test_last_visible_color_batched_fancy_indexing():
    """#2: per-row indices resolve correctly for B>1 (exercises the fancy index
    ``samples[arange(B), last_visible_index, 2:]`` that the real caller hits)."""
    s_cross, colors = _make_samples(_X_TRAJ)
    s_vis, _ = _make_samples([10] * len(_X_TRAJ))  # never enters the mask
    batch = np.concatenate([s_cross, s_vis], axis=0)  # (2, T, 5)
    color, idx = last_visible_color(
        batch, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode="outer", tol=0, return_index=True,
    )
    # row 0 crosses (outer -> frame 3); row 1 all-visible -> last frame (T-1)
    assert idx.tolist() == [3, len(_X_TRAJ) - 1]
    np.testing.assert_array_equal(color[0], colors[3])
    np.testing.assert_array_equal(color[1], colors[len(_X_TRAJ) - 1])


def test_last_visible_color_tol_shifts_boundary():
    """#2: a positive ``tol`` widens the masked band, so the last-visible frame
    moves EARLIER. Outer band with tol=5 -> [30,70]; last x<30 is frame 1."""
    s, _ = _make_samples(_X_TRAJ)
    _, idx0 = last_visible_color(
        s, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode="outer", tol=0, return_index=True,
    )
    _, idx5 = last_visible_color(
        s, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode="outer", tol=5, return_index=True,
    )
    assert int(idx0[0]) == 3
    assert int(idx5[0]) == 1
    assert int(idx5[0]) < int(idx0[0])


def test_last_visible_color_all_visible_returns_last_frame():
    """#2: a trajectory that never enters the mask reports the final frame."""
    t = len(_X_TRAJ)
    s, colors = _make_samples([10] * t)
    color, idx = last_visible_color(
        s, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode="outer", return_index=True,
    )
    assert int(idx[0]) == t - 1
    np.testing.assert_array_equal(color[0], colors[t - 1])


def test_last_visible_color_all_masked_returns_last_frame_CHARACTERIZATION():
    """#2 CHARACTERIZATION (not correctness): an all-masked trajectory has an
    all-False visibility mask; ``argmax`` of all-False is 0, so the function
    silently returns ``timesteps - 1`` (the LAST frame) rather than signalling
    that the ball was never visible. Documented, not endorsed."""
    t = len(_X_TRAJ)
    s, _ = _make_samples([50] * t)  # dead-centre of every band
    _, idx = last_visible_color(
        s, _BALL_RADIUS, _MASK_START, _MASK_END,
        time_step_mode="outer", return_index=True,
    )
    assert int(idx[0]) == t - 1


def test_last_visible_color_unknown_mode_raises_UnboundLocalError_CHARACTERIZATION():
    """#2 CHARACTERIZATION: the mode dispatch has no final ``else``, so an
    unrecognised ``time_step_mode`` leaves ``out_mask`` unassigned and raises
    ``UnboundLocalError`` (NOT a ``ValueError``). Documented, not endorsed."""
    s, _ = _make_samples(_X_TRAJ)
    with pytest.raises(UnboundLocalError):
        last_visible_color(s, _BALL_RADIUS, _MASK_START, _MASK_END,
                           time_step_mode="bogus")


# ---------------------------------------------------------------------------
# Item #4 -- htaskutils.compute_dataset_size_video_based / _time_based
# ---------------------------------------------------------------------------
#
# Shared sizing params (mirror the human defaults): duration=50 ms/frame,
# video_length_min_s=7.5 -> min_ms=7500 -> min_f=150; exp_scale=3.75 s ->
# exp_scale_ms=3750. max_length_mult=3 keeps frames < 450.
_DURATION = 50
_MIN_F = 150
_MIN_MS = 7500
_EXP_MS = 3750
_TRIAL_TYPES = ("catch", "straight", "bounce")


def test_video_based_counts_obey_split_and_catch_uses_min_length():
    """#4: per-type counts are ``rint(p_split * total_videos)``; catch uses the
    fixed minimum length; sampled (straight/bounce) lengths stay within
    ``[min_f, min_f * max_length_mult)``."""
    set_global_seed(42)
    split = create_sequence_splits([0.2, 0.4, 0.4])
    counts, lengths = htaskutils.compute_dataset_size_video_based(
        100, _MIN_F, _MIN_MS, split, None, _EXP_MS, _DURATION, _TRIAL_TYPES,
    )
    assert counts == {"catch": 20, "straight": 40, "bounce": 40}
    # catch: fixed minimum length
    assert np.all(lengths["catch"] == _MIN_F)
    assert len(lengths["catch"]) == 20
    # straight/bounce: sampled, count matches, all under the cutoff
    for tt in ("straight", "bounce"):
        assert len(lengths[tt]) == counts[tt]
        assert lengths[tt].min() >= _MIN_F
        assert lengths[tt].max() < _MIN_F * 3


def test_video_based_fixed_length_returns_constant_lengths():
    """#4: ``fixed_video_length`` forces every type (catch included, since the
    ``if fixed_video_length`` branch precedes the catch branch) to a constant."""
    set_global_seed(42)
    split = create_sequence_splits([0.2, 0.4, 0.4])
    counts, lengths = htaskutils.compute_dataset_size_video_based(
        30, _MIN_F, _MIN_MS, split, 200, _EXP_MS, _DURATION, _TRIAL_TYPES,
    )
    for tt in _TRIAL_TYPES:
        assert np.all(lengths[tt] == 200)


def test_video_based_over_restrictive_raises_ValueError():
    """#4: when ``max_length_mult`` prunes below the requested count,
    ``_video_based`` raises a ``ValueError`` with a documented message.
    max_length_mult=0.5 puts the cutoff below the minimum length, so no sample
    survives."""
    set_global_seed(42)
    with pytest.raises(ValueError, match="too restrictive"):
        htaskutils.compute_dataset_size_video_based(
            10, _MIN_F, _MIN_MS, create_sequence_splits([1.0]), None,
            _EXP_MS, _DURATION, ("straight",), max_length_mult=0.5,
        )


def test_time_based_catch_returns_constant_min_length():
    """#4: ``_time_based`` catch trials all take the fixed minimum length; the
    internal budget assertion holding is proven by the call returning."""
    set_global_seed(42)
    counts, lengths = htaskutils.compute_dataset_size_time_based(
        10, _MIN_F, _MIN_MS, create_sequence_splits([1.0]),
        _EXP_MS, _DURATION, ("catch",),
    )
    assert counts["catch"] > 0
    assert np.all(lengths["catch"] == _MIN_F)


def test_time_based_underfill_raises_IndexError_CHARACTERIZATION():
    """#4 CHARACTERIZATION (asymmetry vs ``_video_based``): ``_time_based`` has
    NO count guard. When the per-type budget is too small to fit anything,
    ``max_video_num`` rounds to 0, the length array is empty, and
    ``np.where(cumsum < budget)[0][-1]`` indexes an empty array -> a bare
    ``IndexError`` (the sibling ``_video_based`` raises a clean ``ValueError``
    in the analogous under-fill case). Documented, not endorsed; frame it as a
    characterization (no spec fixes the intended exception here)."""
    set_global_seed(42)
    with pytest.raises(IndexError):
        htaskutils.compute_dataset_size_time_based(
            0.05, _MIN_F, _MIN_MS, create_sequence_splits([1.0]),
            _EXP_MS, _DURATION, ("catch",),
        )


# ---------------------------------------------------------------------------
# Item #8 -- human_bouncing_ball.dataset.compute_change_stats
# ---------------------------------------------------------------------------
#
# change_sequence is a list of (T, 4) arrays; columns are
#   [0] vcb = velocity change bounce   [1] vcr = velocity change random
#   [2] ccb = colour change bounce     [3] ccr = colour change random
# Tested purely on SYNTHETIC input (never through the production human pipeline;
# see QUESTIONABLE (a) -- on that path these columns are the *pre-adjustment*
# ledger, so a production equality check would be misleading). The one optional
# production-path xfail is intentionally NOT written here: it requires the human
# ``_adjust_labels=True`` build at n>=60 (the @slow path), which is out of scope
# for this fast-unit file. See the report.


def _synthetic_change_sequence():
    # seq0: T=10, vcb=2, vcr=1, ccb=1, ccr=3
    s0 = np.zeros((10, 4), dtype=int)
    s0[:2, 0] = 1
    s0[0, 1] = 1
    s0[0, 2] = 1
    s0[:3, 3] = 1
    # seq1: T=5, vcb=0, vcr=0, ccb=2, ccr=1  (vcb+vcr==0 -> exercises max(1, .))
    s1 = np.zeros((5, 4), dtype=int)
    s1[:2, 2] = 1
    s1[0, 3] = 1
    return [s0, s1]


def test_compute_change_stats_count_columns():
    """#8: the four count columns equal the per-sequence channel sums."""
    df = pd.DataFrame(index=[0, 1])
    df = compute_change_stats(df, _synthetic_change_sequence(), suffix="", duration=None)
    assert df["Bounces"].tolist() == [2, 0]              # vcb
    assert df["Random Bounces"].tolist() == [1, 0]       # vcr
    assert df["Color Change Bounce"].tolist() == [1, 2]  # ccb
    assert df["Color Change Random"].tolist() == [3, 1]  # ccr


def test_compute_change_stats_effective_ratios():
    """#8: effective ratios match hand arithmetic, including the ``max(1, .)``
    divide-by-zero guard on PCCOVC (seq1 has vcb+vcr==0 -> denom clamped to 1)."""
    df = pd.DataFrame(index=[0, 1])
    df = compute_change_stats(df, _synthetic_change_sequence(), suffix="", duration=None)
    # PCCNVC_effective = ccr / timesteps
    np.testing.assert_allclose(df["PCCNVC_effective"], [3 / 10, 1 / 5])
    # PCCOVC_effective = ccb / max(1, vcb + vcr)
    np.testing.assert_allclose(df["PCCOVC_effective"], [1 / 3, 2 / 1])
    # PVC_effective = vcr / timesteps
    np.testing.assert_allclose(df["PVC_effective"], [1 / 10, 0.0])


def test_compute_change_stats_per_second_scaling():
    """#8: the ``_ps`` (per-second) columns scale by 1000/(timesteps*duration)."""
    df = pd.DataFrame(index=[0, 1])
    df = compute_change_stats(df, _synthetic_change_sequence(), suffix="", duration=50)
    # 1000 * ccr / (timesteps * duration)
    np.testing.assert_allclose(df["PCCNVC_effective_ps"], [1000 * 3 / (10 * 50),
                                                           1000 * 1 / (5 * 50)])
    # 1000 * vcr / (timesteps * duration)
    np.testing.assert_allclose(df["PVC_effective_ps"], [1000 * 1 / (10 * 50), 0.0])


def test_compute_change_stats_suffix_names_columns():
    """#8: a non-empty suffix appends ' <suffix>' to count columns and
    '_<suffix>' to effective columns."""
    df = pd.DataFrame(index=[0, 1])
    df = compute_change_stats(df, _synthetic_change_sequence(),
                              suffix="adjusted", duration=50)
    assert "Bounces adjusted" in df.columns
    assert "PCCNVC_effective_adjusted" in df.columns
    assert "PCCNVC_effective_adjusted_ps" in df.columns


# ---------------------------------------------------------------------------
# Item #9 -- htaskutils.group_trial_data
# ---------------------------------------------------------------------------


def _group_inputs(num_trials=2, color_len=None):
    color_len = num_trials if color_len is None else color_len
    return dict(
        num_trials=num_trials,
        final_position=[[1, 1], [2, 2]][:num_trials],
        final_velocity=[[0, 0], [0, 0]][:num_trials],
        final_color=[(255, 0, 0), (0, 255, 0), (0, 0, 255)][:color_len],
        pccnvc=[0.01, 0.02][:num_trials],
        pccovc=[0.1, 0.2][:num_trials],
    )


def test_group_trial_data_scalar_pvc_broadcasts_to_9_tuples():
    """#9: with ``dict_meta_trials=None`` the assembler yields 9-tuples, and a
    scalar ``pvc`` is broadcast to a per-trial value."""
    kw = _group_inputs()
    out = htaskutils.group_trial_data(pvc=0.5, **kw)
    assert len(out) == 2
    assert all(len(trial) == 9 for trial in out)
    assert [trial[5] for trial in out] == [0.5, 0.5]  # pvc position


def test_group_trial_data_with_meta_yields_10_tuples():
    """#9: passing ``dict_meta_trials`` appends a per-trial dict, producing the
    10-tuple that ``generate_video_dataset`` unpacks downstream."""
    kw = _group_inputs()
    out = htaskutils.group_trial_data(
        pvc=0.5, dict_meta_trials={"foo": [10, 20], "bar": ["a", "b"]}, **kw,
    )
    assert all(len(trial) == 10 for trial in out)
    assert out[0][-1] == {"foo": 10, "bar": "a"}
    assert out[1][-1] == {"foo": 20, "bar": "b"}


def test_group_trial_data_length_mismatch_asserts():
    """#9: any input list whose length != num_trials trips the guard assert."""
    kw = _group_inputs(num_trials=2, color_len=3)  # final_color too long
    with pytest.raises(AssertionError, match="Length mismatch"):
        htaskutils.group_trial_data(pvc=0.5, **kw)


# NOTE (documented, not tested per survey #9): the ``bounce_index_x = [[],] *
# num_trials`` defaults alias one shared list object across trials -- mutating
# one row's list would mutate all. Harmless while the defaults stay read-only.


# ---------------------------------------------------------------------------
# Item #12 -- htaskutils.compute_trial_color_and_stats (balance characterization)
# ---------------------------------------------------------------------------
#
# Assigns final colours and (pccnvc, pccovc) pairs across trials via
# pyutils.repeat_sequence. pccnvc uses roll, pccovc uses neither shuffle nor
# roll, so the JOINT distribution is deterministic and seed-independent (verified
# across seeds during authoring) -- which is why the xfails below are safe as
# strict.
#
# Finding: the marginal distributions (colour, pccnvc, pccovc) are balanced, but
# the JOINT (pccnvc x pccovc) distribution is NOT balanced-within-1 at the
# production per-type sizes. On the paper path (human split (0.05,-1,-1,0)):
#   total_videos=12 -> straight/bounce get 6 trials each -> only 4 of 6 (pccnvc,
#     pccovc) conditions ever appear (two conditions entirely missing);
#   total_videos=60 -> straight/bounce get 28 trials each -> pair-count spread=2.
# See the two strict xfails below and the report.
_PCCNVC_LINSPACE = np.linspace(0.01, 0.05, 2)   # num_pccnvc=2
_PCCOVC_LINSPACE = np.linspace(0.1, 0.9, 3)     # num_pccovc=3
_N_PAIRS = 2 * 3


def _color_and_stats(num_trials, seed=42):
    set_global_seed(seed)
    dict_meta = {"pccnvc_linspace": _PCCNVC_LINSPACE,
                 "pccovc_linspace": _PCCOVC_LINSPACE}
    dict_meta_type = {}
    _, _, _, dict_meta_type = htaskutils.compute_trial_color_and_stats(
        num_trials, dict_meta, dict_meta_type,
    )
    return dict_meta_type


def _spread(counts):
    return int(counts.max() - counts.min())


def test_color_and_stats_final_color_balanced():
    """#12: final-colour counts are balanced to within 1 (in fact exact at a
    multiple of 3)."""
    dmt = _color_and_stats(30)
    counts = dmt["final_color_counts"][1]
    assert _spread(counts) <= 1


def test_color_and_stats_marginals_balanced():
    """#12: the pccnvc and pccovc MARGINALS are balanced to within 1 -- the
    defect below is joint-only, not in the marginals."""
    dmt = _color_and_stats(30)
    assert _spread(dmt["pccnvc_counts"][1]) <= 1
    assert _spread(dmt["pccovc_counts"][1]) <= 1


def test_color_and_stats_pairs_balanced_at_multiple_of_12():
    """#12: at num_trials=12 (a multiple of num_pccnvc*num_pccovc*2) the joint
    distribution IS balanced and complete -- proof the imbalance below is a
    size-dependent alignment defect, not an inherent impossibility."""
    dmt = _color_and_stats(12)
    values, counts = dmt["pccnvc_pccovc_counts"]
    assert len(values) == _N_PAIRS       # all 6 conditions present
    assert _spread(counts) <= 1


@pytest.mark.xfail(strict=True, reason=(
    "BUG (item #12): at total_videos=12 straight/bounce each get 6 trials, but "
    "compute_trial_color_and_stats produces only 4 of the 6 (pccnvc, pccovc) "
    "conditions -- two experimental conditions are entirely absent. Seed-"
    "independent. Intended: all num_pccnvc*num_pccovc=6 conditions present."))
def test_color_and_stats_all_conditions_present_at_production_size_12():
    """#12: INTENDED invariant -- every (pccnvc, pccovc) condition appears at the
    production per-type size (6). Currently red (only 4/6 present)."""
    dmt = _color_and_stats(6)
    values = dmt["pccnvc_pccovc_counts"][0]
    assert len(values) == _N_PAIRS


@pytest.mark.xfail(strict=True, reason=(
    "BUG (item #12): at total_videos=60 straight/bounce each get 28 trials and "
    "the (pccnvc, pccovc) joint counts span [4..6] (spread=2), not balanced to "
    "within 1. Marginals are balanced; the joint alignment via repeat_sequence "
    "roll/tile is not. Seed-independent. Intended: joint spread <= 1."))
def test_color_and_stats_pairs_balanced_at_production_size_28():
    """#12: INTENDED invariant -- joint (pccnvc, pccovc) balanced to within 1 at
    the production per-type size (28). Currently red (spread=2)."""
    dmt = _color_and_stats(28)
    counts = dmt["pccnvc_pccovc_counts"][1]
    assert _spread(counts) <= 1
