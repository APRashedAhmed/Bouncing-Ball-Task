"""Content / geometry characterization tests for the HUMAN bouncing-ball path.

Covers coverage-survey items #1 (catch/straight/bounce end geometry), #3
(adjust_dataset_labels colour/label consistency), #5 (compute_effective_stats
Hazard/Contingency binning), #10 (nonwall geometry via an overridden split), and
#11 (generate_dataset_metadata last-visible-colour window arithmetic).

Reverse-mode fact (human task: sequence_mode="reverse", target_future_timestep=0):
a generator's ``final_position`` / ``final_velocity`` is where the trial ENDS, so
every geometric invariant below is phrased on the trajectory END.

HUMAN PATH ONLY. No model trial funcs, no ``generate_model_dataset_nongray``, and
no direct ``estimate_effective_hazard_rates`` call. The human ``_adjust_labels=True``
path (item #3) reaches ``estimate_effective_hazard_rates`` internally with a real
``total_dataset_length`` — the same path ``test_reproducibility`` already exercises.

All five items assert INTENDED behaviour and are expected to pass; none touches a
QUESTIONABLE row (a: pre-adjust ledger staleness; b: rvc/nonwall channel mis-ledger;
c: change-vector delay), so no ``xfail`` is warranted here. In particular item #10
asserts the generator's declared forced-change PARAMETER (``bounce_index_y``) and
open-space geometry, never the runtime change channel (QUESTIONABLE row b).
"""
import copy

import numpy as np
import pytest

from bouncing_ball_task.constants import DEFAULT_COLORS, default_color_to_idx_dict
from bouncing_ball_task.human_bouncing_ball import dataset as hds
from bouncing_ball_task.human_bouncing_ball import defaults as hdefaults
from bouncing_ball_task.utils.pyutils import set_global_seed


_DEFAULT_COLOR_SET = {tuple(c) for c in DEFAULT_COLORS}


def _human_params(seed, total_videos=24, split=None):
    """Local small-dataset builder mirroring the private one in
    test_reproducibility.py; kept local by dispatch. print_stats is silenced to
    keep the (huge) provenance summary out of captured output."""
    dataset_params = {
        key: getattr(hdefaults.HumanDatasetParameters(), key)
        for key in hdefaults.HumanDatasetParameters.keys
    }
    dataset_params["seed"] = seed
    dataset_params["total_videos"] = total_videos
    dataset_params["print_stats"] = False
    if split is not None:
        dataset_params["trial_type_split"] = split
    task_params = {
        key: getattr(hdefaults.TaskParameters(), key)
        for key in hdefaults.TaskParameters.keys
    }
    return dataset_params, task_params


def _params_and_meta(seed, total_videos=24, split=None, funcs=None):
    """Build per-trial parameters + metadata with NO task construction (fast)."""
    funcs = funcs or hds.dict_trial_type_generation_funcs
    dp, _ = _human_params(seed, total_videos=total_videos, split=split)
    set_global_seed(seed)
    dict_params, dict_meta = hds.generate_video_parameters(
        **dp, dict_trial_type_generation_funcs=funcs
    )
    return dict_params, dict_meta


@pytest.fixture(scope="module")
def small_human_dataset():
    """One small human dataset (_adjust_labels=False) shared by items #5 and #11.

    Deterministic: the explicit seed drives the internal set_global_seed, so the
    function-scoped autouse reset_random_state (seed 42) does not perturb it.
    """
    dp, tp = _human_params(seed=7, total_videos=24)
    task, samples, model_samples, targets, df_data, metadata = hds.generate_video_dataset(
        dp, tp, hds.dict_trial_type_generation_funcs,
        _adjust_labels=False, validate=False,
    )
    return dict(
        task=task, samples=samples, model_samples=model_samples,
        targets=targets, df=df_data, meta=metadata,
    )


# --------------------------------------------------------------------------- #
# Item #1 — human trial-type END geometry (catch / straight / bounce)
# --------------------------------------------------------------------------- #
def test_item1_catch_end_geometry():
    """Catch trials END in a non-grayzone x band, never inside the mask band, and
    carry the catch warmup overrides."""
    dict_params, dict_meta = _params_and_meta(seed=101, total_videos=60)
    mask_start, mask_end = dict_meta["mask_start"], dict_meta["mask_end"]
    size_x, size_y = dict_meta["size_x"], dict_meta["size_y"]

    catch = dict_params["catch"]
    meta_catch = dict_meta["catch"]
    assert len(catch) == meta_catch["num_trials"] >= 1

    left_lo, left_hi = meta_catch["nongrayzone_left_x_range"]
    right_lo, right_hi = meta_catch["nongrayzone_right_x_range"]

    for pos, vel, col, *_rest in catch:
        x, y = pos
        in_left = left_lo <= x <= left_hi
        in_right = right_lo <= x <= right_hi
        assert in_left or in_right, f"catch end-x {x} not in either non-grayzone band"
        # Never inside the mask band.
        assert x < mask_start or x > mask_end, f"catch end-x {x} lies inside mask band"
        assert 0 <= x <= size_x and 0 <= y <= size_y
        assert tuple(col) in _DEFAULT_COLOR_SET

    # Catch forces no random velocity/colour changes during the warmup window; the
    # override value is catch_ncc_nvc_timesteps (threaded into dict_meta via kwargs).
    catch_ts = dict_meta["catch_ncc_nvc_timesteps"]
    assert meta_catch["overrides"] == {
        "warmup_t_no_rand_velocity_change": catch_ts,
        "warmup_t_no_rand_color_change": catch_ts,
    }


def test_item1_straight_end_geometry():
    """Straight trials END on a grayzone x-grid value (inside the mask band) with y
    kept inside the frame."""
    dict_params, dict_meta = _params_and_meta(seed=101, total_videos=60)
    mask_start, mask_end = dict_meta["mask_start"], dict_meta["mask_end"]
    size_x, size_y = dict_meta["size_x"], dict_meta["size_y"]
    ball_radius = dict_meta["ball_radius"]
    grid = np.asarray(dict_meta["x_grayzone_linspace_sides"]).ravel()

    straight = dict_params["straight"]
    assert len(straight) == dict_meta["straight"]["num_trials"] >= 1
    for pos, vel, col, *_rest in straight:
        x, y = pos
        assert np.any(np.isclose(x, grid)), f"straight end-x {x} not on grayzone grid {grid}"
        assert mask_start < x < mask_end, f"straight end-x {x} not inside mask band"
        # Straight y is bounded away from the top/bottom frame edges (straight.py).
        assert 2 * ball_radius <= y <= size_y - 2 * ball_radius
        assert 0 <= x <= size_x
        assert tuple(col) in _DEFAULT_COLOR_SET


def test_item1_bounce_end_geometry():
    """Bounce trials END inside the mask band with y on the near-wall bounce grid."""
    dict_params, dict_meta = _params_and_meta(seed=101, total_videos=60)
    mask_start, mask_end = dict_meta["mask_start"], dict_meta["mask_end"]
    size_x, size_y = dict_meta["size_x"], dict_meta["size_y"]

    bounce = dict_params["bounce"]
    meta_bounce = dict_meta["bounce"]
    assert len(bounce) == meta_bounce["num_trials"] >= 1
    pos_y_grid = np.asarray(meta_bounce["pos_y_bounce_linspace"]).ravel()

    for pos, vel, col, *_rest in bounce:
        x, y = pos
        assert mask_start <= x <= mask_end, f"bounce end-x {x} not inside mask band"
        assert np.any(np.isclose(y, pos_y_grid)), f"bounce end-y {y} not on bounce grid"
        assert 0 <= x <= size_x and 0 <= y <= size_y
        assert tuple(col) in _DEFAULT_COLOR_SET

    # Every bounce meta row carries the trial label.
    for _pos, _vel, _col, _n, _o, _p, _fx, _fy, _fcc, meta in bounce:
        assert meta["trial"] == "bounce"


# --------------------------------------------------------------------------- #
# Item #10 — nonwall geometry (off by default; enabled via an overridden split)
# --------------------------------------------------------------------------- #
def test_item10_nonwall_open_space_geometry():
    """With a split that grants nonwall a nonzero share, nonwall trials force a
    velocity change in OPEN SPACE: the change is declared in bounce_index_y (not x),
    and the END position sits away from every wall — strictly farther from the
    nearest wall than any bounce trial in the same dataset (the whole point of the
    type). Geometry only; the runtime change-CHANNEL is not asserted (survey row b).
    """
    split = (0.05, -1, -1, -1)  # (catch, straight, bounce, nonwall)
    dict_params, dict_meta = _params_and_meta(seed=123, total_videos=60, split=split)
    assert "nonwall" in dict_params and len(dict_params["nonwall"]) > 0

    mask_start, mask_end = dict_meta["mask_start"], dict_meta["mask_end"]
    size_x, size_y = dict_meta["size_x"], dict_meta["size_y"]
    bounce_timestep = dict_meta["bounce_timestep"]
    meta_nw = dict_meta["nonwall"]

    # Open-space y bound derived from metadata (no magic threshold).
    dt = dict_meta["dt"]
    vy = dict_meta["final_velocity_y_magnitude_linspace"]
    distance_y = bounce_timestep * vy * dt
    pos_y_grid = np.asarray(meta_nw["pos_y_bounce_linspace"])
    y_lo = pos_y_grid.min() - distance_y.max()
    y_hi = pos_y_grid.max() + distance_y.max()

    nonwall = dict_params["nonwall"]
    assert len(nonwall) == meta_nw["num_trials"]
    for pos, vel, col, _n, _o, _p, fxvc, fyvc, fcc, meta in nonwall:
        x, y = pos
        # Forced change declared on y (mid-arena), not at a wall via x.
        assert list(fxvc) == []
        assert list(fyvc) == [bounce_timestep]
        assert list(fcc) == []
        assert meta["trial"] == "nonwall"
        assert mask_start < x < mask_end, f"nonwall end-x {x} not inside mask band"
        assert y_lo <= y <= y_hi, f"nonwall end-y {y} outside open-space band [{y_lo},{y_hi}]"
        assert 0 <= x <= size_x and 0 <= y <= size_y
        assert tuple(col) in _DEFAULT_COLOR_SET

    # Discriminating cross-type check: nonwall ends farther from any wall than bounce.
    def wall_dist(p):
        px, py = p
        return min(px, size_x - px, py, size_y - py)

    nonwall_min = min(wall_dist(tr[0]) for tr in nonwall)
    bounce_max = max(wall_dist(tr[0]) for tr in dict_params["bounce"])
    assert nonwall_min > bounce_max, (
        f"nonwall nearest-wall distance {nonwall_min} should exceed bounce's "
        f"{bounce_max} (nonwall changes in open space, bounce at a wall)"
    )


# --------------------------------------------------------------------------- #
# Item #5 — compute_effective_stats Hazard Rate / Contingency binning
# --------------------------------------------------------------------------- #
def test_item5_effective_stats_binning(small_human_dataset):
    """Every row is binned into a real Hazard Rate (Low/High) and Contingency
    (Low/Medium/High) category — never NaN — and the bin matches the underlying
    PCCNVC/PCCOVC value under the default num_pccnvc=2, num_pccovc=3."""
    df = small_human_dataset["df"]

    # Category vocabularies are exactly the hardcoded sets.
    assert list(df["Hazard Rate"].cat.categories) == ["Low", "High"]
    assert list(df["Contingency"].cat.categories) == ["Low", "Medium", "High"]

    # No row falls through to NaN ("Unknown").
    assert not df["Hazard Rate"].isna().any()
    assert not df["Contingency"].isna().any()

    # Both hazard categories and all three contingency categories are present.
    assert set(df["Hazard Rate"].dropna().unique()) == {"Low", "High"}
    assert set(df["Contingency"].dropna().unique()) == {"Low", "Medium", "High"}

    # Default parameter grids the binning must reproduce.
    hd = hdefaults.HumanDatasetParameters()
    pccnvc_lo, pccnvc_hi = hd.pccnvc_lower, hd.pccnvc_upper
    cont_grid = np.linspace(hd.pccovc_lower, hd.pccovc_upper, hd.num_pccovc)

    # Hazard Rate bin agrees with the PCCNVC value.
    low = df[df["Hazard Rate"] == "Low"]
    high = df[df["Hazard Rate"] == "High"]
    assert np.allclose(low["PCCNVC"].to_numpy(), pccnvc_lo)
    assert np.allclose(high["PCCNVC"].to_numpy(), pccnvc_hi)

    # Contingency bin agrees with the PCCOVC value.
    for label, expected in zip(["Low", "Medium", "High"], cont_grid):
        rows = df[df["Contingency"] == label]
        assert np.allclose(rows["PCCOVC"].to_numpy(), expected), (
            f"Contingency '{label}' rows expected PCCOVC {expected}"
        )


# --------------------------------------------------------------------------- #
# Item #11 — last-visible-colour window arithmetic (generate_dataset_metadata)
# --------------------------------------------------------------------------- #
def test_item11_last_visible_color_window(small_human_dataset):
    """last_visible_color_idx indexes the FULL per-trial trajectory and marks the
    last frame the ball is outer-band visible: frame[idx] is visible and, when the
    ball later enters the mask, frame[idx+1] is not. The window arithmetic (tail
    min_length window + lengths - min_length) is validated against raw output_targets.
    """
    df = small_human_dataset["df"]
    targets = small_human_dataset["targets"]
    meta = small_human_dataset["meta"]
    r, mask_start, mask_end = meta["ball_radius"], meta["mask_start"], meta["mask_end"]

    lengths = df["length"].to_numpy()
    lvci = df["last_visible_color_idx"].to_numpy()

    def outer_visible(x):
        # Mirrors taskutils.last_visible_color(time_step_mode="outer", tol=0):
        # "visible" == the ball is NOT within [mask_start - r, mask_end + r].
        return (x < mask_start - r) or (x > mask_end + r)

    assert len(targets) == len(df)
    flips_seen = 0
    for i, tgt in enumerate(targets):
        length_i = int(lengths[i])
        idx = int(lvci[i])
        # Window-arithmetic premise: output_targets length equals the trial length.
        assert len(tgt) == length_i, f"trial {i}: len(target) {len(tgt)} != length {length_i}"
        assert 0 <= idx <= length_i - 1
        x = tgt[:, 0]
        # The recorded last-visible frame must itself be visible (this also guards
        # the all-masked -> silently-returns-(len-1) failure mode: an all-masked
        # trajectory would have x[idx] inside the band and trip this assert).
        assert outer_visible(x[idx]), (
            f"trial {i}: frame[{idx}] x={x[idx]:.2f} is not outer-band visible"
        )
        # When the ball goes on to enter the mask, visibility flips right after idx.
        if idx < length_i - 1:
            assert not outer_visible(x[idx + 1]), (
                f"trial {i}: frame[{idx + 1}] x={x[idx + 1]:.2f} should be masked"
            )
            flips_seen += 1

    # Guard against a vacuous pass (e.g. an all-catch dataset never entering the mask).
    assert flips_seen > 0, "no trial exercised the visibility flip"


# --------------------------------------------------------------------------- #
# Item #3 — adjust_dataset_labels colour/label consistency (human production path)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_item3_adjust_labels_color_consistency():
    """On the real human production path (_adjust_labels=True, n>=60), the adjusted
    labels are internally consistent:

      * correct_response == the RGB-argmax colour mapped through Final Color, i.e.
        the two independent code paths (final_color+1 vs the string->idx map) agree;
      * the colour-rotation columns satisfy color_next = (entered % 3) + 1 and
        color_after_next = (next % 3) + 1;
      * PCCNVC_adjusted equals the per-row effective hazard rate for its Hazard
        Rate bin, with no NaN.

    Does NOT assert change-count columns against final target sums (survey item #8 /
    QUESTIONABLE row a — that ledger is pre-adjustment). estimate_effective_hazard_rates
    is reached only internally here, via the real human total_dataset_length.
    """
    dp, tp = _human_params(seed=99, total_videos=60)
    task, samples, model_samples, targets, df, metadata = hds.generate_video_dataset(
        dp, tp, hds.dict_trial_type_generation_funcs,
        _adjust_labels=True, validate=True,
    )

    # correct_response agrees with the Final Color string map (constants: R->1,G->2,B->3).
    mapped = df["Final Color"].map(default_color_to_idx_dict).to_numpy()
    assert np.array_equal(df["correct_response"].to_numpy(), mapped)
    assert set(df["Final Color"].unique()) <= {"red", "green", "blue"}

    # Colour-rotation columns.
    entered = df["color_entered"].to_numpy()
    assert np.array_equal(df["color_next"].to_numpy(), (entered % 3) + 1)
    assert np.array_equal(
        df["color_after_next"].to_numpy(), (df["color_next"].to_numpy() % 3) + 1
    )

    # PCCNVC_adjusted is the per-row effective hazard rate, never NaN.
    assert not df["PCCNVC_adjusted"].isna().any()
    hz_effective = metadata["hz_effective"]
    assert set(df["Hazard Rate"].dropna().unique()) <= set(hz_effective)
    expected_adjusted = df["Hazard Rate"].map(hz_effective).to_numpy()
    assert np.allclose(df["PCCNVC_adjusted"].to_numpy(), expected_adjusted)
