"""Model-condition geometry characterization tests (coverage survey items #6, #7).

These cover the six model trial-type generators that define the model dataset's
experimental conditions (`cc_nvc`, `cc_vc`, `cc_rvc`, `ncc_nvc`, `ncc_vc`,
`ncc_rvc`). None of them was previously tested for its own geometry — the
determinism suite only proves the pipeline runs reproducibly, and `validate=True`
only checks each trial reaches its *requested* endpoint, not that the endpoint is
*correct*. Every assertion here is a content/geometry characterization.

Reverse-mode fact (survey): the human/model tasks run ``sequence_mode="reverse"``
with ``target_future_timestep=0`` (``human_bouncing_ball/defaults.TaskParameters``).
``final_position``/``final_velocity`` handed to a generator is where the trial
ENDS, not starts. All per-type geometry below is phrased on the trajectory END.

Scope discipline (survey C0 / dispatch): these tests NEVER call
``generate_model_dataset_nongray`` and NEVER run the model funcs through the
``_adjust_labels=True`` / ``estimate_effective_hazard_rates`` path (which is
broken at clean HEAD). Item #6 is parameter-level via ``generate_video_parameters``
(no task build). Item #7 builds a small dataset with ``_adjust_labels=False`` only
(the same call the existing model determinism test uses).
"""
import numpy as np
import pytest

from bouncing_ball_task.constants import DEFAULT_COLORS
from bouncing_ball_task.human_bouncing_ball import dataset as hds
from bouncing_ball_task.model_bouncing_ball import defaults as mdefaults
from bouncing_ball_task.model_bouncing_ball.cc_nvc import generate_cc_nvc_trials
from bouncing_ball_task.model_bouncing_ball.cc_vc import generate_cc_vc_trials
from bouncing_ball_task.model_bouncing_ball.cc_rvc import generate_cc_rvc_trials
from bouncing_ball_task.model_bouncing_ball.ncc_nvc import generate_ncc_nvc_trials
from bouncing_ball_task.model_bouncing_ball.ncc_vc import generate_ncc_vc_trials
from bouncing_ball_task.model_bouncing_ball.ncc_rvc import generate_ncc_rvc_trials


# Insertion order fixes the trial-type split order; total_videos=24 with the
# model default split (1,1,1,1,1,1) => 4 trials per type.
_MODEL_TRIAL_FUNCS = {
    "cc_nvc": generate_cc_nvc_trials,
    "cc_vc": generate_cc_vc_trials,
    "cc_rvc": generate_cc_rvc_trials,
    "ncc_nvc": generate_ncc_nvc_trials,
    "ncc_vc": generate_ncc_vc_trials,
    "ncc_rvc": generate_ncc_rvc_trials,
}

# 10-tuple layout returned per trial by htaskutils.group_trial_data:
#   0 final_position, 1 final_velocity, 2 final_color, 3 pccnvc, 4 pccovc,
#   5 pvc, 6 bounce_index_x, 7 bounce_index_y, 8 color_change_index, 9 meta_dict
IDX_POS, IDX_VEL, IDX_COLOR = 0, 1, 2
IDX_BOUNCE_X, IDX_BOUNCE_Y, IDX_COLOR_CHANGE = 6, 7, 8
IDX_META = 9

# Which forced-change fields each type carries (verified against the source):
#   cc_*  carry a scalar color_change_index (forced contingent colour change);
#   *_rvc carry a scalar bounce_index_x (forced open-space velocity change);
#   every other slot is [] (unset).  NOTE: the survey (#6) states "*_vc/*_rvc
#   carry a bounce_index_x" — that is inaccurate: only *_rvc do (see report).
_CARRIES_COLOR_CHANGE = {"cc_nvc", "cc_vc", "cc_rvc"}
_CARRIES_BOUNCE_X = {"cc_rvc", "ncc_rvc"}


def _model_params(seed, total_videos=24):
    """Local model dataset-parameter builder (mirrors the private ``_human_params``
    in ``test_reproducibility.py`` but with the model ``NongrayDatasetParameters``).

    Kept LOCAL to this file per the parallel-implementer contract."""
    dataset_params = {
        key: getattr(mdefaults.NongrayDatasetParameters(), key)
        for key in mdefaults.NongrayDatasetParameters.keys
    }
    dataset_params["seed"] = seed
    dataset_params["total_videos"] = total_videos  # small; 6 types => tv/6 each
    dataset_params["print_stats"] = False
    dataset_params["use_logger"] = False
    task_params = {
        key: getattr(mdefaults.TaskParameters(), key)
        for key in mdefaults.TaskParameters.keys
    }
    return dataset_params, task_params


def _expected_pos_x_sides(dict_meta):
    """Reconstruct the two allowed END x-positions per velocity-change class from
    dict_meta primitives, mirroring the generators (which read
    ``dict_meta["timestep_change"] + 1`` and ``ncc_nvc_timesteps + 1``)."""
    br = dict_meta["ball_radius"]
    sx = dict_meta["size_x"]
    dt = dict_meta["dt"]
    bti = dict_meta["border_tolerance_inner"]
    mask_start = dict_meta["mask_start"]
    mask_end = dict_meta["mask_end"]
    fvxm = dict_meta["final_velocity_x_magnitude"]
    timestep_change = dict_meta["timestep_change"] + 1
    timestep_from_wall = dict_meta["timestep_from_wall"]

    dx = fvxm * dt
    distance_x_from_bounce = timestep_change * dx
    distance_x_from_wall = timestep_from_wall * dx

    return {
        # nvc: ball ends just outside the grayzone, far from walls (no bounce)
        "nvc": np.array([
            mask_start - bti * br,
            mask_end + bti * br,
        ]),
        # vc: ball ends near a wall (a natural wall bounce is the velocity change)
        "vc": np.array([
            br - dx / 2 + distance_x_from_bounce,
            sx - br + dx / 2 - distance_x_from_bounce,
        ]),
        # rvc: ball ends in open space, pushed a further timestep_from_wall*dx
        # away from the wall than vc (the forced change is in open space)
        "rvc": np.array([
            br + distance_x_from_bounce + distance_x_from_wall,
            sx - br - distance_x_from_bounce - distance_x_from_wall,
        ]),
    }


def _vc_class(trial_type):
    # order matters: "nvc" and "rvc" both end in "vc", so test them first
    if trial_type.endswith("rvc"):
        return "rvc"
    if trial_type.endswith("nvc"):
        return "nvc"
    assert trial_type.endswith("vc"), trial_type
    return "vc"


# --------------------------------------------------------------------------- #
# Item #6 — model condition geometry, parameter-level (no task build).
# --------------------------------------------------------------------------- #

def _generate_params(seed=7, total_videos=24):
    dp, _ = _model_params(seed, total_videos=total_videos)
    dict_trials, dict_meta = hds.generate_video_parameters(
        **dp, dict_trial_type_generation_funcs=_MODEL_TRIAL_FUNCS
    )
    return dict_trials, dict_meta


def test_model_condition_trial_counts():
    """Item #6: the model default split (1,1,1,1,1,1) at total_videos=24 gives an
    equal 4 trials per type across all six conditions."""
    dict_trials, dict_meta = _generate_params()
    per_type = dict_meta["num_trials"] // len(_MODEL_TRIAL_FUNCS)
    assert per_type == 4
    for ttype in _MODEL_TRIAL_FUNCS:
        assert len(dict_trials[ttype]) == per_type, ttype
    assert sum(len(dict_trials[t]) for t in _MODEL_TRIAL_FUNCS) == dict_meta["num_trials"]


def test_model_condition_end_x_geometry():
    """Item #6: each type's END x-position is one of exactly two class-specific
    values (nvc just-outside-grayzone / vc at-wall / rvc open-space), matching the
    value selected by the trial's own ``side_left_right`` index."""
    dict_trials, dict_meta = _generate_params()
    pos_x_sides = _expected_pos_x_sides(dict_meta)
    for ttype in _MODEL_TRIAL_FUNCS:
        sides = pos_x_sides[_vc_class(ttype)]
        for trial in dict_trials[ttype]:
            meta = trial[IDX_META]
            end_x = trial[IDX_POS][0]
            expected = sides[meta["side_left_right"]]
            assert np.isclose(end_x, expected), (
                f"{ttype}: end_x={end_x} not the side-{meta['side_left_right']} "
                f"value {expected} (allowed {sides.tolist()})"
            )


def test_model_condition_rvc_further_from_wall_than_vc():
    """Item #6: the ACTUAL generated rvc END positions sit further from the
    nearest wall than the vc END positions by at least the open-space push
    ``distance_x_from_wall = timestep_from_wall * final_velocity_x_magnitude * dt``.

    This reads the real trajectory end positions out of ``dict_trials`` (not a
    test-local recompute), so it bites when the rvc generators collapse onto vc
    geometry. The audit's naive "min(rvc) > max(vc)" does NOT bite that collapse:
    dropping ``distance_x_from_wall`` from the rvc formula still leaves rvc a
    residual ``dx/2`` further from the wall than vc, so a bare inequality stays
    green. Requiring the gap to cover the full ``distance_x_from_wall`` is the
    discriminator: gap >= push  <=>  the rvc change frame sits >=
    ``timestep_from_wall`` steps into open space relative to the vc wall-bounce
    point. GEOMETRY ONLY (QUESTIONABLE (b)): the change channel is never read."""
    dict_trials, dict_meta = _generate_params()
    sx = dict_meta["size_x"]
    push = (
        dict_meta["timestep_from_wall"]
        * dict_meta["final_velocity_x_magnitude"]
        * dict_meta["dt"]
    )
    assert push > 0  # anti-vacuity: a zero push would make the test trivial

    def dist_to_wall(x):
        return min(x, sx - x)

    rvc_dists = [
        dist_to_wall(trial[IDX_POS][0])
        for ttype in ("cc_rvc", "ncc_rvc")
        for trial in dict_trials[ttype]
    ]
    vc_dists = [
        dist_to_wall(trial[IDX_POS][0])
        for ttype in ("cc_vc", "ncc_vc")
        for trial in dict_trials[ttype]
    ]
    assert rvc_dists and vc_dists  # anti-vacuity: both classes are populated

    gap = min(rvc_dists) - max(vc_dists)
    assert gap >= push - 1e-9, (
        f"rvc END wall-distance exceeds vc END wall-distance by only {gap} "
        f"(< open-space push {push}); rvc geometry has collapsed toward vc"
    )


def test_model_condition_end_y_geometry():
    """Item #6: each type's END y-position is exactly the entry its own
    (side_top_bottom, idx_y_positions) indices select from the per-type
    final_y_positions_linspace, whose shape is (2, num_pos_y_endpoints, num_trials)."""
    dict_trials, dict_meta = _generate_params()
    for ttype in _MODEL_TRIAL_FUNCS:
        fyl = np.asarray(dict_meta[ttype]["final_y_positions_linspace"])
        assert fyl.shape == (2, dict_meta["num_pos_y_endpoints"], len(dict_trials[ttype]))
        # Independent clause (not same-source): the wall-closest endpoint of each
        # per-type linspace is the fixed ball_radius+1 (top) / size_y-ball_radius-1
        # (bottom), derived from dict_meta primitives, so a linspace-formula change
        # is caught rather than being invisible to the self-consistent indexing.
        br, sy = dict_meta["ball_radius"], dict_meta["size_y"]
        assert np.all(fyl[0, 0, :] == br + 1), ttype
        assert np.all(fyl[1, 0, :] == sy - br - 1), ttype
        for i, trial in enumerate(dict_trials[ttype]):
            meta = trial[IDX_META]
            end_y = trial[IDX_POS][1]
            expected = fyl[meta["side_top_bottom"], meta["idx_y_positions"], i]
            assert np.isclose(end_y, expected), f"{ttype} trial {i}: y {end_y} != {expected}"


def test_model_condition_positions_within_frame():
    """Item #6: every END position lies inside the frame [0,size_x] x [0,size_y]."""
    dict_trials, dict_meta = _generate_params()
    sx, sy = dict_meta["size_x"], dict_meta["size_y"]
    for ttype in _MODEL_TRIAL_FUNCS:
        for trial in dict_trials[ttype]:
            x, y = trial[IDX_POS]
            assert 0 <= x <= sx, f"{ttype}: x={x} out of frame"
            assert 0 <= y <= sy, f"{ttype}: y={y} out of frame"


def test_model_condition_final_color_from_default_colors():
    """Item #6: every trial's final colour is one of the three DEFAULT_COLORS."""
    dict_trials, _ = _generate_params()
    default_colors = [list(c) for c in DEFAULT_COLORS]
    for ttype in _MODEL_TRIAL_FUNCS:
        for trial in dict_trials[ttype]:
            assert list(trial[IDX_COLOR]) in default_colors, ttype


def test_model_condition_forced_change_fields_presence():
    """Item #6: the forced-change fields are present/absent per type, and carry
    the expected scalar frame index.

    cc_* carry a scalar color_change_index; *_rvc carry a scalar bounce_index_x;
    every other forced slot is the empty-list sentinel. The forced index value is
    ``timestep_change + 2`` (the generators use ``dict_meta["timestep_change"]+1``
    then add 1). bounce_index_y is never set by any of these six generators."""
    dict_trials, dict_meta = _generate_params()
    expected_index = dict_meta["timestep_change"] + 2
    for ttype in _MODEL_TRIAL_FUNCS:
        for trial in dict_trials[ttype]:
            bounce_x = trial[IDX_BOUNCE_X]
            bounce_y = trial[IDX_BOUNCE_Y]
            color_change = trial[IDX_COLOR_CHANGE]

            assert bounce_y == [], f"{ttype}: bounce_index_y should be unset, got {bounce_y!r}"

            if ttype in _CARRIES_COLOR_CHANGE:
                assert color_change == expected_index, (
                    f"{ttype}: color_change_index {color_change!r} != {expected_index}"
                )
            else:
                assert color_change == [], f"{ttype}: color_change_index should be unset"

            if ttype in _CARRIES_BOUNCE_X:
                assert bounce_x == expected_index, (
                    f"{ttype}: bounce_index_x {bounce_x!r} != {expected_index}"
                )
            else:
                assert bounce_x == [], f"{ttype}: bounce_index_x should be unset"


def test_model_condition_warmup_overrides():
    """Item #6: every model type sets the no-random-change warmup overrides to
    ncc_nvc_timesteps+1 for both velocity and colour."""
    dict_trials, dict_meta = _generate_params()
    expected_warmup = dict_meta["ncc_nvc_timesteps"] + 1
    for ttype in _MODEL_TRIAL_FUNCS:
        overrides = dict_meta[ttype]["overrides"]
        assert overrides["warmup_t_no_rand_velocity_change"] == expected_warmup, ttype
        assert overrides["warmup_t_no_rand_color_change"] == expected_warmup, ttype


# --------------------------------------------------------------------------- #
# Item #7 — rvc forced mid-arena velocity change: GEOMETRY ONLY (needs targets).
# --------------------------------------------------------------------------- #
# One small dataset build (_adjust_labels=False), shared by the geometry test and
# the xfail below. This is the same call the existing model determinism test uses;
# it does NOT touch the broken generate_model_dataset_nongray / _adjust_labels
# model path (survey C0).

_SMALL_SEED = 7
_small_dataset_cache = {}


def _small_dataset():
    if "out" not in _small_dataset_cache:
        dp, tp = _model_params(_SMALL_SEED)
        out = hds.generate_video_dataset(
            dp, tp, _MODEL_TRIAL_FUNCS,
            _adjust_labels=False, validate=False, shuffle=False, defaults=mdefaults,
        )
        _small_dataset_cache["out"] = out
    return _small_dataset_cache["out"]


def _open_space_x_flips(targets, ball_radius, size_x):
    """Frames where x-velocity (from consecutive positions) reverses sign while the
    ball is in open space — further than ball_radius from the nearest wall-bounce
    line (x==ball_radius or x==size_x-ball_radius). A wall bounce sits at distance
    ~0; a forced open-space change sits well inside."""
    x = np.asarray(targets)[:, 0]
    vx = np.diff(x)
    flips = np.where(np.sign(vx[:-1]) * np.sign(vx[1:]) < 0)[0] + 1
    open_flips = [
        int(f) for f in flips
        if min(x[f] - ball_radius, (size_x - ball_radius) - x[f]) > ball_radius
    ]
    return open_flips


def _rows_of(df, trial_type):
    return df.index[df["trial"] == trial_type].tolist()


def test_rvc_forced_change_is_in_open_space():
    """Item #7 (geometry only): each rvc trial has EXACTLY ONE open-space
    x-velocity change (the forced mid-arena bounce), located ~timestep_change
    frames from the trajectory end; vc trials have ZERO open-space x-changes (their
    velocity change happens AT a wall). The complementary vc check is what makes
    this non-trivial: it proves the discriminator separates rvc from vc.

    GEOMETRY ONLY per survey QUESTIONABLE (b): asserts the change position is away
    from walls, never which change channel is set (the channel is mis-ledgered;
    see the xfail below). The position-derived flip lands at end-offset
    ``-(timestep_change+2)`` while the ch5 ledger fires one frame later, so nothing
    here is tied to an exact ledger frame."""
    _, _, _, out_targets, df, meta = _small_dataset()
    br, sx = meta["ball_radius"], meta["size_x"]
    expected_offset = -(meta["timestep_change"] + 2)

    for ttype in ("cc_rvc", "ncc_rvc"):
        rows = _rows_of(df, ttype)
        assert rows, f"no {ttype} rows"
        for i in rows:
            tgt = np.asarray(out_targets[i])
            T = tgt.shape[0]
            open_flips = _open_space_x_flips(tgt, br, sx)
            assert len(open_flips) == 1, (
                f"{ttype} trial {i}: expected exactly one open-space velocity "
                f"change, got {open_flips}"
            )
            f = open_flips[0]
            x = tgt[f, 0]
            # not within ~ball_radius of any wall (the rvc-vs-vc separator)
            assert min(x - br, (sx - br) - x) > br, (
                f"{ttype} trial {i}: change at x={x} too close to a wall"
            )
            # sits ~timestep_change frames from the end (exactly -12 today)
            assert abs((f - T) - expected_offset) <= 1, (
                f"{ttype} trial {i}: open flip offset {f - T} != {expected_offset}"
            )

    # Complement: vc trials change AT the wall, so no open-space x-change exists.
    for ttype in ("cc_vc", "ncc_vc"):
        for i in _rows_of(df, ttype):
            open_flips = _open_space_x_flips(np.asarray(out_targets[i]), br, sx)
            assert open_flips == [], (
                f"{ttype} trial {i}: unexpected open-space velocity change {open_flips}"
            )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "survey #7 / QUESTIONABLE (b): the forced rvc mid-arena velocity change is "
        "INTENDED as a random velocity change (targets channel 6, vc_random), but "
        "the runtime ledgers it as a wall bounce (channel 5, vc_bounce). Left red "
        "on purpose; flips green when the ledger is corrected. Do NOT weaken."
    ),
)
def test_rvc_forced_change_intended_in_channel_6():
    """Item #7 documentation (xfail, NOT enshrined): the INTENDED behaviour is that
    each rvc trial records its one forced velocity change in channel 6 (vc_random).
    Under pvc=0 no random velocity change has any other source, so the intended
    count is exactly one per rvc trial. Today channel 6 is empty (the change is
    mis-ledgered into channel 5), so this fails and is expected to xfail. The
    assert references ONLY the intended channel 6 — never channel 5 — so a green
    flip would mean the ledger was corrected, not that the test was relaxed."""
    _, _, _, out_targets, df, _ = _small_dataset()
    for ttype in ("cc_rvc", "ncc_rvc"):
        for i in _rows_of(df, ttype):
            ch6_random_vc = int(np.asarray(out_targets[i])[:, 6].sum())
            assert ch6_random_vc == 1, (
                f"{ttype} trial {i}: intended one random velocity change in channel "
                f"6, found {ch6_random_vc}"
            )
