"""Dataset generation orchestrator for color-controlled bouncing ball experiments.

Trajectories are a pure function of explicit-or-sampled initial conditions
(PVC=0), kept (control the start) or reversed (control the end). Velocity
change channels are the derived, read-only bounce ledger; variants author only
the color change channels, subject to the contingency invariant.
"""
import copy
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from bouncing_ball_task import index
from bouncing_ball_task.bouncing_ball import BouncingBallTask
from bouncing_ball_task.constants import default_ball_colors
from bouncing_ball_task.human_bouncing_ball import dataset as hds
from bouncing_ball_task.utils import logutils, pyutils, htaskutils, taskutils
from bouncing_ball_task.color_controlled_bouncing_ball import defaults
from bouncing_ball_task.color_controlled_bouncing_ball.no_change import generate_no_change_trials


# Dictionary of available trial generation functions
# Import each trial type generator and add to this dict
dict_trial_type_generation_funcs = {
    "no_change": generate_no_change_trials,
}


NUM_COLORS = 3  # RGB initial colors / rotations

# Change-vector channel indices inside targets (N, T, 9)
CH_VC_BOUNCE, CH_VC_RANDOM, CH_CC_BOUNCE, CH_CC_RANDOM = 5, 6, 7, 8


def resolve_task_parameters(task_parameters, defaults_module=defaults):
    """Merge user task parameters UNDER the color-controlled defaults.

    The defaults subclass the human task's parameters, so anything the caller
    does not override inherits the real-task value (dt, sequence_length,
    color_mask_mode="outer", initial_timestep_is_changepoint=False, ...).
    """
    dc = defaults_module.ColorControlledTaskParameters()
    resolved = {key: getattr(dc, key) for key in dc.keys}
    resolved.update(task_parameters or {})

    pvc = resolved.get("probability_velocity_change", 0.0)
    if pvc is None or np.any(np.asarray(pvc) != 0):
        raise ValueError(
            "Color-controlled datasets require probability_velocity_change=0 "
            f"(trajectories are a pure function of initial conditions); got {pvc!r}"
        )

    itic = resolved.get("initial_timestep_is_changepoint", False)
    if itic is not False:
        raise ValueError(
            "Color-controlled datasets require initial_timestep_is_changepoint=False "
            "(frame 0 must carry no change flags, as in the real task; a True frame-0 "
            "bounce flag would let a variant author a contingent color change there "
            f"and would break ch6 == 0); got {itic!r}"
        )

    # Sub-batch attributes are set per sub-batch / per preset task
    for key in ("batch_size", "sequence_mode", "seed", "samples", "targets",
                "initial_position", "initial_velocity"):
        resolved.pop(key, None)

    return resolved


def _as_optional_array(value, name, num_base_sequences, dtype=float):
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype)
    if array.shape != (num_base_sequences, 2):
        raise ValueError(
            f"{name} must have shape ({num_base_sequences}, 2), got {array.shape}"
        )
    return array


def validate_initial_positions(initial_position, size_frame, ball_radius):
    """Explicit starts must lie STRICTLY inside [r, size - r] on both axes.

    An out-of-bounds start produces a bounce flagged at t=1 that the reverse
    shift drops, so it is rejected outright.
    """
    size = np.asarray(size_frame, dtype=float)
    lower = np.full(2, ball_radius, dtype=float)
    upper = size - ball_radius
    inside = np.all((initial_position > lower) & (initial_position < upper), axis=1)
    if not np.all(inside):
        bad = np.flatnonzero(~inside)
        raise ValueError(
            f"initial_position rows {bad.tolist()} are not strictly inside "
            f"[{ball_radius}, size-{ball_radius}] = x in ({lower[0]}, {upper[0]}), "
            f"y in ({lower[1]}, {upper[1]}): {initial_position[bad].tolist()}"
        )


def resolve_initial_conditions(
    base_task_parameters,
    num_base_sequences,
    initial_position=None,
    initial_velocity=None,
):
    """Explicit-else-sampled resolution into explicit (N, 2) arrays.

    Sampling uses the task's own samplers (``sample_position`` /
    ``sample_velocity(initial_sample=True)``) via a throwaway 2-frame task built
    with ``seed=False`` so the single dataset seed is never disturbed. Whichever
    array is explicit is passed INTO the sampler so, e.g., a sampled velocity
    sign points away from the grayzone relative to the EXPLICIT position.
    Sampled positions are never inside the grayzone (the sampler's contract).
    """
    sampler = BouncingBallTask(
        **{**base_task_parameters, "sequence_length": 2},
        batch_size=num_base_sequences,
        sequence_mode="static",
        seed=False,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
    )
    return (
        np.array(sampler.initial_position, dtype=float),
        np.array(sampler.initial_velocity, dtype=float),
    )


def generate_base_trajectories(
    base_task_parameters,
    initial_position,
    initial_velocity,
    control_end,
):
    """Integrate the base trajectories from explicit initial conditions.

    ``sequence_mode`` is a whole-batch attribute, so the trials are split into
    a forward ("static": trial STARTS at the initial position) and a reverse
    ("reverse": trial ENDS at the initial position) sub-batch, each built with
    ``seed=False`` and the explicit arrays, then scattered back into the
    original trial order.

    Returns:
        (targets (N, T, 9), list_of_base_tasks)
    """
    num_base_sequences = len(initial_position)
    sequence_length = base_task_parameters["sequence_length"]
    targets = np.zeros((num_base_sequences, sequence_length, 9))
    base_tasks = []

    for mode, selector in (("static", ~control_end), ("reverse", control_end)):
        indices = np.flatnonzero(selector)
        if len(indices) == 0:
            continue
        task = BouncingBallTask(
            **base_task_parameters,
            batch_size=len(indices),
            sequence_mode=mode,
            seed=False,
            initial_position=initial_position[indices],
            initial_velocity=initial_velocity[indices],
        )
        targets[indices] = task.targets
        base_tasks.append(task)

    return targets, base_tasks


def validate_color_events(color_events, bounce_frames, variant_name):
    """Enforce the contingency invariant on the authored color events.

    For every trial and frame: ``ch7[t]=1 => ch5[t]=1`` (contingent color
    changes only where a real wall bounce occurred) and ``ch7[t]+ch8[t] <= 1``
    (the real task never fires both). Raises ValueError on violation.
    """
    expected_shape = bounce_frames.shape + (2,)
    color_events = np.asarray(color_events)
    if color_events.shape != expected_shape:
        raise ValueError(
            f"Variant '{variant_name}' must return color events of shape "
            f"{expected_shape} (channels [cc_bounce, cc_random]); got {color_events.shape}"
        )
    if not np.all(np.isin(color_events, (0, 1))):
        raise ValueError(f"Variant '{variant_name}' color events must be binary (0/1)")

    cc_bounce = color_events[:, :, 0].astype(bool)
    orphan = cc_bounce & ~bounce_frames
    if np.any(orphan):
        trials, frames = np.nonzero(orphan)
        raise ValueError(
            f"Variant '{variant_name}' fires a contingent color change (ch7) "
            f"where no bounce occurred (ch5=0) at (trial, frame) pairs "
            f"{list(zip(trials.tolist(), frames.tolist()))[:10]}"
        )

    double = color_events.sum(axis=-1) > 1
    if np.any(double):
        trials, frames = np.nonzero(double)
        raise ValueError(
            f"Variant '{variant_name}' fires both contingent and random color "
            f"changes on the same frame at (trial, frame) pairs "
            f"{list(zip(trials.tolist(), frames.tolist()))[:10]}"
        )

    return color_events.astype(float)


def generate_color_controlled_dataset(
    controlled_dataset_parameters,
    task_parameters,
    shuffle=True,
    dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
    defaults_module=defaults,
):
    """Generate a color-controlled bouncing ball dataset.

    Pipeline:
    1. Seed once; resolve explicit-else-sampled initial conditions into
       explicit (N, 2) arrays.
    2. Integrate base trajectories (PVC=0) from those arrays: forward
       ("static", controls the start) or reversed ("reverse", controls the end)
       per ``control_end``. Velocity channels 5/6 are the derived, read-only
       bounce ledger.
    3. Each variant authors ONLY the color channels 7/8 from the positions and
       the bounce ledger; the contingency invariant is enforced.
    4. Colors are built from the authored events (cumsum + RGB rotation, three
       rotations per base x variant).
    5. Samples are masked with the task's own ``color_mask`` (outer mode) and
       wrapped in a preset task; ``task.model_samples`` is the model input.

    Args:
        controlled_dataset_parameters: Dataset generation parameters
            (``num_base_sequences``, ``seed``, ``duration``,
            ``variable_length`` + its human-task knobs ``video_length_min_s``,
            ``exp_scale``, ``fixed_video_length``, ``initial_position``,
            ``initial_velocity``, ``control_end``). Under ``variable_length``
            the task's ``sequence_length`` becomes the longest sampled trial.
        task_parameters: Task-parameter overrides merged under
            ``ColorControlledTaskParameters``.
        shuffle: Whether to shuffle the generated trials
        dict_trial_type_generation_funcs: Dictionary mapping variant names to
            color-authoring functions ``f(positions, bounce_frames, params)``.
        defaults_module: Module containing default parameters

    Returns:
        tuple: (task, output_samples, output_model_samples, output_targets, df_data, dict_metadata)
    """
    # Extract parameters
    num_base_sequences = controlled_dataset_parameters.get('num_base_sequences', 100)
    total_videos = controlled_dataset_parameters.get('total_videos', None)
    duration = controlled_dataset_parameters.get('duration', defaults_module.duration)
    variable_length = controlled_dataset_parameters.get('variable_length', False)
    _dataset_defaults = defaults_module.ColorControlledDatasetParameters
    video_length_min_s = controlled_dataset_parameters.get(
        'video_length_min_s', _dataset_defaults.video_length_min_s)
    exp_scale = controlled_dataset_parameters.get('exp_scale', _dataset_defaults.exp_scale)
    fixed_video_length = controlled_dataset_parameters.get(
        'fixed_video_length', _dataset_defaults.fixed_video_length)

    num_variants = len(dict_trial_type_generation_funcs)
    num_colors = NUM_COLORS

    if total_videos is not None:
        num_base_sequences = total_videos // (num_variants * num_colors)
        logger.info(f"Calculated {num_base_sequences} base sequences for {total_videos} total videos")

    # Resolve the explicit-else-sampled controls (shape-checked, not sampled yet)
    initial_position = _as_optional_array(
        controlled_dataset_parameters.get('initial_position'),
        'initial_position', num_base_sequences,
    )
    initial_velocity = _as_optional_array(
        controlled_dataset_parameters.get('initial_velocity'),
        'initial_velocity', num_base_sequences,
    )
    control_end = controlled_dataset_parameters.get('control_end')
    control_end = (
        np.zeros(num_base_sequences, dtype=bool) if control_end is None
        else np.asarray(control_end, dtype=bool)
    )
    if control_end.shape != (num_base_sequences,):
        raise ValueError(
            f"control_end must have shape ({num_base_sequences},), got {control_end.shape}"
        )
    if variable_length and not np.all(control_end):
        raise ValueError(
            "variable_length=True keeps the LAST `length` frames of a trial, which "
            "destroys a controlled start; it requires control_end=True for every trial"
        )

    # Task parameters: defaults (human-task parity) under the caller's overrides
    base_task_parameters = resolve_task_parameters(task_parameters, defaults_module)
    if initial_position is not None:
        validate_initial_positions(
            initial_position,
            base_task_parameters['size_frame'],
            base_task_parameters['ball_radius'],
        )

    # --- Seeding contract (ordered; nothing reseeds in between) ---------------
    # 1. Seed once. seed=None draws a fresh seed; seed=False is the
    #    external-seeding escape hatch (caller seeded; resolved_seed is None).
    seed = controlled_dataset_parameters.get('seed')
    resolved_seed = None if seed is False else pyutils.set_global_seed(seed)
    initial_rng_state = np.random.get_state()

    # 1b. Sample the per-trial lengths FIRST, as the human/model datasets do
    #     (HDS:257), and integrate at the longest sampled length (HDS:87) so
    #     every trial can keep its LAST `length` frames. Fixed length draws
    #     nothing, so the fixed-length RNG stream is unchanged.
    num_trials = num_base_sequences * num_variants * num_colors
    lengths = resolve_video_lengths(
        num_trials,
        base_task_parameters['sequence_length'],
        variable_length,
        duration,
        video_length_min_s=video_length_min_s,
        exp_scale=exp_scale,
        fixed_video_length=fixed_video_length,
    )
    if variable_length:
        base_task_parameters['sequence_length'] = int(lengths.max())

    # 2. Resolve explicit-else-sampled into explicit (N, 2) arrays.
    initial_position, initial_velocity = resolve_initial_conditions(
        base_task_parameters,
        num_base_sequences,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
    )

    # 3. Build both sub-batch tasks from the explicit arrays with seed=False.
    base_targets, base_tasks = generate_base_trajectories(
        base_task_parameters, initial_position, initial_velocity, control_end,
    )
    ref_task = base_tasks[0]  # For mask geometry / color_mask

    # The derived, read-only ledger
    positions = base_targets[:, :, :2]  # (N, T, 2)
    velocity_changes = base_targets[:, :, CH_VC_BOUNCE:CH_CC_BOUNCE].copy()  # (N, T, 2)
    bounce_frames = velocity_changes[:, :, 0].astype(bool)  # ch5, (N, T)

    # --- Variants author the color channels only --------------------------------
    variant_params = {
        'controlled_dataset_parameters': controlled_dataset_parameters,
        'task_parameters': copy.deepcopy(base_task_parameters),
        'initial_position': initial_position,
        'initial_velocity': initial_velocity,
        'control_end': control_end,
    }

    all_change_vectors = []
    all_metadata = []
    variant_names = []

    for variant_name, variant_func in dict_trial_type_generation_funcs.items():
        color_events = variant_func(
            positions.copy(),
            bounce_frames.copy(),
            copy.deepcopy(variant_params),
        )
        color_events = validate_color_events(color_events, bounce_frames, variant_name)

        # Full change vector = derived velocity ledger + authored color events
        all_change_vectors.append(
            np.concatenate([velocity_changes, color_events], axis=-1)  # (N, T, 4)
        )
        all_metadata.append(
            generate_variant_metadata(variant_name, variant_func, color_events, num_colors)
        )
        variant_names.extend([variant_name] * num_base_sequences)

    change_array_variants = np.concatenate(all_change_vectors, axis=0)  # (N*V, T, 4)

    # --- Color sequences from the authored events -----------------------------
    color_change_sequences = np.any(change_array_variants[:, :, -2:], axis=-1)  # (N*V, T)
    color_sequences_base = np.cumsum(color_change_sequences, axis=1)  # (N*V, T)

    all_color_sequences = []
    all_positions = []
    all_change_vectors_final = []
    all_variant_names = []
    all_base_indices = []
    all_color_indices = []

    for color_start in range(num_colors):
        color_sequences_categorical = (color_start + color_sequences_base) % num_colors
        color_sequences_rgb = np.eye(3)[color_sequences_categorical] * 255  # (N*V, T, 3)

        all_color_sequences.append(color_sequences_rgb)
        all_positions.append(np.tile(positions, (num_variants, 1, 1)))
        all_change_vectors_final.append(change_array_variants)
        all_variant_names.extend(variant_names)

        base_indices = np.tile(np.arange(num_base_sequences), num_variants)
        all_base_indices.extend(base_indices)
        all_color_indices.extend([color_start] * len(base_indices))

    final_positions = np.concatenate(all_positions, axis=0)  # (N*V*C, T, 2)
    final_colors = np.concatenate(all_color_sequences, axis=0)  # (N*V*C, T, 3)
    final_change_vectors = np.concatenate(all_change_vectors_final, axis=0)  # (N*V*C, T, 4)

    final_targets = np.concatenate(
        [final_positions, final_colors, final_change_vectors], axis=-1
    )  # (N*V*C, T, 9)

    # --- Samples: the task's own (outer) grayzone mask ----------------------------
    num_trials, sequence_length = final_targets.shape[:2]
    masked_colors = ref_task.color_mask(
        final_positions.reshape(-1, 2), final_colors.reshape(-1, 3)
    ).reshape(num_trials, sequence_length, 3)
    final_samples = np.concatenate([final_positions, masked_colors], axis=-1)  # (N*V*C, T, 5)

    # Per-trial bookkeeping aligned with the final trial order
    all_base_indices = np.asarray(all_base_indices)
    all_color_indices = np.asarray(all_color_indices)
    all_variant_names = np.asarray(all_variant_names, dtype=object)
    all_control_end = control_end[all_base_indices]

    # --- Shuffle FIRST, so every row describes its own arrays ---------------------
    # The dataframe's ``idx_trial`` must index the arrays actually returned, and
    # ``save_video_dataset`` indexes the arrays by the dataframe's row position
    # (HDS:895-898). Permuting the arrays and the per-trial bookkeeping together
    # BEFORE anything describes them makes that alignment structural rather than
    # a post-hoc repair. The permutation draws from the single dataset seed.
    if shuffle:
        order = np.random.permutation(num_trials)
        final_samples = final_samples[order]
        final_targets = final_targets[order]
        all_base_indices = all_base_indices[order]
        all_color_indices = all_color_indices[order]
        all_variant_names = all_variant_names[order]
        all_control_end = all_control_end[order]

    # --- Preset task -------------------------------------------------------------
    # These are the FINAL task parameters stored in metadata: valid
    # BouncingBallTask kwargs with seed=False so a loader rebuilding the task
    # never reseeds its process's global RNG (HDS:84,152-154,419).
    final_task_parameters = copy.deepcopy(base_task_parameters)
    final_task_parameters['batch_size'] = num_trials
    final_task_parameters['sequence_mode'] = 'preset'
    final_task_parameters['seed'] = False

    task = BouncingBallTask(
        **final_task_parameters,
        samples=final_samples,
        targets=final_targets
    )

    # Model input: outer-masked samples with the true color restored in the
    # transition band (BB:844-857) — what the real task writes to *_samples.csv
    model_samples = task.model_samples

    # --- length / duration contract (C2) -----------------------------------------
    # `length` is a FRAME COUNT, `duration` is ms PER FRAME (human default 50),
    # `length_ms = length * duration`. Variable-length trials keep their LAST
    # `length` frames (HDS:448-450, 833-835), which is why they are restricted to
    # control_end=True. `lengths` were sampled at step 1b (i.i.d. per trial slot,
    # so they need no permuting with the shuffle above).
    assert lengths.shape == (num_trials,) and lengths.max() <= sequence_length

    if variable_length:
        output_samples = [s[-l:] for s, l in zip(final_samples, lengths)]
        output_model_samples = [m[-l:] for m, l in zip(model_samples, lengths)]
        output_targets = [t[-l:] for t, l in zip(final_targets, lengths)]
    else:
        output_samples = final_samples
        output_model_samples = model_samples
        output_targets = final_targets

    dict_metadata = generate_color_controlled_metadata(
        task,
        controlled_dataset_parameters,
        final_task_parameters,
        all_metadata,
        num_base_sequences,
        num_variants,
        num_colors,
        dict_trial_type_generation_funcs,
        lengths=lengths,
        duration=duration,
        variable_length=variable_length,
        resolved_seed=resolved_seed,
        rng_state=initial_rng_state,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        control_end=control_end,
    )

    df_data, dict_metadata = generate_color_controlled_dataframe(
        output_samples,
        output_targets,
        all_variant_names,
        all_base_indices,
        all_color_indices,
        lengths,
        all_control_end,
        initial_velocity,
        dict_metadata,
        controlled_dataset_parameters,
        final_task_parameters,
        duration,
    )

    return task, output_samples, output_model_samples, output_targets, df_data, dict_metadata


def resolve_video_lengths(
    num_trials,
    sequence_length,
    variable_length,
    duration,
    video_length_min_s=None,
    exp_scale=None,
    fixed_video_length=None,
):
    """Per-trial frame counts (the dataframe's ``length``), mirroring the
    human and model datasets.

    Fixed length => every trial is the full ``sequence_length``. Variable
    length => the human task's own sampler
    (``htaskutils.compute_dataset_size_video_based``, the one
    ``human_bouncing_ball`` and ``model_bouncing_ball`` draw from):
    ``frames = rint((Exp(exp_scale s) + video_length_min_s) / duration)``,
    rejecting draws of ``max_length_mult`` (3x) the minimum or more; a truthy
    ``fixed_video_length`` pins every trial to that many frames. The caller
    then integrates at ``sequence_length = max(lengths)`` (HDS:87) and each
    trial keeps its LAST ``length`` frames (HDS:448-450), which is why
    ``variable_length`` requires ``control_end=True``.
    """
    if not variable_length:
        return np.full(num_trials, sequence_length, dtype=int)
    video_length_min_f = int(np.rint(video_length_min_s * 1000 / duration))
    _, dict_lengths = htaskutils.compute_dataset_size_video_based(
        num_trials,
        video_length_min_f,
        video_length_min_f * duration,
        trial_type_split=[1.0],
        fixed_video_length=fixed_video_length,
        exp_scale_ms=exp_scale * 1000,
        duration=duration,
        trial_types=("color_controlled",),
    )
    return np.asarray(dict_lengths["color_controlled"], dtype=int)


def generate_variant_metadata(variant_name, variant_func, color_events, num_colors):
    """Derive per-variant metadata from the authored events.

    A variant may attach a ``variant_metadata`` dict to its function; it is
    merged on top of the derived fields.
    """
    num_trials = color_events.shape[0]
    meta = {
        "variant_name": variant_name,
        "num_trials": num_trials,
        "num_trials_with_rotations": num_trials * num_colors,
        "authored_changes": {
            "color_change_bounce": int(color_events[:, :, 0].sum()),
            "color_change_random": int(color_events[:, :, 1].sum()),
        },
    }
    extra = getattr(variant_func, "variant_metadata", None)
    if extra:
        meta.update(copy.deepcopy(extra))
    return meta


def generate_color_controlled_dataset_with_videos(
    controlled_dataset_parameters,
    task_parameters,
    output_dir,
    shuffle=True,
    dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
    defaults_module=defaults,
):
    """Generate controlled dataset and save videos - convenience function for tests.
    
    This function generates a controlled dataset and saves the videos,
    useful for tests that expect video files to be created.
    
    Args:
        controlled_dataset_parameters: Dictionary of dataset generation parameters
        task_parameters: Dictionary of task-specific parameters
        output_dir: Directory to save the dataset
        shuffle: Whether to shuffle the generated trials
        dict_trial_type_generation_funcs: Dictionary mapping trial types to generation functions
        defaults_module: Module containing default parameters
        
    Returns:
        tuple: (task, output_samples, output_model_samples, output_targets, df_data, dict_metadata)
    """
    # Generate the dataset
    task, output_samples, output_model_samples, output_targets, df_data, dict_metadata = generate_color_controlled_dataset(
        controlled_dataset_parameters,
        task_parameters,
        shuffle=shuffle,
        dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
        defaults_module=defaults_module,
    )
    
    # Save videos
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name = dict_metadata["name"]
        
        path_videos = hds.save_video_dataset(
            output_dir,
            dataset_name,
            df_data,
            dict_metadata,
            output_samples,
            output_model_samples,
            output_targets,
            task,
            duration=controlled_dataset_parameters.get("duration", defaults_module.ColorControlledDatasetParameters().duration),
            mode=defaults_module.mode,
            multiplier=defaults_module.multiplier,
            save_target=True,
            save_animation=True,
            display_animation=defaults_module.display_animation,
            num_sequences=1,
            as_mp4=True,
            include_timestep=defaults_module.include_timestep,
            return_path=True,
            dryrun=False,
        )
        
        dict_metadata["dataset_dir"] = output_dir / dataset_name
    
    return task, output_samples, output_model_samples, output_targets, df_data, dict_metadata


def generate_color_controlled_metadata(
    task,
    controlled_dataset_parameters,
    task_parameters,
    variant_metadata,
    num_base_sequences,
    num_variants,
    num_colors,
    dict_trial_type_generation_funcs,
    lengths=None,
    duration=None,
    variable_length=False,
    resolved_seed=None,
    rng_state=None,
    initial_position=None,
    initial_velocity=None,
    control_end=None,
):
    """Generate metadata dictionary for controlled dataset.

    Besides the color-controlled bookkeeping, this populates the keys the reused
    real-task helpers READ before they are called (``duration``,
    ``video_length_min_f``, ``ball_radius``, ``mask_start``, ``mask_end`` —
    HDS:345-374, 501, 558-559) plus the ``generate_initial_dict_metadata`` keys
    that apply to a color-controlled dataset.

    Args:
        task: BouncingBallTask instance
        controlled_dataset_parameters: Dataset parameters
        task_parameters: The FINAL preset task parameters (valid
            ``BouncingBallTask`` kwargs with ``seed=False``, ``batch_size``,
            ``sequence_mode="preset"``) — a loader rebuilds the task from them
        initial_position / initial_velocity: resolved explicit (N, 2) arrays
        control_end: resolved (N,) bool array
        variant_metadata: List of metadata from each variant
        num_base_sequences: Number of base sequences
        num_variants: Number of variants
        num_colors: Number of color rotations
        
    Returns:
        dict: Complete metadata dictionary
    """
    # Generate timestamp and seed
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    # P0-3: record the seed actually resolved by the task, never a false 0.
    # resolved_seed is correct for all three contract states: a real int for
    # seed=None/seed=int, and None for the seed=False escape hatch. Fall back to
    # the configured seed only when no resolved seed was threaded at all.
    seed = resolved_seed if resolved_seed is not None \
        else controlled_dataset_parameters.get('seed')
    
    # Generate name based on variants (via the real task's name helper)
    variant_str = "_".join(sorted(dict_trial_type_generation_funcs.keys()))
    name = htaskutils.generate_dataset_name(
        f"color_controlled_{variant_str}", seed=seed,
    )

    num_trials = int(task.batch_size)
    if lengths is None:
        lengths = np.full(num_trials, task.sequence_length, dtype=int)
    lengths = np.asarray(lengths, dtype=int)
    if duration is None:
        duration = controlled_dataset_parameters.get("duration", defaults.duration)

    video_length_max_f = int(lengths.max())
    video_length_min_f = int(lengths.min())

    dict_metadata = {
        # Basic info
        "name": name,
        "seed": seed,
        "resolved_seed": resolved_seed,
        "provenance": pyutils.capture_provenance(rng_state=rng_state),
        "dataset_type": "controlled",
        "timestamp": timestamp,
        "total_trials": num_base_sequences * num_variants * num_colors,

        # --- Keys the reused real-task helpers read (and real-task parity) -----
        # `duration` is ms PER FRAME; `length` is a frame count (C2).
        "duration": duration,
        "dt": task.dt,
        "ball_radius": task.ball_radius,
        "size_x": int(task.size_frame[0]),
        "size_y": int(task.size_frame[1]),
        "mask_center": task.mask_center,
        "mask_fraction": task.mask_fraction,
        "mask_size": int(task.mask_end - task.mask_start),
        "mask_start": int(task.mask_start),
        "mask_end": int(task.mask_end),
        "pvc": task_parameters.get("probability_velocity_change", 0.0),
        "transition_tol": task_parameters.get("transition_tol"),
        "variable_length": variable_length,
        "num_trials": num_trials,
        "length_trials_ms": int(lengths.sum()) * duration,
        "video_length_max_f": video_length_max_f,
        "video_length_max_ms": video_length_max_f * duration,
        "video_length_min_f": video_length_min_f,
        "video_length_min_ms": video_length_min_f * duration,

        # Dataset generation parameters
        "controlled_dataset_parameters": controlled_dataset_parameters,
        "task_parameters": task_parameters,
        
        # Controlled-specific metadata
        "controlled_parameters": {
            "generation_method": "change_vector_manipulation",
            "num_base_sequences": num_base_sequences,
            "num_variants": num_variants,
            "num_colors": num_colors,
            "variants": list(dict_trial_type_generation_funcs.keys()),
            "base_sequence_reuse": num_variants * num_colors,
            # Resolved initial conditions per base sequence
            "initial_position": None if initial_position is None else np.asarray(initial_position),
            "initial_velocity": None if initial_velocity is None else np.asarray(initial_velocity),
            "control_end": None if control_end is None else np.asarray(control_end, dtype=bool),
        },
        
        # Generation process info
        "base_generation_info": {
            "method": "Base trajectories from explicit initial conditions (PVC=0); "
                      "static (control start) or reverse (control end); "
                      "variants author color channels 7/8 only",
            "change_vector_shape": (num_base_sequences, task.sequence_length, 4),
            "color_sequence_generation": "Cumulative sum of authored color events with rotation",
            "mask_application": f"BouncingBallTask color_mask, mode={task.color_mask_mode}",
        },
    }
    
    # Add variant-specific metadata
    for i, (variant_name, meta) in enumerate(zip(dict_trial_type_generation_funcs.keys(), variant_metadata)):
        dict_metadata[variant_name] = {
            "num_trials": num_base_sequences * num_colors,
            "variant_metadata": meta,
            "task_parameters": copy.deepcopy(task_parameters),
        }
    
    return dict_metadata


def generate_color_controlled_dataframe(
    output_samples,
    output_targets,
    variant_names,
    base_indices,
    color_indices,
    lengths,
    control_end,
    initial_velocity,
    dict_metadata,
    controlled_dataset_parameters,
    task_parameters,
    duration,
    num_blocks=1,
):
    """Build the trial dataframe in the REAL task's on-disk schema.

    The per-trial rows carry only what is genuinely color-controlled; every
    derived column is produced by the real task's own helpers via
    ``hds.generate_dataset_metadata`` (HDS:322-419), which in order:
    int-coerces the ``idx_*`` sentinels, adds ``last_visible_color*`` /
    ``color_entered`` / ``color_next`` / ``color_after_next``
    (``taskutils.last_visible_color``), renames ``idx`` -> ``idx_trial``, names
    the index ``Video ID``, assigns blocks (``generate_blocks_from_data_df``),
    adds the effective / observable change statistics and the ``Hazard Rate`` /
    ``Contingency`` categories (``compute_effective_stats`` ->
    ``compute_change_stats``), and maps ``correct_response``.

    Columns that are pure human-task sampling artifacts are emitted as their
    real sentinels, split by dtype (v3_2_2 snapshot): ``idx_time`` /
    ``idx_position`` / ``idx_velocity_y`` are int64 ``-1``; ``idx_x_position`` /
    ``side_left_right`` / ``side_top_bottom`` are float64 ``NaN``.

    ``PCCNVC_adjusted`` and ``cwl`` are omitted by design (``adjust_dataset_labels``
    artifacts nothing in the loader or the callback reads).

    Args:
        output_samples / output_targets: per-trial arrays AS RETURNED (already
            truncated under ``variable_length``); row i describes trial i.
        variant_names / base_indices / color_indices / control_end: per-trial
            bookkeeping in the same order.
        lengths: per-trial frame counts (the ``length`` column).
        initial_velocity: resolved (num_base_sequences, 2) array — reversed
            trials take ``Final X/Y Velocity = -initial_velocity`` (HDS:458-459).
        dict_metadata: mutated in place by the helpers.
        task_parameters: the FINAL preset parameters (stored by the helper).
        duration: milliseconds PER FRAME.
        num_blocks: single block by default — blocks are mandatory, the LSTM
            loader cannot read ``save_video_dataset``'s block-less fallback
            (HDS:922-925).

    Returns:
        (df_trial_metadata, dict_metadata)
    """
    num_trials = len(output_targets)
    lengths = np.asarray(lengths, dtype=int)
    base_indices = np.asarray(base_indices, dtype=int)
    control_end = np.asarray(control_end, dtype=bool)
    initial_velocity = np.asarray(initial_velocity, dtype=float)
    dt = task_parameters["dt"]

    row_data = []
    for i in range(num_trials):
        target = np.asarray(output_targets[i])
        sample = np.asarray(output_samples[i])

        rgb = target[-1, 2:5]
        if np.count_nonzero(rgb) != 1 or not np.isclose(rgb.max(), 255):
            raise ValueError(f"Unrecognized final RGB value in trial {i}: {rgb}")
        final_color = default_ball_colors[int(np.argmax(rgb))]

        # `trial` is a design label in the human task; for a color-controlled
        # trial it is DERIVED from the trajectory's bounce content, capitalised
        # as on disk, so AggregatedLoggingCallbackV2's trial == "Bounce" rows exist.
        trial = "Bounce" if np.any(target[:, CH_VC_BOUNCE]) else "Straight"

        if control_end[i]:
            # Reverse trials end at the controlled point with the human
            # convention -initial_velocity (HDS:458-459).
            final_velocity = -initial_velocity[base_indices[i]]
        else:
            # Static trials: the last step already uses the post-bounce velocity.
            final_velocity = (target[-1, :2] - target[-2, :2]) / dt

        row_data.append({
            # Core identification (`idx` is renamed to `idx_trial` by the helper)
            "idx": i,
            "trial": trial,
            "length": int(lengths[i]),
            "Final Color": final_color,
            "Final X Position": float(sample[-1, 0]),
            "Final Y Position": float(sample[-1, 1]),
            "Final X Velocity": float(final_velocity[0]),
            "Final Y Velocity": float(final_velocity[1]),
            # True color-change probabilities: the colors are authored, not sampled
            "PCCNVC": 0.0,
            "PCCOVC": 0.0,
            "PVC": float(task_parameters.get("probability_velocity_change", 0.0)),
            "length_ms": int(lengths[i]) * duration,
            # Human-task sampling artifacts, by dtype (v3_2_2 snapshot)
            "idx_time": -1,
            "side_left_right": np.nan,
            "side_top_bottom": np.nan,
            "idx_velocity_y": -1,
            "idx_position": -1,
            "idx_x_position": np.nan,
            # Color-controlled extras (tolerated by the parity spec)
            "variant": variant_names[i],
            "base_sequence_idx": int(base_indices[i]),
            "color_rotation_idx": int(color_indices[i]),
            "control_end": bool(control_end[i]),
        })

    validate_last_visible_color(output_targets, lengths, dict_metadata)

    # compute_effective_type_stats (HDS:667-668) writes into one lower-cased
    # sub-dict per `trial` value present; create them before the helper runs.
    for trial in {row["trial"] for row in row_data}:
        dict_metadata.setdefault(trial.lower(), {})

    # `duration` must be explicit: generate_dataset_metadata forwards
    # dataset_parameters.get("duration") and a None silently drops the *_ps columns.
    dataset_parameters = dict(controlled_dataset_parameters)
    dataset_parameters["duration"] = duration

    df_data, dict_metadata = hds.generate_dataset_metadata(
        row_data,
        dict_metadata,
        dataset_parameters,
        task_parameters,
        output_samples=output_samples,
        output_targets=output_targets,
        num_blocks=num_blocks,
    )

    df_data["Total Changes"] = (
        df_data["Bounces"]
        + df_data["Random Bounces"]
        + df_data["Color Change Bounce"]
        + df_data["Color Change Random"]
    ).astype(int)

    return df_data, dict_metadata


def validate_last_visible_color(output_targets, lengths, dict_metadata):
    """Reject trajectories whose last visible frame is frame 0.

    ``compute_effective_stats`` slices each change sequence as
    ``target[:last_visible_color_idx]`` (HDS:546-549); a zero-length slice makes
    ``compute_change_stats`` divide by ``timesteps == 0`` (HDS:628,638,643),
    producing NaN statistics and RuntimeWarnings rather than an error. That
    happens when the ball is outside the outer grayzone band only at frame 0 —
    unreachable for SAMPLED starts (the sampler never draws inside the grayzone
    and starts land far from the band) but reachable with an explicit start just
    outside ``[mask_start - r, mask_end + r]`` moving inward. A trajectory that
    never leaves the band at all is not degenerate: ``last_visible_color``
    returns ``T - 1`` for an all-inside mask.
    """
    min_length = dict_metadata["video_length_min_f"]
    window = np.stack([
        np.asarray(target)[-min_length:, :5] for target in output_targets
    ])
    _, last_idx = taskutils.last_visible_color(
        window,
        dict_metadata["ball_radius"],
        dict_metadata["mask_start"],
        dict_metadata["mask_end"],
        time_step_mode="outer",
        return_index=True,
    )
    last_visible_color_idx = last_idx + np.asarray(lengths, dtype=int) - min_length
    degenerate = np.flatnonzero(last_visible_color_idx <= 0)
    if len(degenerate):
        raise ValueError(
            f"trials {degenerate.tolist()} are last visible at frame "
            f"{last_visible_color_idx[degenerate].tolist()}: the ball leaves the "
            f"outer grayzone band [{dict_metadata['mask_start']} - r, "
            f"{dict_metadata['mask_end']} + r] no later than frame 0, so the "
            "effective change statistics would be computed over an empty window. "
            "Start further from the grayzone or use a longer sequence."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Inferred args from the dictionaries
    parser = pyutils.add_dataclass_args(parser, defaults.ColorControlledTaskParameters)
    parser = pyutils.add_dataclass_args(parser, defaults.ColorControlledDatasetParameters)

    # Manual additions
    parser.add_argument("--dir_base", type=Path, default=index.dir_repo/"data/raw/bb_datasets/color_controlled")
    parser.add_argument("--name_dataset", default=defaults.name_dataset)
    parser.add_argument("--display_animation", default=defaults.display_animation)
    parser.add_argument("--mode", type=str, default=defaults.mode)
    parser.add_argument("--multiplier", type=int, default=defaults.multiplier)
    parser.add_argument("--include_timestep", default=defaults.include_timestep)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    # Setup the logger
    logger = logutils.configure_logger(verbose=args.verbose, trace=args.debug)
    dir_base = Path(args.dir_base)
    
    task_parameters = {
        key: getattr(args, key) for key in defaults.ColorControlledTaskParameters.keys
    }
    controlled_dataset_parameters = {
        key: getattr(args, key) for key in defaults.ColorControlledDatasetParameters.keys
    }

    # Generate the dataset
    task, output_samples, output_model_samples, output_targets, df_data, dict_metadata = generate_color_controlled_dataset(
        controlled_dataset_parameters,
        task_parameters,
        shuffle=False,
    )

    # Generate dataset name
    dict_metadata["name"] = name_dataset = htaskutils.generate_dataset_name(
        args.name_dataset,
        seed=dict_metadata["seed"],
    )    

    # Save the dataset
    path_videos = hds.save_video_dataset(
        dir_base,
        name_dataset,
        df_data,
        dict_metadata,
        output_samples,
        output_model_samples,
        output_targets,
        task,
        duration=args.duration,
        mode=args.mode,
        multiplier=args.multiplier,
        save_target=True,
        save_animation=True,
        display_animation=args.display_animation,
        num_sequences=1,
        as_mp4=True,
        include_timestep=args.include_timestep,
        return_path=True,
        dryrun=args.dryrun,
    )
    
    logger.info(f"Color controlled dataset saved to: {dir_base / name_dataset}")
