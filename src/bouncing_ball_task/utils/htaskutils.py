from collections import Counter
from collections.abc import Iterable
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from bouncing_ball_task.utils import pyutils
from bouncing_ball_task.constants import DEFAULT_COLORS
from bouncing_ball_task.bouncing_ball import BouncingBallTask


def print_task_summary(meta, use_logger=True):
    if use_logger:
        out_func = logger.info
    else:
        out_func = print
        
    length_trials = meta["length_trials_ms"] / 60000
    length_trials_min = int(length_trials)
    length_trials_s = np.round((length_trials % 1) * 60, 1)
    out_func("Dataset Generation Summary")
    out_func(
        f'  Num Total Trials: {meta["num_trials"]} ({length_trials_min} min {length_trials_s} sec)'
    )
    max_key_len = max([len(key) for key in meta.keys()])

    for key, val in meta.items():
        if key == "num_trials":
            continue
        key += ":"
        out_func(f"    {key:<{max_key_len + 2}}{val}")


def print_type_stats(
        trials,
        trial_type,
        duration,
        use_logger=True,
        return_str=False,
):
    if use_logger:
        out_funcs = [logger.info, logger.debug]
    else:
        out_funcs = [print] * 2 
    
    position, velocity, color, pccnvc, pccovc, pvc, fxvc, fyvc, meta = zip(*trials)
    stats_comb = [f"{nvc}-{ovc}" for nvc, ovc in zip(pccnvc, pccovc)]
    list_messages = []

    trial_type_lengths_s = (
        np.array([m["length"] for m in meta]) * duration
    ) / 1000
    trial_type_lengths_min = trial_type_lengths_s / 60
    length_trial_total_min = int(sum(trial_type_lengths_min))
    length_trial_total_s = np.round((sum(trial_type_lengths_min) % 1) * 60, 1)

    length_trial_s_min = np.round(min(trial_type_lengths_s), 1)
    length_trial_s_max = np.round(max(trial_type_lengths_s), 1)

    msg = f"  Num {trial_type.title()} Trials: {len(trials)} ({length_trial_total_min} min {length_trial_total_s} sec)"
    if return_str:
        list_messages.append(msg)
    else:
        out_funcs[0](msg)

    trial_type_stats = [
        ("Min Video Length (s):", length_trial_s_min),
        ("Max Video Length (s):", length_trial_s_max),
        ("Color Splits:", Counter(np.argmax(color, axis=1))),
        ("pccnvc Splits:", Counter(pccnvc)),
        ("pccovc Splits:", Counter(pccovc)),
        ("Stat Comb Splits:", Counter(stats_comb)),
    ]

    if trial_type.lower() != "catch":
        trial_type_stats += [
            ("End time Splits:", Counter(m["idx_time"] for m in meta)),
            ("Left/Right Splits:", Counter(m["side_left_right"] for m in meta)),
            ("Top/Bottom Splits:", Counter(m["side_top_bottom"] for m in meta)),
            ("Velocity Splits:", Counter(m["idx_velocity_y"] for m in meta)),
        ]

    max_desc_len = max([len(desc) for (desc, _) in trial_type_stats])
    for (description, value) in trial_type_stats:
        if isinstance(value, Counter):
            total = value.total()
            value_str = f"({', '.join([str(float(np.round(val[1] / total, 2))) for val in sorted(value.items(), key=lambda pair: pair[0])])})"
        else:
            value_str = f"{value}"
        msg = f"    {description:<{max_desc_len + 2}}{value_str}"
        if return_str:
            list_messages.append(msg)
        else:
            out_funcs[1](msg)

    if return_str:
        return list_messages


def print_block_stats(df_data, dict_metadata, duration, use_logger=True):
    if use_logger:
        out_funcs = [logger.info, logger.debug, logger.trace]
    else:
        out_funcs = [print] * 3
    
    # blocks, meta_blocks = zip(*params_blocks)
    num_blocks = dict_metadata['num_blocks']
    block_lengths = [
        block['num_trials'] for _, block in
        dict_metadata['blocks'].items()
    ]
    out_funcs[0](f"  Num blocks: {num_blocks} - Lengths = {block_lengths}")
    
    for idx_block, df_block in df_data.groupby("Dataset Block"):
        metadata_block = dict_metadata["blocks"][idx_block]
        # position, velocity, color, pccnvc, pccovc, pvc, meta = zip(*block)

        stats_comb = [
            f"{nvc}-{ovc}" for nvc, ovc in
            df_block[["PCCNVC", "PCCOVC"]].values
        ]
        # stats_comb = [f"{nvc}-{ovc}" for nvc, ovc in zip(pccnvc, pccovc)]

        assert len(df_block) == metadata_block["num_trials"]

        length_block_min = int(metadata_block["length_block_min"])
        length_block_s_rem = np.round((metadata_block["length_block_min"] % 1) * 60)

        out_funcs[1](
            f"    Block {idx_block} - {len(df_block)} videos "
            f"({length_block_min} min {length_block_s_rem} sec)"
        )

        block_stats = [
            ("Min Video Length (s):", df_block["length_ms"].min() / 1000),
            ("Max Video Length (s):", df_block["length_ms"].max() / 1000),
            ("Trial type Counts:", Counter(df_block["trial"].values)),
            ("Color Counts:", Counter(df_block["Final Color"].values)),
            ("pccnvc Counts:", Counter(df_block["PCCNVC"])),
            ("pccovc Counts:", Counter(df_block["PCCOVC"])),
            ("stats comb Counts:", Counter(stats_comb)),
        ]
        
        max_desc_len = max([len(desc) for (desc, _) in block_stats])
        
        for (description, value) in block_stats:
            if isinstance(value, Counter):
                total = value.total()
                value_str = f"({', '.join([str(float(np.round(val[1] / total, 2))) for val in sorted(value.items(), key=lambda pair: pair[0])])})"
            else:
                value_str = f"{value}"
            out_funcs[2](f"      {description:<{max_desc_len + 2}}{value_str}")


def plot_params(
    params,
    size_frame: Iterable[int] = (256, 256),
    sequence_length=60,
    duration=40,
    target_future_timestep=1,
    ball_radius: int = 10,
    dt: float = 0.1,
    mask_center: float = 0.5,
    mask_fraction: float = 1 / 3,
    mask_color=(127, 127, 127),
    min_t_color_change=15,
    sample_mode="parameter_array",
    target_mode="parameter_array",
    save_target=True,
    sequence_mode="reverse",
    debug=True,
    save_animation=False,
    display_animation=True,
    mode="combined",
    multiplier=2,
    include_timestep=True,
    as_mp4=True,
):

    batch_size = len(params)
    (
        initial_position,
        initial_velocity,
        initial_color,
        pccnvc,
        pccovc,
        pvc,
        meta,
    ) = zip(*params)
    task = BouncingBallTask(
        size_frame=size_frame,
        sequence_length=sequence_length,
        ball_radius=ball_radius,
        target_future_timestep=target_future_timestep,
        dt=dt,
        batch_size=batch_size,
        sample_mode=sample_mode,
        target_mode=target_mode,
        mask_center=mask_center,
        mask_fraction=mask_fraction,
        mask_color=mask_color,
        sequence_mode=sequence_mode,
        debug=debug,
        probability_velocity_change=pvc,
        probability_color_change_no_velocity_change=pccnvc,
        probability_color_change_on_velocity_change=pccovc,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        initial_color=initial_color,
        min_t_color_change=min_t_color_change,
    )
    initial_params = np.concatenate(
        [np.array(initial_position), np.stack(initial_color)],
        axis=1,
    )
    assert np.all(
        np.isclose(np.array(initial_position), task.sequence[-1][1][:, :2])
    )
    assert np.all(
        np.isclose(np.stack(initial_color), task.sequence[-1][1][:, 2:])
    )
    samples, targets = zip(*[x for x in task])
    assert np.all(
        np.isclose(np.array(initial_position), task.sequence[-1][1][:, :2])
    )
    assert np.all(
        np.isclose(np.stack(initial_color), task.sequence[-1][1][:, 2:])
    )

    samples = np.array(samples)
    targets = np.array(targets)

    for i in range(batch_size):
        print(np.array(params[i][0]) * multiplier, params[i][1:])
        assert np.all(
            np.isclose(np.array(initial_position[i]), targets[-1][i, :2])
        )
        assert np.all(
            np.isclose(np.stack(initial_color[i]), targets[-1][i, 2:])
        )

        _, _ = task.animate(
            arrays=(targets[:, i, :]),
            path_dir=None,
            name="",
            duration=duration,
            mode=mode,
            multiplier=multiplier,
            save_target=save_target,
            save_animation=save_animation,
            display_animation=display_animation,
            num_sequences=1,
            as_mp4=as_mp4,
            include_timestep=include_timestep,
            return_path=True,
        )
            

def generate_dataset_name(name_dataset, seed=None):
    if name_dataset is None:
        name_dataset = "hbb_dataset"        
    name_list = [name_dataset, datetime.now().strftime("%y%m%d_%H%M%S")]
    if seed is not None:
        name_list.append(str(seed))
    return "_".join(name_list)
        

def compute_dataset_size(
        exp_scale,
        fixed_video_length,
        video_length_min_s,
        duration,
        total_dataset_length,
        total_videos,
        trial_type_split,
        trial_types,
):
    # Bring exp to same scale as time
    exp_scale_ms = exp_scale * 1000

    # Set the min length to be the fixed length if passed
    if fixed_video_length is not None:
        video_length_min_f = fixed_video_length

    # Set it according to the min number of seconds
    elif video_length_min_s is not None:
        video_length_min_desired_ms = int(np.array(video_length_min_s) * 1000)
        video_length_min_f = np.rint(
            video_length_min_desired_ms / duration
        ).astype(int)

    # Turn into ms
    video_length_min_ms = video_length_min_f * duration

    # Get the parsed trial type splits
    trial_type_split = pyutils.create_sequence_splits(trial_type_split)
        
    # Compute the number of videos we can make given the contraints if not
    # explicitly provided with a total number of videos
    if total_videos is None:
        return compute_dataset_size_time_based(
            total_dataset_length,
            video_length_min_f,
            video_length_min_ms,
            trial_type_split,
            exp_scale_ms,
            duration,
            trial_types,
        )
    else:
        return compute_dataset_size_video_based(
            total_videos,
            video_length_min_f,
            video_length_min_ms,
            trial_type_split,
            fixed_video_length,
            exp_scale_ms,
            duration,
            trial_types,
        )

    
def compute_dataset_size_time_based(
        total_ds_length,
        video_length_min_f,
        video_length_min_ms,
        trial_type_split,
        exp_scale_ms,
        duration,
        trial_types,
        min_length_catch=True,
):
    max_ds_length_ms = remaining_ds_lenth_ms = total_ds_length * 60 * 1000  # * s * ms
    dict_num_trials_type, dict_video_lengths_f_type = {}, {}
    
    for i, p_split in enumerate(trial_type_split):
        # Skip if no trials for this type
        if p_split == 0.0:
            continue
        
        # Grab the trial type
        trial_type = trial_types[i]

        # Define the max possible number of videos for this trial type
        max_length_trial_type_ms = p_split * max_ds_length_ms
        max_video_num = np.rint(max_length_trial_type_ms / video_length_min_ms).astype(int)

        # Sample exponential lengths for this number of videos
        if trial_type == "catch" and min_length_catch:
            max_video_lengths_ms = np.ones(max_video_num) * video_length_min_ms
            
        else:
            max_video_lengths_ms = (
                np.random.exponential(exp_scale_ms, max_video_num)
                + video_length_min_ms
            )
        # Convert ms to frames for each video
        max_video_lengths_f = np.rint(max_video_lengths_ms / duration).astype(int)
        
        # Calculate how much time each video adds to the overall trial type
        cumsum_video_lengths = np.cumsum(max_video_lengths_ms)

        # Find the idx of the first video that doesn't exceed desired dataset length
        length_trial_type_idx = (
            np.where(cumsum_video_lengths < max_length_trial_type_ms)[0][-1] + 1
        )

        # Subselect all videos that add to the dataset length and number of trials
        dict_video_lengths_f_type[trial_type] = max_video_lengths_f[:length_trial_type_idx]
        dict_num_trials_type[trial_type] = len(dict_video_lengths_f_type[trial_type])

    assert sum(sum(v) for _, v in dict_video_lengths_f_type.items()) * duration < max_ds_length_ms

    return dict_num_trials_type, dict_video_lengths_f_type


def compute_dataset_size_video_based(
        total_videos,
        video_length_min_f,
        video_length_min_ms,
        trial_type_split,
        fixed_video_length,
        exp_scale_ms,
        duration,
        trial_types,
        min_length_catch=True,
):
    dict_num_trials_type, dict_video_lengths_f_type = {}, {}

    for i, p_split in enumerate(trial_type_split):
        # Skip if no trials for this type
        if p_split == 0.0:
            continue
        
        # Grab the trial type
        trial_type = trial_types[i]
        dict_num_trials_type[trial_type] = num_trials = np.rint(p_split * total_videos).astype(int)

        # Sample exponential lengths for this number of videos
        if fixed_video_length:
            dict_video_lengths_f_type[trial_type] = np.ones(num_trials).astype(int) * fixed_video_length
        
        elif trial_type == "catch" and min_length_catch:
            dict_video_lengths_f_type[trial_type] = np.ones(num_trials).astype(int) * video_length_min_f
            
        else:
            dict_video_lengths_f_type[trial_type] = np.rint(
                (
                    np.random.exponential(exp_scale_ms, num_trials)
                    + video_length_min_ms
                )
                / duration
            ).astype(int)
    
    return dict_num_trials_type, dict_video_lengths_f_type


def generate_initial_dict_metadata(
        dict_num_trials_type,
        dict_video_lengths_f_type,
        size_frame,
        duration,
        ball_radius,
        dt,
        exp_scale,
        velocity_lower,
        velocity_upper,
        num_y_velocities,
        pvc,
        pccnvc_lower,
        pccnvc_upper,
        num_pccnvc,
        pccovc_lower,
        pccovc_upper,
        num_pccovc,
        mask_fraction,
        mask_center,
        bounce_offset,
        num_pos_x_endpoints,
        num_pos_y_endpoints,
        y_pos_multiplier,
        border_tolerance_outer,
        border_tolerance_inner,
        num_pos_x_linspace_bounce,
        idx_linspace_bounce,
        bounce_timestep,
        repeat_factor,        
        seed,
        min_pos_x_endpoints=2,
        
        **kwargs,
):
    # Convenience
    size_x, size_y = size_frame

    # Start with basic params
    dict_metadata = {
        "ball_radius": ball_radius,
        "dt": dt,
        "duration": duration,
        "exp_scale": exp_scale,
        "border_tolerance_outer": border_tolerance_outer,
        "mask_center": mask_center,
        "mask_fraction": mask_fraction,
        "size_x": size_x,
        "size_y": size_y,
        "num_pos_x_endpoints": num_pos_x_endpoints,
        "num_pos_y_endpoints": num_pos_y_endpoints,
        "y_pos_multiplier": y_pos_multiplier,
        "pvc": pvc,
        "num_y_velocities": num_y_velocities,
        "bounce_offset": bounce_offset,
        "border_tolerance_inner": border_tolerance_inner,
        "border_tolerance_outer": border_tolerance_outer,
        "num_pos_x_linspace_bounce": num_pos_x_linspace_bounce,
        "idx_linspace_bounce": idx_linspace_bounce,
        "bounce_timestep": bounce_timestep,
        "repeat_factor": repeat_factor,
        "seed": seed,
    } | kwargs
    
    # Useful quantities
    dict_metadata["num_trials"] = sum(
        num_trials for _, num_trials in dict_num_trials_type.items()
    )
    
    dict_metadata["length_trials_ms"] = duration * sum(
        sum(video_lengths_f)
        for _, video_lengths_f in dict_video_lengths_f_type.items()
    )
    
    dict_metadata["video_length_max_f"] = video_length_max_f = max(
        max(video_lengths_f)
        for _, video_lengths_f in dict_video_lengths_f_type.items()
        if video_lengths_f.size > 0
    )
    dict_metadata["video_length_max_ms"] = video_length_max_f * duration
    
    dict_metadata["video_length_min_f"] = video_length_min_f = min(
        min(video_lengths_f)
        for _, video_lengths_f in dict_video_lengths_f_type.items()
        if video_lengths_f.size > 0
    )
    dict_metadata["video_length_min_ms"] = video_length_min_f * duration


    # Velocity Linspaces
    dict_metadata["final_velocity_x_magnitude"] = (
        np.mean([velocity_lower, velocity_upper]) * size_x
    )
    dict_metadata["final_velocity_y_magnitude_linspace"] = np.linspace(
        velocity_lower * size_y,
        velocity_upper * size_y,
        num_y_velocities,
        endpoint=True,
    )

    # Probability Linspaces
    dict_metadata["pccnvc_linspace"] = np.linspace(
        pccnvc_lower,
        pccnvc_upper,
        num_pccnvc,
        endpoint=True,
    )
    dict_metadata["pccovc_linspace"] = np.linspace(
        pccovc_lower,
        pccovc_upper,
        num_pccovc,
        endpoint=True,
    )

    # Mask positions
    dict_metadata["mask_size"] = mask_size = int(np.round(size_x * mask_fraction))
    dict_metadata["mask_start"] = mask_start = int(
        np.round((mask_center * size_x) - mask_size / 2)
    )
    dict_metadata["mask_end"] = mask_end = size_x - mask_start

    # Normal trials
    # x position Grayzone linspace
    min_num_pos_x_endpoints = 2    
    if num_pos_x_endpoints is None or num_pos_x_endpoints < min_num_pos_x_endpoints:

        dict_metadata["x_grayzone_linspace_sides"] = np.array(
            [
                [mask_start + (border_tolerance_inner + 1) * ball_radius],
                [mask_end - (border_tolerance_inner + 1) * ball_radius],
            ]
        )
        dict_metadata["x_grayzone_linspace"] = x_grayzone_linspace = dict_metadata["x_grayzone_linspace_sides"][0]
                
    else:
        dict_metadata["x_grayzone_linspace"] = x_grayzone_linspace = np.linspace(
            mask_start + (border_tolerance_inner + 1) * ball_radius,
            mask_end - (border_tolerance_inner + 1) * ball_radius,
            num_pos_x_endpoints,
            endpoint=True,
        )
    
        # Final x position linspaces that correspond to approaching from each side
        x_grayzone_linspace_reversed = x_grayzone_linspace[::-1]
        dict_metadata["x_grayzone_linspace_sides"] = np.vstack(
            [
                x_grayzone_linspace,
                x_grayzone_linspace_reversed,
            ]
        )

    # Get the final y positions depending on num of end x
    dict_metadata["diff"] = np.diff(x_grayzone_linspace).mean()

    # Fill with starting values
    return dict_metadata


def compute_trial_idx_vals(
        num_trials,
        dict_meta,
        dict_meta_trials,
        dict_meta_type,        
):
    repeat_factor = dict_meta["repeat_factor"]
    num_y_velocities = dict_meta["num_y_velocities"]
    
    # Binary arrays for whether the ball ends to the left or right of the grayzone
    # and if it is going towards the top or bottom    
    sides_left_right = pyutils.repeat_sequence(
        np.array([0, 1] * repeat_factor),
        num_trials,
    ).astype(int)
    dict_meta_trials["side_left_right"] = sides_left_right.tolist()
    dict_meta_type["side_left_right_counts"] = np.unique(
        dict_meta_trials["side_left_right"],
        return_counts=True,
    )

    sides_top_bottom = pyutils.repeat_sequence(
        np.array([0, 1] * repeat_factor),
        num_trials,
    ).astype(int)
    dict_meta_trials["side_top_bottom"] = sides_top_bottom.tolist()
    dict_meta_type["side_top_bottom_counts"] = np.unique(
        dict_meta_trials["side_top_bottom"],
        return_counts=True,
    )        

    indices_velocity_y_magnitude = pyutils.repeat_sequence(
        np.array(list(range(num_y_velocities)) * repeat_factor),
        num_trials,
    ).astype(int)
    dict_meta_trials["idx_velocity_y"] = indices_velocity_y_magnitude.tolist()
    dict_meta_type["indices_velocity_y_magnitude_counts"] = np.unique(
        indices_velocity_y_magnitude,
        return_counts=True,
    )    

    return sides_left_right, sides_top_bottom, indices_velocity_y_magnitude, dict_meta_trials, dict_meta_type

def compute_trial_color_and_stats(
        num_trials,
        dict_meta,
        dict_meta_type,
):
    pccnvc_linspace = dict_meta["pccnvc_linspace"]
    pccovc_linspace = dict_meta["pccovc_linspace"]
   
    # Precompute colors
    final_color = pyutils.repeat_sequence(
        np.array(DEFAULT_COLORS),
        num_trials,
        shuffle=False,
        roll=True,
        shift=1,
    ).tolist()

    # Keep track of color counts
    dict_meta_type["final_color_counts"] = np.unique(
        final_color,
        return_counts=True,
        axis=0,
    )

    # Precompute the statistics
    pccnvc = pyutils.repeat_sequence(
        pccnvc_linspace,
        # np.tile(pccnvc_linspace, num_pccovc),
        num_trials,
        shuffle=False,
        roll=True,
        # roll=False if num_pccovc % num_pccnvc else True,
    ).tolist()
    pccovc = pyutils.repeat_sequence(
        pccovc_linspace,
        # np.tile(pccovc_linspace, num_pccnvc),
        num_trials,
        shuffle=False,
    ).tolist()

    # Keep track of pcc counts
    dict_meta_type["pccnvc_counts"] = np.unique(
        pccnvc,
        return_counts=True,
    )
    dict_meta_type["pccovc_counts"] = np.unique(
        pccovc,
        return_counts=True,
    )
    dict_meta_type["pccnvc_pccovc_counts"] = np.unique(
        [x for x in zip(*(pccnvc, pccovc))],
        return_counts=True,
        axis=0,
    )

    return final_color, pccnvc, pccovc, dict_meta_type    
    

def group_trial_data(
        num_trials,
        final_position,
        final_velocity,
        final_color,
        pccnvc,
        pccovc,
        pvc,
        bounce_index_x=None,
        bounce_index_y=None,
        color_change_index=None,
        dict_meta_trials=None,
):
    if bounce_index_x is None:
        bounce_index_x = [[],] * num_trials

    if bounce_index_y is None:
        bounce_index_y = [[],] * num_trials

    if color_change_index is None:
        color_change_index = [[],] * num_trials

    if not pyutils.isiterable(pvc):
        pvc = [pvc,] * num_trials

    list_to_zip = [
        final_position,
        final_velocity,
        final_color,
        pccnvc,
        pccovc,
        pvc,
        bounce_index_x,
        bounce_index_y,
        color_change_index,
    ]

    if dict_meta_trials is not None:
        list_to_zip.append([
            dict(zip(dict_meta_trials, values))
            for values in zip(*dict_meta_trials.values())
        ])

    for i, val in enumerate(list_to_zip):
        assert len(val) == num_trials, f"Length mismatch for list_to_zip at [{i}]"

    return list(zip(*list_to_zip))

        
