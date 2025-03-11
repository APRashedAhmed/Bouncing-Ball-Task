import copy
import argparse
import copy
import itertools
import pickle
import random
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from bouncing_ball_task import index
from bouncing_ball_task.constants import (
    CONSTANT_COLOR,
    DEFAULT_COLORS,
    default_ball_colors,
    default_color_to_idx_dict,
)
from bouncing_ball_task.bouncing_ball import BouncingBallTask
from bouncing_ball_task.utils import logutils, pyutils, taskutils, htaskutils
from bouncing_ball_task.human_bouncing_ball.catch import generate_catch_trials
from bouncing_ball_task.human_bouncing_ball.straight import generate_straight_trials
from bouncing_ball_task.human_bouncing_ball.bounce import generate_bounce_trials


dict_trial_type_generation_funcs = {
    "catch": generate_catch_trials,
    "straight": generate_straight_trials,
    "bounce": generate_bounce_trials,
}
trial_types = tuple(key for key, _ in dict_trial_type_generation_funcs.items())


def generate_video_dataset(
    human_video_parameters,
    task_parameters,    
    shuffle=True,
    validate=True,
    num_blocks=None,
    variable_length=True
):
    *params, dict_metadata = generate_video_parameters(
        **human_video_parameters
    )
    # Assuming each parameter in params is a list of tuples, and you want to flatten and separate them
    params_flattened = list(itertools.chain.from_iterable(params))
    num_params = len(params_flattened)

    # Shuffle if asked
    if shuffle:
        random.shuffle(params_flattened)
    
    positions, velocities, colors, pccnvcs, pccovcs, pvcs, meta_trials = (
        list(param) for param in zip(*params_flattened)
    )

    # Grab relevant variables
    max_length = dict_metadata["video_length_max_f"]
    task_parameters = copy.deepcopy(task_parameters)
    task_parameters["initial_position"] = positions
    task_parameters["initial_velocity"] = velocities
    task_parameters["initial_color"] = colors
    task_parameters["probability_velocity_change"] = pvcs
    task_parameters["probability_color_change_no_velocity_change"] = pccnvcs
    task_parameters["probability_color_change_on_velocity_change"] = pccovcs
    task_parameters["pccnvc_lower"] = None
    task_parameters["pccnvc_upper"] = None
    task_parameters["pccovc_lower"] = None
    task_parameters["pccovc_upper"] = None
    task_parameters["batch_size"] = len(positions)
    task_parameters["sequence_mode"] = "reverse"
    task_parameters["target_future_timestep"] = 0
    task_parameters["sequence_length"] = max_length
    task_parameters["sample_velocity_discretely"] = True
    task_parameters["initial_velocity_points_away_from_grayzone"] = True
    task_parameters["initial_timestep_is_changepoint"] = False
    task_parameters["sample_mode"] = "parameter_array"
    task_parameters["target_mode"] = "parameter_array"
    task_parameters["return_change"] = True
    task_parameters["return_change_mode"] = "source"
    task_parameters["warmup_t_no_rand_velocity_change"] = 20
    task_parameters["warmup_t_no_rand_color_change"] = max(
        3, task_parameters.get("warmup_t_no_rand_color_change", 0)
    )
    
    task = BouncingBallTask(**task_parameters)

    samples = task.samples
    targets = task.targets

    # Update metadata
    dict_metadata["min_t_color_change"] = task.min_t_color_change
    output_data, output_samples, output_targets, timesteps, change_sums = shorten_trials_and_update_meta(
        params_flattened,
        samples,
        targets,
        human_task_parameters["duration"],
        variable_length=variable_length,
    )

    df_data, dict_metadata = generate_data_df(
        output_data,
        dict_metadata,
        targets,
        num_blocks=num_blocks,
    )

    df_data = add_effective_stats_to_df(df_data, timesteps, change_sums)

    if validate:
        assert np.all(np.isclose(np.array(positions), targets[:, -1, :2]))
        assert np.all(np.isclose(np.stack(colors), targets[:, -1, 2:5]))
    
    return task, output_samples, output_targets, df_data, dict_metadata


def generate_video_parameters(
    size_frame: Iterable[int] = (256, 256),
    ball_radius: int = 10,
    dt: float = 0.1,
    video_length_min_s: Optional[float] = 8.0, # seconds
    fixed_video_length: Optional[int] = None, # frames
    duration: Optional[int] = 45, # ms
    total_dataset_length: Optional[int] = 35,  # minutes
    mask_center: float = 0.5,
    mask_fraction: float = 1 / 3,
    num_pos_endpoints: int = 5,
    velocity_lower: float = 1 / 12.5,
    velocity_upper: float = 1 / 7.5,
    num_y_velocities: int = 2,
    p_catch_trials: float = 0.05,
    pccnvc_lower: float = 0.00575,
    pccnvc_upper: float = 0.0575,
    pccovc_lower: float = 0.025,
    pccovc_upper: float = 0.975,
    num_pccnvc: int = 2,
    num_pccovc: int = 3,
    pvc: float = 0.0,
    border_tolerance_outer: float = 1.25,
    border_tolerance_inner: float = 1.0,
    bounce_straight_split: float = 0.5,
    bounce_offset: float = 2 / 5,
    total_videos: Optional[int] = None,
    exp_scale: float = 1.0,  # seconds
    print_stats: bool = True,
    use_logger: bool = True,
    seed: Optional[int] = None,
):
    """Generates parameters for video simulations of a bouncing ball in a
    bounded frame.

    Parameters
    ----------
    size_frame : Iterable[int], default=(256, 256)
        Dimensions of the video frame (width, height).

    ball_radius : int, default=10
        Radius of the ball in pixels.

    dt : float, default=0.1
        Time step for the simulation in seconds.

    video_length_min_s : Optional[float], default=8.0
        Length of each video in seconds.

    fixed_video_length : Optional[int], default=None
    	Imposes every video have a fixed length of `fixed_video_length` frames.
    
    duration : Optional[int], default=45
        Duration of each video segment within the total video in frames.

    total_dataset_length : Optional[int], default=35
        Total length of the dataset in minutes.

    mask_center : float, default=0.5
        Center position of the mask as a fraction of frame height.

    mask_fraction : float, default=1/3
        Fraction of the frame height covered by the mask.

    mask_center : float, default=0.5
        The center x position of the mask.

    num_pos_endpoints : int, default=5
        Number of positive endpoints for the ball trajectory.

    velocity_lower : float, default=1/12.5
        Lower limit for the ball's velocity.

    velocity_upper : float, default=1/7.5
        Upper limit for the ball's velocity.

    num_y_velocities : int, default=2
        Number of distinct velocities in the y-direction.

    p_catch_trials : float, default=0.05
        Probability of trials where the ball catches (stops moving).

    pccnvc_lower : float, default=0.01
        Lower probability of a color change not accompanying a velocity change.

    pccnvc_upper : float, default=0.045
        Upper probability of a color change not accompanying a velocity change.

    pccovc_lower : float, default=0.1
        Lower probability of a color change on a velocity change.

    pccovc_upper : float, default=0.9
        Upper probability of a color change on a velocity change.

    num_pccnvc : int, default=2
        Number of instances where a color change does not accompany a velocity
        change.

    num_pccovc : int, default=3
        Number of instances where a color change accompanies a velocity change.

    pvc : float, default=0.0
        Probability of a velocity change occurring independently.

    border_tolerance_outer : float, default=2.0
        Tolerance for the ball touching the outer border of the frame.

    border_tolerance_inner : float, default=1.0
        Tolerance for the ball touching the inner border of the frame.

    bounce_straight_split : float, default=0.5
        Probability of the ball bouncing straight back upon collision.

    bounce_offset : float, default=0.8
        Offset applied to the ball's bounce direction, expressed as a fraction
        of the ball radius.

    total_videos : Optional[int], default=None
        Total number of videos to generate, if specified.

    exp_scale : float, default=1.0
        Scale factor applied to exponential distributions for time-related
        parameters.

    print_stats : bool, default=True
    	Print stats for each trial type.

    use_logger : bool, default=True,
    	Use the logger methods or print function to display stats

    seed : Optional[int], default=None
    	Random seed to use for generating videos

    Returns
    -------
    tuple : (trials_catch, trials_straight, trials_bounce, meta)
        A tuple containing a list of video parameters and a dictionary
        containing metadata about the dataset.
    """
    # Set the seed
    seed = pyutils.set_global_seed(seed)
    
    # Convenience
    dict_num_trials_type, dict_video_lengths_f_type = compute_dataset_size(
        exp_scale,
        fixed_video_length,
        video_length_min_s,
        duration,
        total_dataset_length,
        total_videos,
        p_catch_trials,
        bounce_straight_split,
    )

    dict_metadata = generate_initial_dict_metadata(
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
        num_pos_endpoints,
        border_tolerance_outer,
        border_tolerance_inner,
        seed,
    )

    if print_stats:
        htaskutils.print_task_summary(dict_metadata, use_logger=use_logger)
        
    list_trials_all = []
    
    for trial_type, trial_generator_func in dict_trial_type_generation_funcs.items():
        if dict_num_trials_type[trial_type] > 0:
            trials, dict_metadata[trial_type] = trial_generator_func(
                dict_num_trials_type[trial_type],
                dict_metadata,
                dict_video_lengths_f_type[trial_type],
                print_stats=print_stats,
                use_logger=use_logger,
            )
            list_trials_all.append(trials)
    
    return *list_trials_all, dict_metadata


def compute_dataset_size(
        exp_scale,
        fixed_video_length,
        video_length_min_s,
        duration,
        total_dataset_length,
        total_videos,
        p_catch_trials,
        bounce_straight_split,
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
        
    # Compute the number of videos we can make given the contraints if not
    # explicitly provided with a total number of videos
    if total_videos is None:
        return compute_dataset_size_time_based(
            total_dataset_length,
            p_catch_trials,
            video_length_min_f,
            video_length_min_ms,
            bounce_straight_split,
            exp_scale_ms,
            duration,
        )
    else:
        return compute_dataset_size_video_based(
            total_videos,
            p_catch_trials,
            video_length_min_f,
            video_length_min_ms,
            bounce_straight_split,
            fixed_video_length,
            exp_scale_ms,
            duration,
        )

    
def compute_dataset_size_time_based(
        total_dataset_length,
        p_catch_trials,
        video_length_min_f,
        video_length_min_ms,
        bounce_straight_split,
        exp_scale_ms,
        duration,
):
    max_dataset_length_ms = total_dataset_length * 60 * 1000  # * s * ms

    # Compute the lengths of catch trials
    length_ms_rough_catch = max_dataset_length_ms * p_catch_trials
    num_trials_catch = int(length_ms_rough_catch / video_length_min_ms)
    length_ms_catch = num_trials_catch * video_length_min_ms
    video_lengths_f_catch = np.rint(
        np.array(num_trials_catch * [video_length_min_f])
    ).astype(int)

    # Get the remaining lengths of the dataset
    effective_dataset_length_ms = max_dataset_length_ms - length_ms_catch
    length_ms_bounce = effective_dataset_length_ms * bounce_straight_split
    length_ms_straight = effective_dataset_length_ms - length_ms_bounce

    # Estimate the number of videos and then sample lengths
    estimated_num_samples = (
        int(effective_dataset_length_ms / video_length_min_ms) + 1
    )
    estimated_video_lengths_ms = (
        np.random.exponential(exp_scale_ms, estimated_num_samples)
        + video_length_min_ms
    )
    estimated_video_lengths_f = np.rint(
        estimated_video_lengths_ms / duration
    ).astype(int)
    cumsum_video_lengths = np.cumsum(estimated_video_lengths_ms)

    # Grab a subset for the bounce trials
    length_idx_bounce = (
        np.where(cumsum_video_lengths < length_ms_bounce)[0][-1] + 1
    )
    length_ms_bounce = cumsum_video_lengths[length_idx_bounce]
    video_lengths_f_bounce = estimated_video_lengths_f[:length_idx_bounce]
    num_trials_bounce = len(video_lengths_f_bounce)

    # Get another subset for the straight trials
    length_idx_straight = (
        np.where(
            cumsum_video_lengths[length_idx_bounce + 1 :] - length_ms_bounce
            < length_ms_straight
        )[0][-1]
        + 1
    )
    length_ms_straight = (
        cumsum_video_lengths[length_idx_bounce + length_idx_straight + 1]
        - length_ms_bounce
    )
    video_lengths_f_straight = estimated_video_lengths_f[
        length_idx_bounce + 1 : length_idx_straight + length_idx_bounce + 1
    ]
    num_trials_straight = len(video_lengths_f_straight)

    # # Get the overall lengths and do a quick sanity check
    # total_videos = (
    #     num_trials_catch + num_trials_bounce + num_trials_straight
    # )

    dict_num_trials_type = {
        "catch": num_trials_catch,
        "straight": num_trials_straight,
        "bounce": num_trials_bounce,
    }
    dict_video_lengths_f_type = {
        "catch": video_lengths_f_catch,
        "straight": video_lengths_f_straight,
        "bounce": video_lengths_f_bounce,
    }
    assert sum(sum(v) for _, v in dict_video_lengths_f_type.items()) * duration < max_dataset_length_ms
    
    return dict_num_trials_type, dict_video_lengths_f_type


def compute_dataset_size_video_based(
        total_videos,
        p_catch_trials,
        video_length_min_f,
        video_length_min_ms,
        bounce_straight_split,
        fixed_video_length,
        exp_scale_ms,
        duration,
):
    # Split according to the desired number of trials
    num_trials_catch = int(total_videos * p_catch_trials)
    video_lengths_f_catch = np.rint(
        np.array(num_trials_catch * [video_length_min_f])
    ).astype(int)

    num_trials_noncatch = total_videos - num_trials_catch
    num_trials_bounce = int(num_trials_noncatch * bounce_straight_split)
    if fixed_video_length is None:
        video_lengths_f_bounce = np.rint(
            (
                np.random.exponential(exp_scale_ms, num_trials_bounce)
                + video_length_min_ms
            )
            / duration
        ).astype(int)
    else:
        video_lengths_f_bounce = np.array(
            [fixed_video_length] * num_trials_bounce
        ).astype(int)

    num_trials_straight = num_trials_noncatch - num_trials_bounce
    if fixed_video_length is None:
        video_lengths_f_straight = np.rint(
            (
                np.random.exponential(exp_scale_ms, num_trials_straight)
                + video_length_min_ms
            )
            / duration
        ).astype(int)
    else:
        video_lengths_f_straight = np.array(
            [fixed_video_length] * num_trials_straight
        ).astype(int)
        
    dict_num_trials_type = {
        "catch": num_trials_catch,
        "straight": num_trials_straight,
        "bounce": num_trials_bounce,
    }
    dict_video_lengths_f_type = {
        "catch": video_lengths_f_catch,
        "straight": video_lengths_f_straight,
        "bounce": video_lengths_f_bounce,
    }
    
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
        num_pos_endpoints,
        border_tolerance_outer,
        border_tolerance_inner,
        seed,
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
        "num_pos_endpoints": num_pos_endpoints,
        "pvc": pvc,
        "num_y_velocities": num_y_velocities,
        "bounce_offset": bounce_offset,
        "seed": seed,
    }
    
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
    dict_metadata["x_grayzone_linspace"] = x_grayzone_linspace = np.linspace(
        mask_start + (border_tolerance_inner + 1) * ball_radius,
        mask_end - (border_tolerance_inner + 1) * ball_radius,
        num_pos_endpoints,
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



def shorten_trials_and_update_meta(params_flattened, samples, targets, duration, variable_length=True):
    output_samples, output_targets, output_data = [], [], []    
    for idx_param, (param, sample, target) in enumerate(
        zip(
            params_flattened,
            samples,
            targets,
        )
    ):
        # Grab the relevant params
        position, velocity, color, pccnvc, pccovc, pvc, meta_trial = param
        length = meta_trial["length"]

        if variable_length:
            # Shorten the videos to the specified length
            output_samples.append(sample := sample[max_length - length :])
            output_targets.append(target := target[max_length - length :])
            
        # Update the metadata for the trial
        meta_trial.update(
            {
                "Final Color": default_ball_colors[np.argmax(target[-1, 2:])],
                "Final X Position": sample[-1, 0],
                "Final Y Position": sample[-1, 1],
                "Final X Velocity": -velocity[0],
                "Final Y Velocity": -velocity[1],
                "PCCNVC": pccnvc,
                "PCCOVC": pccovc,
                "PVC": pvc,
                "length_ms": length * duration,
            }
        )
        output_data.append(meta_trial)
        
    if variable_length:
        timesteps = np.array([target.shape[0] for target in output_targets])
        change_sums = np.array([target.sum(axis=0) for target in output_targets])[:, -4:]
        
    else:
        output_samples = samples
        output_targets = targets
        timesteps = np.array([output_targets.shape[1]] * output_targets.shape[0])
        change_sums = output_targets[:, :, -4:].sum(axis=1)

    return output_data, output_samples, output_targets, timesteps, change_sums    
        

def generate_data_df(
    row_data,
    dict_metadata,
    targets=None,
    num_blocks=None,
):
    df_trial_metadata = pd.DataFrame(row_data)
    
    # Change the column to ints
    for col in ["idx_time", "idx_position", "idx_velocity_y"]:
       df_trial_metadata[col] = (
           pd.to_numeric(df_trial_metadata[col], errors="coerce")
           .fillna(-1)
           .astype(int)
       )    

    if targets is not None:
        # Add in the last color entered
        df_trial_metadata["last_visible_color"] = color_entered = 1 + np.argmax(
            taskutils.last_visible_color(
                targets[:, :, :5],
                dict_metadata["ball_radius"],
                dict_metadata["mask_start"],
                dict_metadata["mask_end"],
            ),
            axis=1,
        )
        color_next = (color_entered % 3) + 1
        color_after_next = (color_next % 3) + 1

        # Add it to the df
        df_trial_metadata.loc[:, "color_entered"] = color_entered
        df_trial_metadata.loc[:, "color_next"] = color_next
        df_trial_metadata.loc[:, "color_after_next"] = color_after_next

    # Add a new column called final_color_response
    df_trial_metadata.loc[:, "correct_response"] = (
        df_trial_metadata.loc[:, "Final Color"]
        .map(default_color_to_idx_dict)
        .values
    )

    # Rename the column 'idx' to 'idx_trial'
    df_trial_metadata.rename(columns={'idx': 'idx_trial'}, inplace=True)
    df_trial_metadata.loc[:, "trial"] = df_trial_metadata["trial"].apply(
        lambda s: s.title()
    )
    
    # Rename it
    df_trial_metadata.index.name = 'Video ID'

    if num_blocks is not None:
        df_trial_metadata, dict_metadata = generate_blocks_from_data_df(
            df_trial_metadata,
            dict_metadata,
            num_blocks,
        )
        
    return df_trial_metadata, dict_metadata

    

def generate_blocks_from_data_df(
    df_trial_metadata,
    dict_metadata,
    num_blocks,
):
    idx_trials = list(df_trial_metadata.index)
    random.shuffle(idx_trials)
    num_rows = len(df_trial_metadata)
    meta_blocks = {}

    blocks = [[] for _ in range(num_blocks)]
    # Distribute elements across the lists in a round-robin manner
    for item, block in zip(idx_trials, itertools.cycle(blocks)):
        block.append(item)
    random.shuffle(blocks)

    for block_num, block in enumerate(blocks):
        random.shuffle(block)
        for video_num, video_idx in enumerate(block):
            df_trial_metadata.loc[video_idx, "Dataset Block"] = block_num + 1
            df_trial_metadata.loc[video_idx, "Dataset Block Video"] = video_num + 1

        df_block = df_trial_metadata[df_trial_metadata["Dataset Block"] == block_num + 1]
        length_block_ms = df_block["length"].sum() * dict_metadata["duration"]
        length_block_s = length_block_ms / 1000
        length_block_min = length_block_s / 60

        meta_blocks[block_num + 1] = {
            "num_trials": len(block),
            "length_block_ms": length_block_ms,
            "length_block_s": length_block_s,
            "length_block_min": length_block_min,            
        }

    # Change the column to ints
    for col in ["Dataset Block", "Dataset Block Video"]:
       df_trial_metadata[col] = (
           pd.to_numeric(df_trial_metadata[col], errors="coerce")
           .fillna(-1)
           .astype(int)
       )

    dict_metadata["num_blocks"] = num_blocks
    dict_metadata["blocks"] = meta_blocks
    dict_metadata["block_length_max_s"] = np.round(max([
        block["length_block_s"] for _, block in meta_blocks.items()
    ]))
    dict_metadata["block_length_min_s"] = np.round(min([
        block["length_block_s"] for _, block in meta_blocks.items()
    ]))

    return df_trial_metadata, dict_metadata


def add_effective_stats_to_df(
        df_data,
        timesteps,
        change_sums,
):
    df_data["PCCNVC_effective"] = (
        change_sums[:, -1] /
        (timesteps - change_sums[:, -4] - change_sums[:, -3])
    )
    df_data["PCCOVC_effective"] = change_sums[:, -2] / change_sums[:, -4]
    df_data["PVC_effective"] = change_sums[:, -3] / timesteps
    df_data["Bounces"] = change_sums[:, -4].astype(int)
    df_data["Random Bounces"] = change_sums[:, -3].astype(int)
    df_data["Color Change Bounce"] = change_sums[:, -2].astype(int)
    df_data["Color Change Random"] = change_sums[:, -1].astype(int)
    # Add observable changes

    # Overall condition descriptors
    hzs = np.sort(df_data["PCCNVC"].unique())
    conts = np.sort(df_data["PCCOVC"].unique())
    df_data["Hazard Rate"] = pd.Categorical(
        df_data["PCCNVC"].apply(
            lambda hz: (
                "Low" if np.isclose(hz, hzs[0]) else
                "High" if np.isclose(hz, hzs[1]) else
                "Unknown"
            )
        ),
        categories=["Low", "High"],
    )
    df_data["Contingency"] = pd.Categorical(
        df_data["PCCOVC"].apply(
            lambda cont: (
                "Low" if np.isclose(cont, conts[0]) else
                "Medium" if np.isclose(cont, conts[1]) else
                "High" if np.isclose(cont, conts[2]) else
                "Unknown"
            )
        ),
        categories=["Low", "Medium", "High"],
    )
    return df_data    


def save_video_dataset(
        dir_base,
        name_dataset,
        df_data,
        dict_metadata,
        output_samples,
        output_targets,
        duration=40,
        mode="original",
        multiplier=2,
        save_target=True,
        save_animation=True,
        display_animation=False,
        num_sequences=1,
        as_mp4=True,
        include_timestep=False,
        return_path=True,
        dryrun=False,
):
    dir_dataset = dir_base / name_dataset
    dir_all_videos = dir_dataset / "videos"
    msg = f"Saving dataset to {dir_dataset} (dir_dataset)"
    if dryrun:
        msg = f"Dryrun - {msg}"
    else:
        dir_all_videos.mkdir(parents=True)
    logger.info(msg)

    path_df_trial_meta = dir_dataset / "trial_meta.csv"
    msg = f"Saving trial metadata to to /dir_dataset/{path_df_trial_meta.stem}"
    if dryrun:
        msg = f"Dryrun - {msg}"
        logger.info(msg)
    else:
        logger.info(msg)
        df_data.to_csv(str(path_df_trial_meta))

    path_dataset_meta = dir_dataset / "dataset_meta.pkl"
    msg = f"Saving dataset metadata to to /dir_dataset/{path_dataset_meta.stem}"
    if dryrun:
        msg = f"Dryrun - {msg}"
        logger.info(msg)
    else:
        logger.info(msg)
        with open(str(path_dataset_meta), "wb") as handle:
            pickle.dump(dict_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path_videos = []
    sample_columns = ["x", "y", "r", "g", "b"]
    target_columns = sample_columns + ["vc_bounce", "vc_random", "cc_bounce", "cc_random"]        
    
    for idx_video in df_data.index:
        params = df_data.loc[idx_video]
        sample = output_samples[idx_video]
        target = output_targets[idx_video]
        color_change = target[:, -2:].any(axis=-1)
        idx_block = int(params["Dataset Block"])
        idx_block_video = int(params["Dataset Block Video"])
        target_color = params["Final Color"]
        
        # Create the df for the targets and color changes
        timestamps = np.arange(params["length"]) * duration        
        df_target = pd.DataFrame(target, index=timestamps, columns=target_columns)
        df_target.index.name = "Timestamp"
        df_sample = pd.DataFrame(sample, index=timestamps, columns=sample_columns)
        df_sample.index.name = "Timestamp"        
        df_color_change = pd.DataFrame(color_change, index=timestamps, columns=["Color Changed"],)
        df_color_change.index.name = "Timestamp"

        # Create the relevant paths
        dir_block = dir_all_videos / f"block_{idx_block}"
        dir_video = dir_block / f"video_{idx_block_video}"
        msg = f"Generating video files in /dir_dataset/videos/{dir_block.stem}/{dir_video.stem}"
        if dryrun:
            msg = f"  Dryrun - {msg}"
        elif not dir_video.exists():
            dir_video.mkdir(parents=True)
        logger.debug(msg)

        path_df_target = dir_video / f"video_{idx_block_video}_parameters.csv"
        path_df_sample = dir_video / f"video_{idx_block_video}_samples.csv"
        path_df_color_change = (
            dir_video / f"video_{idx_block_video}_color_change.csv"
        )
        if dryrun:
            logger.trace(
                f"    Dryrun - Saving target df as {path_df_target.stem}.csv"
            )
            logger.trace(
                f"    Dryrun - Saving sample df as {path_df_sample.stem}.csv"
            )
            logger.trace(
                f"    Dryrun - Saving color_change df as {path_df_color_change.stem}.csv"
            )
        else:
            df_target.to_csv(str(path_df_target))
            df_sample.to_csv(str(path_df_sample))
            df_color_change.to_csv(str(path_df_color_change))

        video_name = f"video_{idx_block_video}_{target_color}"
        if dryrun:
            path_video = dir_video / f"{video_name}.mp4"
            logger.trace(f"    Dryrun - Saving video in {path_video.stem}.mp4")
        else:
            path_video, _ = task.animate(
                target,
                path_dir=dir_video,
                name=video_name,                
                duration=duration,
                mode=mode,
                multiplier=multiplier,
                save_target=save_target,
                save_animation=save_animation,
                display_animation=display_animation,
                num_sequences=num_sequences,
                as_mp4=as_mp4,
                include_timestep=include_timestep,
                return_path=return_path,
            )
        path_videos.append(path_video)

    if return_path:
        return path_videos


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Human Bouncing Ball Data Generation Script"
    )

    parser.add_argument(
        "--size_frame",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Frame size as two integers (width, height)",
    )
    parser.add_argument(
        "--ball_radius", type=int, default=10, help="Radius of the ball"
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Time delta")
    parser.add_argument(
        "--num_blocks", type=int, default=20, help="Number of blocks"
    )
    parser.add_argument(
        "--video_length_min_s",
        type=float,
        default=6.5,
        help="Video length in seconds",
    )
    parser.add_argument("--duration", type=int, default=35, help="Duration")
    parser.add_argument(
        "--total_dataset_length",
        type=int,
        default=35,
        help="Total dataset length",
    )
    parser.add_argument(
        "--mask_center", type=float, default=0.5, help="Mask center"
    )
    parser.add_argument(
        "--mask_fraction", type=float, default=1 / 3, help="Mask fraction"
    )
    parser.add_argument(
        "--num_pos_endpoints",
        type=int,
        default=5,
        help="Number of positive endpoints",
    )
    parser.add_argument(
        "--velocity_lower",
        type=float,
        default=1 / 12.5,
        help="Lower velocity limit",
    )
    parser.add_argument(
        "--velocity_upper",
        type=float,
        default=1 / 7.5,
        help="Upper velocity limit",
    )
    parser.add_argument(
        "--num_y_velocities", type=int, default=2, help="Number of y velocities"
    )
    parser.add_argument(
        "--p_catch_trials",
        type=float,
        default=0.05,
        help="Probability of catch trials",
    )
    parser.add_argument("--dryrun", action="store_true", help="Enable dry run")

    parser.add_argument(
        "--pccnvc_lower", type=float, default=0.01, help="PCCNVC lower limit"
    )
    parser.add_argument(
        "--pccnvc_upper", type=float, default=0.045, help="PCCNVC upper limit"
    )
    parser.add_argument(
        "--pccovc_lower", type=float, default=0.1, help="PCCOVC lower limit"
    )
    parser.add_argument(
        "--pccovc_upper", type=float, default=0.9, help="PCCOVC upper limit"
    )
    parser.add_argument(
        "--num_pccnvc", type=int, default=2, help="Number of PCCNVC"
    )
    parser.add_argument(
        "--num_pccovc", type=int, default=3, help="Number of PCCOVC"
    )
    parser.add_argument("--pvc", type=float, default=0.0, help="PVC")
    parser.add_argument(
        "--border_tolerance_outer",
        type=float,
        default=2.0,
        help="Border tolerance outer",
    )
    parser.add_argument(
        "--border_tolerance_inner",
        type=float,
        default=1.0,
        help="Border tolerance inner",
    )
    parser.add_argument(
        "--bounce_straight_split",
        type=float,
        default=0.5,
        help="Bounce straight split",

    )
    parser.add_argument(
        "--sequence_length", type=int, default=None, help="Sequence length"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random Seed"
    )    
    parser.add_argument(
        "--target_future_timestep",
        type=int,
        default=0,
        help="Target future timestep",
    )
    parser.add_argument(
        "--mask_color",
        type=int,
        nargs=3,
        default=(127, 127, 127),
        help="Mask color as three integers (R, G, B)",
    )
    parser.add_argument(
        "--min_t_color_change",
        type=int,
        default=20,
        help="Minimum time for color change",
    )
    parser.add_argument(
        "--sample_mode", type=str, default="parameter_array", help="Sample mode"
    )
    parser.add_argument(
        "--target_mode", type=str, default="parameter_array", help="Target mode"
    )

    parser.add_argument("--no_change", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--return_change_mode", type=str, default="source", help="Change Mode"
    )
    
    parser.add_argument(
        "--sequence_mode", type=str, default="reverse", help="Sequence mode"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--display_animation", action="store_true", help="Display animation"
    )
    parser.add_argument("--mode", type=str, default="original", help="Mode")
    parser.add_argument("--multiplier", type=int, default=2, help="Multiplier")
    parser.add_argument(
        "--include_timestep",
        action="store_true",
        help="Include timestep in the output",
    )
    parser.add_argument(
        "--total_videos", type=int, default=None, help="Total number of videos"
    )
    parser.add_argument(
        "--dir_base",
        type=Path,
        default=index.dir_data / "hmdcpd",
        help="Base directory for output",
    )
    parser.add_argument(
        "--name_dataset",
        type=str,
        default="hbb_dataset_" + datetime.now().strftime("%y%m%d_%H%M%S"),
        help="Dataset name",
    )
    parser.add_argument(
        "--exp_scale",
        type=float,
        default=1.0,
        help="Exponential scaling factor",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Setup the logger
    logger = logutils.configure_logger(verbose=args.verbose, trace=args.debug)

    # Generate video parameters based on the arguments provided
    *params, dict_metadata = generate_video_parameters(
        size_frame=args.size_frame,
        ball_radius=args.ball_radius,
        dt=args.dt,
        video_length_min_s=args.video_length_min_s,
        duration=args.duration,
        total_dataset_length=args.total_dataset_length,
        mask_center=args.mask_center,
        mask_fraction=args.mask_fraction,
        num_pos_endpoints=args.num_pos_endpoints,
        velocity_lower=args.velocity_lower,
        velocity_upper=args.velocity_upper,
        num_y_velocities=args.num_y_velocities,
        p_catch_trials=args.p_catch_trials,
        num_pccnvc=args.num_pccnvc,
        num_pccovc=args.num_pccovc,
        border_tolerance_outer=args.border_tolerance_outer,
        border_tolerance_inner=args.border_tolerance_inner,
        bounce_straight_split=args.bounce_straight_split,
        total_videos=args.total_videos,
        exp_scale=args.exp_scale,
    )

    # Add some extra parameters to the metadata
    dict_metadata["min_t_color_change"] = args.min_t_color_change

    num_params = sum(len(param) for param in params)
    if args.total_videos is not None:
        assert num_params == args.total_videos

    if args.sequence_length is None:
        max_length = dict_metadata["video_length_max_f"]
    else:
        max_length = args.sequence_length
        
    params_flattened = list(itertools.chain.from_iterable(params))    
    positions, velocities, colors, pccnvcs, pccovcs, pvcs, meta_trials = (
        list(param) for param in zip(*params_flattened)
    )

    task = BouncingBallTask(
        size_frame=args.size_frame,
        sequence_length=max_length,
        ball_radius=args.ball_radius,
        target_future_timestep=args.target_future_timestep,
        dt=args.dt,
        batch_size=len(positions),
        sample_mode=args.sample_mode,
        target_mode=args.target_mode,
        mask_center=args.mask_center,
        mask_fraction=args.mask_fraction,
        mask_color=args.mask_color,
        sequence_mode=args.sequence_mode,
        return_change=not args.no_change,
        return_change_mode=args.return_change_mode,
        debug=True,
        probability_velocity_change=pvcs,
        probability_color_change_no_velocity_change=pccnvcs,
        probability_color_change_on_velocity_change=pccovcs,
        initial_position=positions,
        initial_velocity=velocities,
        initial_color=colors,
        min_t_color_change=args.min_t_color_change,
    )

    samples = task.samples
    targets = task.targets
    # Target change indices:
    #     targets[:, : -4] - Velocity Change Bounce - vcr
    #     targets[:, : -3] - Velocity Change Random - vcb
    #     targets[:, : -2] - Color Change bounce - ccb
    #     targets[:, : -1] - Color change random - ccr

    assert np.all(np.isclose(np.array(positions), targets[:, -1, :2]))
    assert np.all(np.isclose(np.stack(colors), targets[:, -1, 2:5]))

    output_data, output_samples, output_targets, timesteps, change_sums = shorten_trials_and_update_meta(
        params_flattened, samples, targets, args.duration, variable_length=True,
    )

    df_data, dict_metadata = generate_data_df(
        output_data,
        dict_metadata,
        targets,
        num_blocks=args.num_blocks,
    )
    df_data = add_effective_stats_to_df(df_data, timesteps, change_sums)
    htaskutils.print_block_stats(df_data, dict_metadata, args.duration)

    dir_base = Path(args.dir_base)
    dict_metadata["name"] = name_dataset = htaskutils.generate_dataset_name(
        args.name_dataset,
        seed=dict_metadata["seed"],
    )   

    path_videos = save_video_dataset(
        dir_base,
        name_dataset,
        df_data,
        dict_metadata,
        output_samples,
        output_targets,        
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
