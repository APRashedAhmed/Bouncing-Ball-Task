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
from bouncing_ball_task.human_bouncing_ball import defaults
from bouncing_ball_task.human_bouncing_ball.catch import generate_catch_trials
from bouncing_ball_task.human_bouncing_ball.straight import generate_straight_trials
from bouncing_ball_task.human_bouncing_ball.bounce import generate_bounce_trials
from bouncing_ball_task.human_bouncing_ball.nonwall import generate_nonwall_trials


dict_trial_type_generation_funcs = {
    "catch": generate_catch_trials,
    "straight": generate_straight_trials,
    "bounce": generate_bounce_trials,
    "nonwall": generate_nonwall_trials,
}
trial_types = tuple(key for key, _ in dict_trial_type_generation_funcs.items())


def generate_video_dataset(
    human_dataset_parameters,
    task_parameters,    
    shuffle=True,
    validate=True,
):
    num_blocks = human_dataset_parameters["num_blocks"]
    total_videos = human_dataset_parameters["total_videos"]
    if num_blocks is not None and total_videos is not None:
        assert total_videos >= num_blocks
    
    *params, dict_metadata = generate_video_parameters(
        **human_dataset_parameters
    )
    # Assuming each parameter in params is a list of tuples, and you want to flatten and separate them
    params_flattened = list(itertools.chain.from_iterable(params))
    num_params = len(params_flattened)
    
    # Shuffle if asked
    if shuffle:
        random.shuffle(params_flattened)
    
    positions, velocities, colors, pccnvcs, pccovcs, pvcs, fxvc, fyvc, meta_trials = (
        list(param) for param in zip(*params_flattened)
    )

    # Grab relevant variables
    task_parameters = copy.deepcopy(task_parameters)
    task_parameters["initial_position"] = positions
    task_parameters["initial_velocity"] = velocities
    task_parameters["initial_color"] = colors
    task_parameters["probability_velocity_change"] = pvcs
    task_parameters["probability_color_change_no_velocity_change"] = pccnvcs
    task_parameters["probability_color_change_on_velocity_change"] = pccovcs
    task_parameters["forced_velocity_bounce_x"] = fxvc
    task_parameters["forced_velocity_bounce_y"] = fyvc
    task_parameters["batch_size"] = len(positions)
    task_parameters["sample_mode"] = defaults.sample_mode
    task_parameters["target_mode"] = defaults.target_mode
    task_parameters["return_change"] = defaults.return_change
    task_parameters["return_change_mode"] = defaults.return_change_mode
    task_parameters["sequence_mode"] = defaults.sequence_mode
    task_parameters["pccnvc_lower"] = None
    task_parameters["pccnvc_upper"] = None
    task_parameters["pccovc_lower"] = None
    task_parameters["pccovc_upper"] = None
    
    # Apply standardized params to task_parameters if requested
    if human_dataset_parameters["standard"]:
        task_parameters["target_future_timestep"] = defaults.target_future_timestep
        task_parameters["sequence_length"] = dict_metadata["video_length_max_f"]
        task_parameters["sample_velocity_discretely"] = defaults.sample_velocity_discretely
        task_parameters["initial_velocity_points_away_from_grayzone"] = defaults.initial_velocity_points_away_from_grayzone
        task_parameters["initial_timestep_is_changepoint"] = defaults.initial_timestep_is_changepoint
        task_parameters["min_t_color_change_after_bounce"] = defaults.min_t_color_change_after_bounce
        task_parameters["min_t_velocity_change_after_bounce"] = defaults.min_t_velocity_change_after_bounce
        task_parameters["min_t_color_change_after_random"] = defaults.min_t_color_change_after_random
        task_parameters["min_t_velocity_change_after_random"] = defaults.min_t_velocity_change_after_random
        task_parameters["warmup_t_no_rand_velocity_change"] = defaults.warmup_t_no_rand_velocity_change
        task_parameters["warmup_t_no_rand_color_change"] = defaults.warmup_t_no_rand_color_change
    else:
        logger.info("Not generating data using standardized parameters")
        
    task = BouncingBallTask(**task_parameters)

    samples = task.samples
    targets = task.targets
    dict_metadata["min_t_color_change_after_bounce"] = task.min_t_color_change_after_bounce
    dict_metadata["min_t_velocity_change_after_bounce"] = task.min_t_velocity_change_after_bounce
    dict_metadata["min_t_color_change_after_random"] = task.min_t_color_change_after_random
    dict_metadata["min_t_velocity_change_after_random"] = task.min_t_velocity_change_after_random
    
    # Update metadata
    output_data, output_samples, output_targets, timesteps, change_sums = shorten_trials_and_update_meta(
        params_flattened,
        samples,
        targets,
        human_dataset_parameters["duration"],
        variable_length=human_dataset_parameters["variable_length"],
    )
    
    df_data, dict_metadata = generate_data_df(
        output_data,
        dict_metadata,
        targets,
        num_blocks=human_dataset_parameters["num_blocks"],
    )

    df_data = add_effective_stats_to_df(df_data, timesteps, change_sums)

    if validate:
        assert np.all(np.isclose(np.array(positions), targets[:, -1, :2]))
        assert np.all(np.isclose(np.stack(colors), targets[:, -1, 2:5]))
    
    return task, output_samples, output_targets, df_data, dict_metadata


def generate_video_parameters(
    size_frame: Iterable[int] = defaults.size_frame,
    ball_radius: int = defaults.ball_radius,
    dt: float = defaults.dt,
    video_length_min_s: Optional[float] = defaults.video_length_min_s, # seconds
    fixed_video_length: Optional[int] = defaults.fixed_video_length, # frames
    duration: Optional[int] = defaults.duration, # ms
    total_dataset_length: Optional[int] = defaults.total_dataset_length,  # minutes
    mask_center: float = defaults.mask_center,
    mask_fraction: float = defaults.mask_fraction,
    num_pos_x_endpoints: int = defaults.num_pos_x_endpoints,
    num_pos_y_endpoints: int = defaults.num_pos_y_endpoints,
    y_pos_multiplier: int = defaults.y_pos_multiplier,
    velocity_lower: float = defaults.velocity_lower,
    velocity_upper: float = defaults.velocity_upper,
    num_y_velocities: int = defaults.num_y_velocities,
    pccnvc_lower: float = defaults.pccnvc_lower,
    pccnvc_upper: float = defaults.pccnvc_upper,
    pccovc_lower: float = defaults.pccovc_lower,
    pccovc_upper: float = defaults.pccovc_upper,
    num_pccnvc: int = defaults.num_pccnvc,
    num_pccovc: int = defaults.num_pccovc,
    pvc: float = defaults.pvc,
    border_tolerance_outer: float = defaults.border_tolerance_outer,
    border_tolerance_inner: float = defaults.border_tolerance_inner,
    trial_type_split: float = defaults.trial_type_split,
    bounce_offset: float = defaults.bounce_offset,
    total_videos: Optional[int] = defaults.total_videos,
    exp_scale: float = defaults.exp_scale,  # seconds
    print_stats: bool = defaults.print_stats,
    use_logger: bool = defaults.use_logger,
    num_pos_x_linspace_bounce: int = defaults.num_pos_x_linspace_bounce,
    idx_linspace_bounce: int = defaults.idx_linspace_bounce,
    bounce_timestep: int = defaults.bounce_timestep,
    repeat_factor: int = defaults.repeat_factor,
    seed: Optional[int] = defaults.seed,
    **kwargs,
):
    # Set the seed
    seed = pyutils.set_global_seed(seed)
    
    # Convenience
    dict_num_trials_type, dict_video_lengths_f_type = htaskutils.compute_dataset_size(
        exp_scale,
        fixed_video_length,
        video_length_min_s,
        duration,
        total_dataset_length,
        total_videos,
        trial_type_split,
    )

    dict_metadata = htaskutils.generate_initial_dict_metadata(
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
        **kwargs,
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
        # Add in the last color entered and its index
        last_color, last_idx = taskutils.last_visible_color(
            targets[:, :, :5],
            dict_metadata["ball_radius"],
            dict_metadata["mask_start"],
            dict_metadata["mask_end"],
            return_index=True,
        )
        df_trial_metadata["last_visible_color_idx"] = last_idx
        df_trial_metadata["last_visible_color"] = color_entered = 1 + np.argmax(
            last_color,
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
        position, velocity, color, pccnvc, pccovc, pvc, fxvc, fyvc, meta_trial = param
        length = meta_trial["length"]

        if variable_length:
            # Shorten the videos to the specified length
            output_samples.append(sample := sample[-length:])
            output_targets.append(target := target[-length:])
            
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
    """Adds 'effective' statistics of each individual sequence which is based on
    the actual number of changes that occured, including the unobservable ones.
    
    change_sums[:, 0] - Total velocity change Bounce - vcb
    change_sums[:, 1] - Total velocity change Random - vcr
    change_sums[:, 2] - Total color change bounce - ccb
    change_sums[:, 3] - Total color change random - ccr
    """
    # Add observable changes
    df_data["Bounces"] = vcb = change_sums[:, 0].astype(int)
    df_data["Random Bounces"] = vcr =change_sums[:, 1].astype(int)
    df_data["Color Change Bounce"] = ccb = change_sums[:, 2].astype(int)
    df_data["Color Change Random"] = ccr = change_sums[:, 3].astype(int)
    
    # Number of random changes / length of the sequence minus timesteps where a
    # velocity change occured - Random color changes are not sampled when there
    # is a velocity change
    df_data["PCCNVC_effective"] = ccr / (timesteps - vcb - vcr)
    # (
    #     change_sums[:, 3] /
    #     (timesteps - change_sums[:, 0] - change_sums[:, 1])
    # )

    # Number of bounce color changes / the number of velocity changes that
    # occured
    df_data["PCCOVC_effective"] = ccb / (vcb + vcr)
    # (
    #     change_sums[:, 2] /
    #     (change_sums[:, 0] + change_sums[:, 1])
    # )

    # Number of random velocity changes / length of sequence minus timesteps
    # where a wall bounce occured - random velocity changes are not sampled
    # when there is a wall bounce
    df_data["PVC_effective"] = vcr / (timesteps - vcb)
    # change_sums[:, 1] / (timesteps - change_sums[:, 0])
    
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
        task,
        duration=defaults.duration,
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
                animate_as_sample=True,
            )
        path_videos.append(path_video)

    if return_path:
        return path_videos


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    # Inferred args from the dictionaries
    parser = pyutils.add_dataclass_args(parser, defaults.HumanDatasetParameters)
    parser = pyutils.add_dataclass_args(parser, defaults.TaskParameters)
    
    # Manual additions
    parser.add_argument("--dir_base", type=Path, default=index.dir_data/"hmdcpd")
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
        key: getattr(args, key) for key in defaults.TaskParameters.keys
    }
    human_dataset_parameters = {
        key: getattr(args, key) for key in defaults.HumanDatasetParameters.keys
    }

    size_x, size_y = args.size_frame

    task, samples, targets, df_data, dict_metadata = generate_video_dataset(
        human_dataset_parameters,
        task_parameters,
        shuffle=False,
    )

    dict_metadata["name"] = name_dataset = htaskutils.generate_dataset_name(
        args.name_dataset,
        seed=dict_metadata["seed"],
    )

    path_videos = save_video_dataset(
        dir_base,
        name_dataset,
        df_data,
        dict_metadata,
        samples,
        targets,
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
