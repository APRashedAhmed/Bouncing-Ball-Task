import copy
import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


from bouncing_ball_task import index
from bouncing_ball_task.constants import default_idx_to_color_dict
from bouncing_ball_task.bouncing_ball import BouncingBallTask
from bouncing_ball_task.utils import logutils, pyutils, taskutils, htaskutils
from bouncing_ball_task.model_bouncing_ball import defaults
from bouncing_ball_task.human_bouncing_ball import dataset as hds
from bouncing_ball_task.model_bouncing_ball.ncc_nvc import generate_ncc_nvc_trials
from bouncing_ball_task.model_bouncing_ball.cc_nvc import generate_cc_nvc_trials
from bouncing_ball_task.model_bouncing_ball.ncc_vc import generate_ncc_vc_trials
from bouncing_ball_task.model_bouncing_ball.cc_vc import generate_cc_vc_trials
from bouncing_ball_task.model_bouncing_ball.ncc_rvc import generate_ncc_rvc_trials
from bouncing_ball_task.model_bouncing_ball.cc_rvc import generate_cc_rvc_trials


dict_trial_type_generation_funcs = {
    "ncc_nvc": generate_ncc_nvc_trials,
    "cc_nvc": generate_cc_nvc_trials,
    "ncc_vc": generate_ncc_vc_trials,
    "cc_vc": generate_cc_vc_trials,
    "ncc_rvc": generate_ncc_rvc_trials,
    "cc_rvc": generate_cc_rvc_trials,
}
trial_types = tuple(key for key, _ in dict_trial_type_generation_funcs.items())


def generate_model_dataset_nongray(
    model_dataset_parameters,
    task_parameters,    
    shuffle=True,
    validate=True,
    dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
):
    assert (
        len(dict_trial_type_generation_funcs.items()) ==
        len(model_dataset_parameters["trial_type_split"])
    )

    dict_params, dict_metadata = hds.generate_video_parameters(
        **model_dataset_parameters,
        dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
    )
    trial_types = tuple(key for key, _ in dict_params.items())

    task_parameters = copy.deepcopy(task_parameters)
    
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
    
    task_parameters["sample_mode"] = defaults.sample_mode
    task_parameters["target_mode"] = defaults.target_mode
    task_parameters["return_change"] = defaults.return_change
    task_parameters["return_change_mode"] = defaults.return_change_mode
    task_parameters["sequence_mode"] = defaults.sequence_mode
    task_parameters["pccnvc_lower"] = None
    task_parameters["pccnvc_upper"] = None
    task_parameters["pccovc_lower"] = None
    task_parameters["pccovc_upper"] = None

    list_params_type = []
    list_samples_type = []
    list_targets_type = []

    for trial_type, params in dict_params.items():
        list_params_type += params
        
        positions, velocities, colors, pccnvcs, pccovcs, pvcs, fxvc, fyvc, fcc, meta_trials = (
            list(param) for param in zip(*params)
        )    
            
        # Set relevant variables
        task_parameters_type = copy.deepcopy(task_parameters)
        task_parameters_type["initial_position"] = positions
        task_parameters_type["initial_velocity"] = velocities
        task_parameters_type["initial_color"] = colors
        task_parameters_type["probability_velocity_change"] = pvcs
        task_parameters_type["probability_color_change_no_velocity_change"] = pccnvcs
        task_parameters_type["probability_color_change_on_velocity_change"] = pccovcs
        task_parameters_type["forced_velocity_bounce_x"] = fxvc
        task_parameters_type["forced_velocity_bounce_y"] = fyvc
        task_parameters_type["forced_color_changes"] = fcc
        task_parameters_type["batch_size"] = len(positions)

        # Apply overrides if they are defined
        if (overrides := dict_metadata[trial_type].get("overrides", None)):
            task_parameters_type.update(overrides)

        # Keep track of the underlying parameters
        dict_metadata[trial_type]["task_parameters"] = task_parameters_type

        # Create the underlying task instance
        task = BouncingBallTask(**task_parameters_type)

        if validate:
            assert np.all(np.isclose(np.array(positions), task.targets[:, -1, :2]))
            assert np.all(np.isclose(np.stack(colors), task.targets[:, -1, 2:5]))

        list_samples_type.append(task.samples)
        list_targets_type.append(task.targets)

    # Combine all the samples and targets to create one preset dataset
    samples = np.concatenate(list_samples_type)
    targets = np.concatenate(list_targets_type)
    task_parameters["sequence_mode"] = "preset"
    task_parameters["batch_size"] = len(samples)
    task = BouncingBallTask(**task_parameters, samples=samples, targets=targets)

    # Turn the samples and targets into the videos that will be used in the dataset
    output_data, output_samples, output_targets  = hds.shorten_trials_and_update_meta(    
        list_params_type,
        samples,
        targets,
        model_dataset_parameters["duration"],
        variable_length=model_dataset_parameters["variable_length"],
    )

    # Generate the complete metadata for the dataset
    df_data, dict_metadata = generate_dataset_metadata(
        output_data,
        dict_metadata,
        task_parameters_type,
        output_samples=output_samples,
        output_targets=output_targets,                
        num_blocks=model_dataset_parameters["num_blocks"],
    )

    return task, output_samples, output_targets, df_data, dict_metadata


def generate_dataset_metadata(
        output_data,
        dict_metadata,
        task_parameters_type,
        output_samples=None,
        output_targets=None,
        *args,
        **kwargs,
):
    df_data, dict_metadata = hds.generate_dataset_metadata(
        output_data,
        dict_metadata,
        task_parameters_type,
        output_samples=output_samples,
        output_targets=output_targets,
        *args,
        **kwargs,
    )

    color_final = df_data["Final Color"].values
    color_prev = df_data["color_after_next"].apply(
        lambda row: default_idx_to_color_dict[row]
    ).values
    color_final_prev = np.stack([color_final, color_prev], axis=-1)
    cc = df_data["trial"].str.startswith("cc").values.astype(int)
    df_data["Start Color"] = color_final_prev[np.arange(len(cc)), cc]
    
    return df_data, dict_metadata


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    parser = argparse.ArgumentParser()

    # Inferred args from the dictionaries
    parser = pyutils.add_dataclass_args(parser, defaults.TaskParameters)
    parser = pyutils.add_dataclass_args(parser, defaults.NongrayDatasetParameters)

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
    model_dataset_parameters = {
        key: getattr(args, key) for key in defaults.NongrayDatasetParameters.keys
    }

    size_x, size_y = args.size_frame

    task, samples, targets, df_data, dict_metadata = generate_model_dataset_nongray(
        model_dataset_parameters,
        task_parameters,
        dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
        shuffle=False,
    )

    dict_metadata["name"] = name_dataset = htaskutils.generate_dataset_name(
        args.name_dataset,
        seed=dict_metadata["seed"],
    )    

    path_videos = hds.save_video_dataset(
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
