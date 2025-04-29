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
from bouncing_ball_task.human_bouncing_ball import dataset as hds
from bouncing_ball_task.model_bouncing_ball import defaults
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
    task, output_samples, output_targets, df_data, dict_metadata = hds.generate_video_dataset(
        model_dataset_parameters,
        task_parameters, 
        dict_trial_type_generation_funcs,
        shuffle=shuffle,
        validate=validate,
        defaults=defaults,
    )

    color_final = df_data["Final Color"].values
    color_prev = df_data["color_after_next"].apply(
        lambda row: default_idx_to_color_dict[row]
    ).values
    color_final_prev = np.stack([color_final, color_prev], axis=-1)
    cc = df_data["trial"].str.startswith("cc").values.astype(int)
    df_data["Start Color"] = color_final_prev[np.arange(len(cc)), cc]

    return task, output_samples, output_targets, df_data, dict_metadata
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    parser = argparse.ArgumentParser()

    # Inferred args from the dictionaries
    parser = pyutils.add_dataclass_args(parser, defaults.TaskParameters)
    parser = pyutils.add_dataclass_args(parser, defaults.NongrayDatasetParameters)

    # Manual additions
    parser.add_argument("--dir_base", type=Path, default=index.dir_repo/"data/hmdcpd")
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
