from loguru import logger
import numpy as np
from bouncing_ball_task.utils import pyutils, htaskutils
from bouncing_ball_task.constants import DEFAULT_COLORS


def generate_catch_trials(
    num_trials,
    dict_meta,
    video_lengths_f,        
    print_stats=True,
    use_logger=True,
):
    border_tolerance_outer = dict_meta["border_tolerance_outer"]
    ball_radius = dict_meta["ball_radius"]
    mask_end = dict_meta["mask_end"]
    mask_start = dict_meta["mask_start"]
    size_x = dict_meta["size_x"]
    size_y = dict_meta["size_y"]
    num_pos_x_endpoints = dict_meta["num_pos_x_endpoints"]
    num_pos_y_endpoints = dict_meta["num_pos_y_endpoints"]
    final_velocity_x_magnitude = dict_meta["final_velocity_x_magnitude"]
    final_velocity_y_magnitude_linspace = dict_meta["final_velocity_y_magnitude_linspace"]
    pccnvc_linspace = dict_meta["pccnvc_linspace"]
    pccovc_linspace = dict_meta["pccovc_linspace"]
    pvc = dict_meta["pvc"]
    duration = dict_meta["duration"]
    catch_ncc_nvc_timesteps = dict_meta["catch_ncc_nvc_timesteps"]
    
    # x positions Non-Grayzone
    nongrayzone_left_x_range = (
        border_tolerance_outer * ball_radius,
        mask_start - border_tolerance_outer * ball_radius,
    )
    nongrayzone_right_x_range = (
        mask_end + border_tolerance_outer * ball_radius,
        size_x - border_tolerance_outer * ball_radius,
    )
    
    dict_meta_type = {"num_trials": num_trials}

    dict_meta_trials = {    
        "idx": list(range(num_trials)),
        "trial": ["catch"] * num_trials,
        "length": video_lengths_f.astype(int).tolist(),
    }

    # Keep track of possible catch x positions
    dict_meta_type["nongrayzone_left_x_range"] = nongrayzone_left_x_range
    dict_meta_type["nongrayzone_right_x_range"] = nongrayzone_right_x_range

    # Catch Trial Positions
    final_x_position = pyutils.alternating_ab_sequence(
        np.linspace(
            *nongrayzone_left_x_range,
            num_pos_x_endpoints,
            endpoint=True,
        ),
        np.linspace(
            *nongrayzone_right_x_range,
            num_pos_x_endpoints,
            endpoint=True,
        ),
        num_trials,
    )

    final_y_position = pyutils.repeat_sequence(
        np.linspace(
            border_tolerance_outer * ball_radius,
            size_y - border_tolerance_outer * ball_radius,
            num_pos_y_endpoints,
            endpoint=True,
        ),
        num_trials,
    )
    final_position = np.stack([final_x_position, final_y_position], axis=-1).tolist()
    
    # Catch trial velocities
    final_velocity = list(
        zip(
            [final_velocity_x_magnitude.item()] * num_trials,
            pyutils.repeat_sequence(
                final_velocity_y_magnitude_linspace,
                num_trials,
            ).tolist(),
        )
    )

    final_color, pccnvc, pccovc, dict_meta_type = htaskutils.compute_trial_color_and_stats(
        num_trials,
        dict_meta,
        dict_meta_type,
    )

    trials = htaskutils.group_trial_data(
        num_trials,
        final_position,
        final_velocity,
        final_color,
        pccnvc,
        pccovc,
        dict_meta["pvc"],
        dict_meta_trials=dict_meta_trials,
    )    

    dict_meta_type["overrides"] = {
        "warmup_t_no_rand_velocity_change": catch_ncc_nvc_timesteps,
        "warmup_t_no_rand_color_change": catch_ncc_nvc_timesteps,
    }    

    if print_stats:
        htaskutils.print_type_stats(trials, "catch", duration, use_logger=use_logger)

    return trials, dict_meta_type
