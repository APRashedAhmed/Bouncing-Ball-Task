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
    num_pos_endpoints = dict_meta["num_pos_endpoints"]
    final_velocity_x_magnitude = dict_meta["final_velocity_x_magnitude"]
    final_velocity_y_magnitude_linspace = dict_meta["final_velocity_y_magnitude_linspace"]
    pccnvc_linspace = dict_meta["pccnvc_linspace"]
    pccovc_linspace = dict_meta["pccovc_linspace"]
    pvc = dict_meta["pvc"]
    duration = dict_meta["duration"]
    
    # x positions Non-Grayzone
    nongrayzone_left_x_range = (
        border_tolerance_outer * ball_radius,
        mask_start - border_tolerance_outer * ball_radius,
    )
    nongrayzone_right_x_range = (
        mask_end + border_tolerance_outer * ball_radius,
        size_x - border_tolerance_outer * ball_radius,
    )
    
    dict_meta = {"num_trials": num_trials}

    # Keep track of possible catch x positions
    dict_meta["nongrayzone_left_x_range"] = nongrayzone_left_x_range
    dict_meta["nongrayzone_right_x_range"] = nongrayzone_right_x_range

    # Catch Trial Positions
    final_x_position = pyutils.alternating_ab_sequence(
        np.linspace(
            *nongrayzone_left_x_range,
            num_pos_endpoints,
            endpoint=True,
        ),
        np.linspace(
            *nongrayzone_right_x_range,
            num_pos_endpoints,
            endpoint=True,
        ),
        num_trials,
    ).tolist()

    final_y_position = pyutils.repeat_sequence(
        np.linspace(
            border_tolerance_outer * ball_radius,
            size_y - border_tolerance_outer * ball_radius,
            2 * num_pos_endpoints,
            endpoint=True,
        ),
        num_trials,
    ).tolist()
    final_position = zip(
        final_x_position, final_y_position
    )

    # Catch trial velocities
    final_velocity = zip(
        [final_velocity_x_magnitude.item()] * num_trials,
        pyutils.repeat_sequence(
            final_velocity_y_magnitude_linspace,
            num_trials,
        ).tolist(),
    )

    # Catch colors
    final_color = pyutils.repeat_sequence(
        np.array(DEFAULT_COLORS),
        num_trials,
    ).tolist()

    # Catch probabilities
    pccnvc = pyutils.repeat_sequence(
        pccnvc_linspace,
        num_trials,
    ).tolist()
    pccovc = pyutils.repeat_sequence(
        pccovc_linspace,
        num_trials,
    ).tolist()

    # Put parameters together
    trials = list(
        zip(
            final_position,
            final_velocity,
            final_color,
            pccnvc,
            pccovc,
            [
                pvc,
            ]
            * num_trials,
            [pvc,] * num_trials,
            [[],] * num_trials,
            [[],] * num_trials,
            [
                {
                    "idx": idx,
                    "trial": "catch",
                    "idx_time": -1,
                    "side_left_right": -1,
                    "side_top_bottom": -1,
                    "idx_velocity_y": -1,
                    "idx_x_position": -1,
                    "idx_y_position": -1,
                    "length": video_lengths_f[idx],
                }
                for idx in range(num_trials)
            ],
        )
    )

    if print_stats:
        htaskutils.print_type_stats(trials, "catch", duration, use_logger=use_logger)

    return trials, dict_meta
