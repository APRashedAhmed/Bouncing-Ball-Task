from loguru import logger
import numpy as np
from bouncing_ball_task.utils import pyutils, htaskutils
from bouncing_ball_task.constants import DEFAULT_COLORS


def generate_straight_trials(
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
    y_pos_multiplier = dict_meta["y_pos_multiplier"]    
    final_velocity_x_magnitude = dict_meta["final_velocity_x_magnitude"]
    final_velocity_y_magnitude_linspace = dict_meta["final_velocity_y_magnitude_linspace"]
    pccnvc_linspace = dict_meta["pccnvc_linspace"]
    pccovc_linspace = dict_meta["pccovc_linspace"]
    pvc = dict_meta["pvc"]
    duration = dict_meta["duration"]
    num_y_velocities = dict_meta["num_y_velocities"]
    diff = dict_meta["diff"]
    dt = dict_meta["dt"]
    x_grayzone_linspace_sides = dict_meta["x_grayzone_linspace_sides"]    
    dict_meta_type = {"num_trials": num_trials}

    dict_meta_trials = {
        "idx": list(range(num_trials)),
        "length": video_lengths_f.astype(int).tolist(),
        "trial": ["straight"] * num_trials,
    }    

    multipliers = np.arange(1, num_pos_x_endpoints + 1)
    time_x_diff = diff / (final_velocity_x_magnitude * dt)
    position_y_diff = final_velocity_y_magnitude_linspace * time_x_diff * dt

    indices_time_in_grayzone = dict_meta_trials["idx_time"] = pyutils.repeat_sequence(
        np.arange(num_pos_x_endpoints),
        num_trials,
        shuffle=False,
    ).astype(int)

    (
        sides_left_right,
        sides_top_bottom,
        indices_velocity_y_magnitude,
        dict_meta_trials,
        dict_meta_type,
    ) = htaskutils.compute_trial_idx_vals(
        num_trials,
        dict_meta,
        dict_meta_trials,
        dict_meta_type,
        repeat_factor=num_pos_x_endpoints,
    )    

    # Compute the signs of the velocities using the sides
    velocity_x_sign = 2 * sides_left_right - 1
    velocity_y_sign = (
        2 * np.logical_not(sides_top_bottom) - 1
    )

    # Keep track of velocities
    dict_meta_type["indices_velocity_y_magnitude_counts"] = np.unique(
        indices_velocity_y_magnitude,
        return_counts=True,
    )

    # Straight y positions
    y_distance_traversed = dict_meta_type["y_distance_traversed"] = (
        position_y_diff[:, np.newaxis] * multipliers
    )

    final_y_positions_left = np.stack(
        [
            # Top
            np.linspace(
                np.ones_like(y_distance_traversed)
                * 2
                * ball_radius,
                size_y - y_distance_traversed - y_pos_multiplier * ball_radius,
                num_pos_y_endpoints,
                endpoint=True,
                axis=-1,
            ),
            # Bottom
            np.linspace(
                size_y - 2 * ball_radius,
                y_distance_traversed + y_pos_multiplier * ball_radius,
                num_pos_y_endpoints,
                endpoint=True,
                axis=-1,
            ),
        ]
    )

    # This is shape [2 x 2 x num_vel x num_pos_x_endpoints x 2*num_pos_x_endpoints]
    # [left/right, top/bottom, each vel, num x positions, num y pos per x pos]
    final_y_positions = dict_meta_type[
        "final_y_positions"
    ] = np.stack(
        [
            final_y_positions_left,
            final_y_positions_left[:, :, ::-1],
        ]
    )
    
    final_velocity = np.stack(
        [
            final_velocity_x_magnitude * velocity_x_sign,            
            final_velocity_y_magnitude_linspace[
                indices_velocity_y_magnitude
            ] * velocity_y_sign,
        ],
        axis=-1,
    ).tolist()
    final_x_positions = x_grayzone_linspace_sides[
        sides_left_right,
        indices_time_in_grayzone
    ]
    final_y_positions = np.array([
        np.random.choice(trial_choice) 
        for trial_choice in final_y_positions[
                sides_left_right,
                sides_top_bottom,
                indices_velocity_y_magnitude,
                indices_time_in_grayzone,
        ]
    ])
    final_position = np.stack([final_x_positions, final_y_positions], axis=-1).tolist()

    final_color, pccnvc, pccovc, dict_meta_type = htaskutils.compute_trial_color_and_stats(
        num_trials,
        dict_meta,
        dict_meta_type,
    )    
    
    # Keep track of position counts
    dict_meta_type["x_grayzone_position_counts"] = np.unique(
        [x for x in zip(*final_position)][0],
        return_counts=True,
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

    if print_stats:
        htaskutils.print_type_stats(
            trials,
            "straight",
            duration,
            use_logger=use_logger,
        )

    return trials, dict_meta_type
