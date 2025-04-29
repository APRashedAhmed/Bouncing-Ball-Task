from loguru import logger
import numpy as np
from bouncing_ball_task.utils import pyutils, htaskutils
from bouncing_ball_task.constants import DEFAULT_COLORS


def generate_bounce_trials(
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
    final_velocity_x_magnitude = dict_meta["final_velocity_x_magnitude"]
    final_velocity_y_magnitude_linspace = dict_meta["final_velocity_y_magnitude_linspace"]
    pccnvc_linspace = dict_meta["pccnvc_linspace"]
    pccovc_linspace = dict_meta["pccovc_linspace"]
    pvc = dict_meta["pvc"]
    duration = dict_meta["duration"]
    num_y_velocities = dict_meta["num_y_velocities"]
    dt = dict_meta["dt"]

    # num_pos_y_endpoints = dict_meta["num_pos_y_endpoints"]    
    num_pos_x_linspace_bounce = dict_meta["num_pos_x_linspace_bounce"]
    idx_linspace_bounce = dict_meta["idx_linspace_bounce"] - 1
    bounce_timestep = dict_meta["bounce_timestep"]   
    repeat_factor = dict_meta["repeat_factor"]
    border_tolerance_inner = dict_meta["border_tolerance_inner"] 

    dict_meta_type = {"num_trials": num_trials}
    
    dict_meta_trials = {
        "idx": list(range(num_trials)),
        "length": video_lengths_f.astype(int).tolist(),
        "trial": ["bounce"] * num_trials,
        "idx_time": [-1] * num_trials,
        "idx_position": [-1] * num_trials,
    }

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
    )

    # Compute the signs of the velocities as they exit the sides
    velocity_x_sign = 2 * sides_left_right - 1
    velocity_y_sign = 2 * sides_top_bottom - 1

    # Find the locations where the ball will randomly bounce
    pos_x_bounce_linspace = np.linspace(
        mask_start + (border_tolerance_inner + 1) * ball_radius,
        mask_end - (border_tolerance_inner + 1) * ball_radius,
        num_pos_x_linspace_bounce,
        endpoint=True,
    )
    pos_x_bounce = dict_meta_type["pos_x_bounce"] = np.vstack(
        [pos_x_bounce_linspace, pos_x_bounce_linspace[::-1]]
    )[:, idx_linspace_bounce]

    pos_x_bounce_trials = pos_x_bounce[sides_left_right]
    distance_x_to_bounce = bounce_timestep * final_velocity_x_magnitude * dt
    final_x_positions = pos_x_bounce_trials - velocity_x_sign * distance_x_to_bounce
    
    # Keep track of position counts
    unique_x_positions, _ = dict_meta_type["final_x_positions"] = np.unique(
        final_x_positions,
        return_counts=True,
        axis=0,
    )
    dict_meta_trials["idx_x_position"] = np.searchsorted(unique_x_positions, final_x_positions).tolist()  

    # Keep track of velocities
    dict_meta_type["indices_velocity_y_magnitude_counts"] = np.unique(
        indices_velocity_y_magnitude,
        return_counts=True,
    )

    # Find the y bounce positions
    distance_y_to_bounce = bounce_timestep * final_velocity_y_magnitude_linspace * dt
    final_y_pos_linspace = dict_meta_type["pos_y_bounce_linspace"] = np.array([
        ball_radius * 0.95 + distance_y_to_bounce,
        size_y - ball_radius * 0.95 - distance_y_to_bounce
    ]) 
    final_y_positions = final_y_pos_linspace[
        sides_top_bottom,
        indices_velocity_y_magnitude,
    ]
    unique_y_positions, _ = dict_meta_type["final_y_positions_counts"] = np.unique(
        final_y_positions,
        return_counts=True,
        axis=0,
    )
    
    # Combine into final positions    
    final_position = np.stack([final_x_positions, final_y_positions], axis=-1).tolist()

    # Define the final velocity positions and combine
    final_velocity_x = final_velocity_x_magnitude * velocity_x_sign
    final_velocity_y = final_velocity_y_magnitude_linspace[
        indices_velocity_y_magnitude
    ] * velocity_y_sign
    final_velocity = np.stack([final_velocity_x, final_velocity_y], axis=-1).tolist()

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
    
    if print_stats:
        htaskutils.print_type_stats(
            trials,
            "bounce",
            duration,
            use_logger=use_logger,
        )

    return trials, dict_meta_type
