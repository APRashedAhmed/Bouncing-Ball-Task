from loguru import logger
import numpy as np
from bouncing_ball_task.utils import pyutils, htaskutils
from bouncing_ball_task.constants import DEFAULT_COLORS


def generate_cc_rvc_trials(
    num_trials,
    dict_meta,
    video_lengths_f, 
    print_stats=True,
    use_logger=True,
):
    dict_meta_type = {"num_trials": num_trials}
    size_y = dict_meta["size_y"]
    size_x = dict_meta["size_x"]
    dt = dict_meta["dt"]    
    ball_radius = dict_meta["ball_radius"]    
    mask_end = dict_meta["mask_end"]
    mask_start = dict_meta["mask_start"]    
    repeat_factor = dict_meta["repeat_factor"]
    num_pos_y_endpoints = dict_meta["num_pos_y_endpoints"]
    border_tolerance_inner = dict_meta["border_tolerance_inner"]
    final_velocity_x_magnitude = dict_meta["final_velocity_x_magnitude"]
    final_velocity_y_magnitude_linspace = dict_meta["final_velocity_y_magnitude_linspace"]
    ncc_nvc_timesteps = dict_meta["ncc_nvc_timesteps"] + 1
    timestep_change = dict_meta["timestep_change"] + 1
    timestep_from_wall = dict_meta["timestep_from_wall"]
    bounce_index_x = [timestep_change + 1,] * num_trials    
    color_change_index = [timestep_change + 1,] * num_trials
    
    dict_meta_trials = {
        "idx": list(range(num_trials)),
        "length": video_lengths_f.astype(int).tolist(),
        "trial": ["cc_rvc"] * num_trials,
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

    velocity_x_sign = 2 * sides_left_right - 1
    velocity_y_sign = -(2 * sides_top_bottom - 1)

    # Distance traveled in x per timestep
    dx = final_velocity_x_magnitude * dt
    # Find the distance traveled in x from the wall
    distance_x_from_bounce = timestep_change * dx
    # Find distance away from the wall
    distance_x_from_wall = timestep_from_wall * dx
    
    # Final x positions choices
    pos_x_sides = np.array([
        ball_radius + distance_x_from_bounce + distance_x_from_wall,
        size_x - ball_radius - distance_x_from_bounce - distance_x_from_wall
    ])
    final_x_positions = pos_x_sides[sides_left_right]

    # Keep track of position counts
    unique_x_positions, _ = dict_meta_type["final_x_positions"] = np.unique(
        final_x_positions,
        return_counts=True,
        axis=0,
    )
    dict_meta_trials["idx_x_position"] = np.searchsorted(unique_x_positions, final_x_positions).tolist()  

    # Find the distance traveled in y that the ball cant change
    final_velocity_y_magnitude = final_velocity_y_magnitude_linspace[indices_velocity_y_magnitude]
    distance_y_no_change = ncc_nvc_timesteps * final_velocity_y_magnitude * dt

    # Create the y linspace for the top and bottom
    final_y_positions_linspace = dict_meta_type["final_y_positions_linspace"] = np.array(
        [
            # Top
            np.linspace(
                ball_radius + 1,
                size_y - ball_radius - distance_y_no_change - 1,
                num_pos_y_endpoints,
                endpoint=True,
            ),
            
            # Bottom
            np.linspace(
                size_y - ball_radius - 1,
                ball_radius + distance_y_no_change + 1,
                num_pos_y_endpoints,
                endpoint=True,
            ),
        ]
    )

    # Create an idx for each endpoint position, starting with 0 at the point
    # closest to the top or bottom
    dict_meta_trials["idx_y_positions"] = idx_y_positions = pyutils.repeat_sequence(
        np.array(list(range(num_pos_y_endpoints)) * repeat_factor),
        num_trials,
    ).astype(int)

    # Use all the relevant indices to select the final pos and combine 
    final_y_positions = final_y_positions_linspace[
        sides_top_bottom,
        idx_y_positions,
        np.arange(num_trials)
    ]
    final_position = np.stack([final_x_positions, final_y_positions], axis=-1).tolist()

    # Create the final velocities
    final_velocity = np.stack(
        [
            final_velocity_x_magnitude * velocity_x_sign,
            final_velocity_y_magnitude * velocity_y_sign,
        ],
        axis=-1,
    ).tolist()

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
        bounce_index_x=bounce_index_x,
        color_change_index=color_change_index,
        dict_meta_trials=dict_meta_trials,
    )

    dict_meta_type["overrides"] = {
        "warmup_t_no_rand_velocity_change": ncc_nvc_timesteps,
        "warmup_t_no_rand_color_change": ncc_nvc_timesteps,
    }

    return trials, dict_meta_type
    
