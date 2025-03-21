from typing import Optional
from collections.abc import Iterable
from loguru import logger
import numpy as np
from bouncing_ball_task.constants import DEFAULT_COLORS
from bouncing_ball_task.utils import pyutils, htaskutils
from bouncing_ball_task.human_bouncing_ball import defaults


def generate_nonwall_trials(
    num_trials,
    dict_meta,
    video_lengths_f,        
    print_stats=True,
    use_logger=True,
):
    ball_radius = dict_meta["ball_radius"]
    mask_end = dict_meta["mask_end"]
    mask_start = dict_meta["mask_start"]
    size_x = dict_meta["size_x"]
    size_y = dict_meta["size_y"]
    dt = dict_meta["dt"]    
    duration = dict_meta["duration"]

    num_pos_y_endpoints = dict_meta["num_pos_y_endpoints"]
    y_pos_multiplier = dict_meta["y_pos_multiplier"]
    
    num_pos_x_linspace_bounce = dict_meta["num_pos_x_linspace_bounce"]
    idx_linspace_bounce = dict_meta["idx_linspace_bounce"]
    bounce_timestep = dict_meta["bounce_timestep"]
    bounce_options = [[], [bounce_timestep,]]    
    
    final_velocity_x_magnitude = dict_meta["final_velocity_x_magnitude"]
    final_velocity_y_magnitude_linspace = dict_meta["final_velocity_y_magnitude_linspace"]
    
    pccnvc_linspace = dict_meta["pccnvc_linspace"]
    pccovc_linspace = dict_meta["pccovc_linspace"]
    pvc = dict_meta["pvc"]
    num_y_velocities = dict_meta["num_y_velocities"]

    border_tolerance_inner = dict_meta["border_tolerance_inner"] 
    repeat_factor = dict_meta["repeat_factor"]

    dict_meta_type = {"num_trials": num_trials}

    dict_meta_trials = {
        "idx": list(range(num_trials)),
        "length": video_lengths_f.astype(int).tolist(),
        "trial": ["nonwall"] * num_trials,
        "idx_time": [-1] * num_trials,
    }
    
    # Binary arrays for whether the ball exits the grayzone from the left or
    # right and if it is exiting away from the top or bottom
    sides_left_right = pyutils.repeat_sequence(
        np.array([0, 1] * repeat_factor),
        num_trials,
    ).astype(int)
    dict_meta_trials["side_left_right"] = sides_left_right.tolist()
    
    sides_top_bottom = pyutils.repeat_sequence(
        np.array([0, 1] * repeat_factor),
        num_trials,
    ).astype(int)
    dict_meta_trials["side_top_bottom"] = sides_top_bottom.tolist()

    # Binary array for whether the ball would have an x or y bounce
    bounce_x_y = pyutils.repeat_sequence(
        np.array([0, 1] * repeat_factor),
        num_trials,
    ).astype(int)
    dict_meta_trials["bounce_x_y"] = bounce_x_y.tolist()

    # Define the forced velocity changes 
    bounce_index_x = [bounce_options[i] for i in bounce_x_y]
    bounce_index_y = [bounce_options[int(i)] for i in np.logical_not(bounce_x_y)]
        
    # Compute the signs of the velocities as they exit the sides
    velocity_x_sign_after_bounce = -(2 * sides_left_right - 1)
    velocity_y_sign_after_bounce = -(2 * sides_top_bottom - 1)
    # dict_meta_trials["velocity_x_sign_after_bounce"] = velocity_x_sign_after_bounce.tolist()
    # dict_meta_trials["velocity_y_sign_after_bounce"] = velocity_y_sign_after_bounce.tolist()

    # Compute the starting velocity based on which bounce occurs
    velocity_x_sign = velocity_x_sign_after_bounce * (2 * bounce_x_y - 1)
    velocity_y_sign = velocity_y_sign_after_bounce * -(2 * bounce_x_y - 1)
    # dict_meta_trials["velocity_x_sign"] = velocity_x_sign.tolist()
    # dict_meta_trials["velocity_y_sign"] = velocity_y_sign.tolist()

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

    # Use the x position that the random bounce occurs, the distance that is
    # traversed between the start and bounce, and the starting sign of the
    # velocity to determine the position of the ball at the start
    # pos_x_bounce_trials = pos_x_bounce[np.logical_not(sides_left_right).astype(int)]
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
    
    # Precompute indices to sample the velocities from
    indices_velocity_y_magnitude = pyutils.repeat_sequence(
        np.array(list(range(num_y_velocities)) * num_pos_y_endpoints),
        num_trials,
    ).astype(int)
    dict_meta_trials["idx_velocity_y"] = indices_velocity_y_magnitude.tolist()

    # Keep track of velocities
    dict_meta_type["indices_velocity_y_magnitude_counts"] = np.unique(
        indices_velocity_y_magnitude,
        return_counts=True,
    )

    # Find the y bounce positions
    distance_y_to_bounce = bounce_timestep * final_velocity_y_magnitude_linspace * dt
    pos_y_bounce_linspace = dict_meta_type["pos_y_bounce_linspace"] = np.linspace(
        (y_pos_multiplier + 1) * ball_radius + distance_y_to_bounce,
        size_y - (y_pos_multiplier + 1) * ball_radius - distance_y_to_bounce,
        num_pos_y_endpoints,
        endpoint=True,
    )
    indices_pos_y_bounce = pyutils.repeat_sequence(
        np.array(list(range(num_pos_y_endpoints)) * repeat_factor),
        num_trials,
    ).astype(int)
    dict_meta_trials["idx_pos_y_bounce"] = indices_pos_y_bounce.tolist()

    # Set final y position according to the sampled bounce position, the sampled
    # velocity magnitudes, and the sampled velocity directions
    pos_y_bounce_trials = pos_y_bounce_linspace[
        indices_pos_y_bounce,
        indices_velocity_y_magnitude,
    ]
    final_y_positions = pos_y_bounce_trials - velocity_y_sign * distance_y_to_bounce[indices_velocity_y_magnitude]
    unique_y_positions, _ = dict_meta_type["final_y_positions_counts"] = np.unique(
        final_y_positions,
        return_counts=True,
        axis=0,
    )
    dict_meta_trials["idx_y_position"] = np.searchsorted(unique_y_positions, final_y_positions).tolist()  
    
    # Combine into final positions
    final_position = np.stack([final_x_positions, final_y_positions], axis=-1).tolist()

    # Define the final velocity positions and combine
    final_velocity_x = final_velocity_x_magnitude * velocity_x_sign
    final_velocity_y = final_velocity_y_magnitude_linspace[
        indices_velocity_y_magnitude
    ] * velocity_y_sign
    final_velocity = np.stack([final_velocity_x, final_velocity_y], axis=-1).tolist()
    
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

    # Put nonwall parameters together
    trials = list(
        zip(
            final_position,
            final_velocity,
            final_color,
            pccnvc,
            pccovc,
            [pvc,] * num_trials,
            bounce_index_x,
            bounce_index_y,
            [dict(zip(dict_meta_trials, values)) for values in zip(*dict_meta_trials.values())],
        )
    )

    if print_stats:
        htaskutils.print_type_stats(
            trials,
            "nonwall",
            duration,
            use_logger=use_logger,
        )
        
    return trials, dict_meta_type


def main_test(
    size_frame: Iterable[int] = (256, 256),
    ball_radius: int = 10,
    dt: float = 0.1,
    video_length_min_s: Optional[float] = 8.0, # seconds
    fixed_video_length: Optional[int] = None, # frames
    duration: Optional[int] = 45, # ms
    total_dataset_length: Optional[int] = 35,  # minutes
    mask_center: float = 0.5,
    mask_fraction: float = 1 / 3,
    num_pos_x_endpoints: int = 3,
    num_pos_y_endpoints: int = 4,
    y_pos_multiplier: int = 6,
    num_pos_bounce: int = 1,
    velocity_lower: float = 1 / 12.5,
    velocity_upper: float = 1 / 7.5,
    num_y_velocities: int = 2,
    pccnvc_lower: float = 0.00575,
    pccnvc_upper: float = 0.0575,
    pccovc_lower: float = 0.025,
    pccovc_upper: float = 0.975,
    num_pccnvc: int = 2,
    num_pccovc: int = 3,
    pvc: float = 0.0,
    border_tolerance_outer: float = 1.25,
    border_tolerance_inner: float = 1.0,
    trial_type_split: float = (0.05, -1, -1, -1),
    bounce_offset: float = 2 / 5,
    total_videos: Optional[int] = None,
    exp_scale: float = 1.0,  # seconds
    print_stats: bool = True,
    use_logger: bool = True,
    seed: Optional[int] = None,
):

    from bouncing_ball_task.human_bouncing_ball.catch import generate_catch_trials
    from bouncing_ball_task.human_bouncing_ball.straight import generate_straight_trials
    from bouncing_ball_task.human_bouncing_ball.bounce import generate_bounce_trials


    dict_trial_type_generation_funcs = {
        "catch": generate_catch_trials,
        "straight": generate_straight_trials,
        "bounce": generate_bounce_trials,
        "nonwall": generate_nonwall_trials,
    }

    
    # Set the seed
    seed = pyutils.set_global_seed(seed)

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
        num_pos_bounce,
        border_tolerance_outer,
        border_tolerance_inner,
        num_pos_x_linspace_bounce,
        idx_linspace_bounce,
        bounce_timestep,
        repeat_factor,        
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
    

if __name__ == "__main__":
    task_parameters = defaults.task_parameters
    human_dataset_parameters = defaults.human_dataset_parameters
    
    main_test(**human_dataset_parameters)
    # Convenience

