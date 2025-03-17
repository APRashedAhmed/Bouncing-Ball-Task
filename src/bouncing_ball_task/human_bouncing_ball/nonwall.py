from typing import Optional
from collections.abc import Iterable
from loguru import logger
import numpy as np
from bouncing_ball_task.constants import DEFAULT_COLORS
from bouncing_ball_task.utils import pyutils, htaskutils


def generate_nonwall_trials(
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
    
    num_pos_bounce = dict_meta["num_pos_bounce"]
    
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
    
    dict_meta = {"num_trials": num_trials}

    bounce_dim = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_bounce),
        num_trials,
    ).astype(int)


    # Binary arrays for whether the ball enters the grayzone from the left or
    # right and if it is going towards the top or bottom
    sides_left_right = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_bounce),
        num_trials,
    ).astype(int)
    sides_top_bottom = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_bounce),
        num_trials,
    ).astype(int)

    # Compute the signs of the velocities using the sides
    velocity_x_sign = 2 * sides_left_right - 1
    velocity_y_sign = (
        2 * np.logical_not(sides_top_bottom) - 1
    )

    # Precompute indices to sample the velocities from
    indices_velocity_y_magnitude = pyutils.repeat_sequence(
        np.array(list(range(num_y_velocities)) * num_pos_bounce),
        num_trials,
    ).astype(int)

    # Keep track of velocities
    dict_meta["indices_velocity_y_magnitude_counts"] = np.unique(
        indices_velocity_y_magnitude,
        return_counts=True,
    )

    


    

    multipliers = np.arange(1, num_pos_endpoints + 1)
    time_x_diff = diff / (final_velocity_x_magnitude * dt)
    position_y_diff = final_velocity_y_magnitude_linspace * time_x_diff * dt

    indices_time_in_grayzone = pyutils.repeat_sequence(
        np.arange(num_pos_endpoints),
        num_trials,
        shuffle=False,
    ).astype(int)

    # Nonwall y positions
    y_distance_traversed = dict_meta["y_distance_traversed"] = (
        position_y_diff[:, np.newaxis] * multipliers
    )

    final_y_positions_left = np.stack(
        [
            # Top
            np.linspace(
                np.ones_like(y_distance_traversed)
                * 2
                * ball_radius,
                size_y - y_distance_traversed - 4 * ball_radius,
                2 * num_pos_endpoints,
                endpoint=True,
                axis=-1,
            ),
            # Bottom
            np.linspace(
                size_y - 2 * ball_radius,
                y_distance_traversed + 4 * ball_radius,
                2 * num_pos_endpoints,
                endpoint=True,
                axis=-1,
            ),
        ]
    )

    # This is shape [2 x 2 x num_vel x num_pos_endpoints x 2*num_pos_endpoints]
    # [left/right, top/bottom, each vel, num x positions, num y pos per x pos]
    final_y_positions = dict_meta[
        "final_y_positions"
    ] = np.stack(
        [
            final_y_positions_left,
            final_y_positions_left[:, :, ::-1],
        ]
    )

    # Precompute colors
    final_color = pyutils.repeat_sequence(
        np.array(DEFAULT_COLORS),
        num_trials,
        shuffle=False,
        roll=True,
        shift=1,
    ).tolist()

    # Keep track of color counts
    dict_meta["final_color_counts"] = np.unique(
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
    dict_meta["pccnvc_counts"] = np.unique(
        pccnvc,
        return_counts=True,
    )
    dict_meta["pccovc_counts"] = np.unique(
        pccovc,
        return_counts=True,
    )
    dict_meta["pccnvc_pccovc_counts"] = np.unique(
        [x for x in zip(*(pccnvc, pccovc))],
        return_counts=True,
        axis=0,
    )

    final_position = []
    final_velocity = []
    meta = []

    for idx in range(num_trials):
        # Get an index of time
        idx_time = indices_time_in_grayzone[idx]

        # Choose the sides to enter the grayzone
        side_left_right = sides_left_right[idx]
        side_top_bottom = sides_top_bottom[idx]

        # Get the y velocity index for this trial
        idx_velocity_y = indices_velocity_y_magnitude[idx]

        # Final velocities
        final_velocity_x = (
            final_velocity_x_magnitude * velocity_x_sign[idx]
        ).item()
        final_velocity_y = (
            final_velocity_y_magnitude_linspace[idx_velocity_y]
            * velocity_y_sign[idx]
        ).item()
        final_velocity.append((final_velocity_x, final_velocity_y))

        # Grab the final positions
        final_position_x = x_grayzone_linspace_sides[
            side_left_right, idx_time
        ].item()
        final_position_y = np.random.choice(
            final_y_positions[
                side_left_right,
                side_top_bottom,
                idx_velocity_y,
                idx_time,
            ]
        ).item()
        final_position.append((final_position_x, final_position_y))

        meta.append(
            {
                "idx": idx,
                "trial": "nonwall",
                "idx_time": idx_time,
                "side_left_right": side_left_right,
                "side_top_bottom": side_top_bottom,
                "idx_velocity_y": idx_velocity_y,
                "idx_x_position": -1,
                "idx_y_position": -1,
                "length": video_lengths_f[idx],
            }
        )

    # Keep track of position counts
    dict_meta["x_grayzone_position_counts"] = np.unique(
        [x for x in zip(*final_position)][0],
        return_counts=True,
    )

    # Put nonwall parameters together
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
            meta,
        )
    )

    if print_stats:
        htaskutils.print_type_stats(
            trials,
            "nonwall",
            duration,
            use_logger=use_logger,
        )

    return trials, dict_meta


if __name__ == "__main__":    
    size_frame: Iterable[int] = (256, 256)
    ball_radius: int = 10
    dt: float = 0.1
    video_length_min_s: Optional[float] = 8.0 # seconds
    fixed_video_length: Optional[int] = None # frames
    duration: Optional[int] = 45 # ms
    total_dataset_length: Optional[int] = 35  # minutes
    mask_center: float = 0.5
    mask_fraction: float = 1 / 3
    num_pos_endpoints_straight: int = 3
    num_pos_endpoints_bounce: int = 1
    velocity_lower: float = 1 / 12.5
    velocity_upper: float = 1 / 7.5
    num_y_velocities: int = 2
    pccnvc_lower: float = 0.00575
    pccnvc_upper: float = 0.0575
    pccovc_lower: float = 0.025
    pccovc_upper: float = 0.975
    num_pccnvc: int = 2
    num_pccovc: int = 3
    pvc: float = 0.0
    border_tolerance_outer: float = 1.25
    border_tolerance_inner: float = 1.0
    trial_type_split: float = (0.05, -1, -1, -1)
    bounce_offset: float = 2 / 5
    total_videos: Optional[int] = None
    exp_scale: float = 1.0  # seconds
    print_stats: bool = True
    use_logger: bool = True
    seed = pyutils.set_global_seed(None)

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
        num_pos_endpoints_straight,
        num_pos_endpoints_bounce,
        border_tolerance_outer,
        border_tolerance_inner,
        seed,
    )
    
