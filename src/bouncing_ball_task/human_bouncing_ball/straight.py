from loguru import logger
import numpy as np
from bouncing_ball_task.utils import pyutils, htaskutils
from bouncing_ball_task.constants import DEFAULT_COLORS


def generate_straight_trials(
    num_trials_straight,
    dict_meta,
    video_lengths_f_straight,        
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
    num_y_velocities = dict_meta["num_y_velocities"]
    diff = dict_meta["diff"]
    dt = dict_meta["dt"]
    x_grayzone_linspace_sides = dict_meta["x_grayzone_linspace_sides"]
    
    dict_meta_straight = {"num_trials": num_trials_straight}

    multipliers = np.arange(1, num_pos_endpoints + 1)
    time_x_diff = diff / (final_velocity_x_magnitude * dt)
    position_y_diff = final_velocity_y_magnitude_linspace * time_x_diff * dt

    indices_time_in_grayzone_straight = pyutils.repeat_sequence(
        np.arange(num_pos_endpoints),
        num_trials_straight,
        shuffle=False,
    ).astype(int)

    # Binary arrays for whether the ball enters the grayzone from the left or
    # right and if it is going towards the top or bottom
    sides_left_right_straight = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_endpoints),
        num_trials_straight,
    ).astype(int)
    sides_top_bottom_straight = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_endpoints),
        num_trials_straight,
    ).astype(int)

    # Compute the signs of the velocities using the sides
    velocity_x_sign_straight = 2 * sides_left_right_straight - 1
    velocity_y_sign_straight = (
        2 * np.logical_not(sides_top_bottom_straight) - 1
    )

    # Precompute indices to sample the velocities from
    indices_velocity_y_magnitude = pyutils.repeat_sequence(
        np.array(list(range(num_y_velocities)) * num_pos_endpoints),
        num_trials_straight,
    ).astype(int)

    # Keep track of velocities
    dict_meta_straight["indices_velocity_y_magnitude_counts"] = np.unique(
        indices_velocity_y_magnitude,
        return_counts=True,
    )

    # Straight y positions
    y_distance_traversed_straight = dict_meta_straight["y_distance_traversed_straight"] = (
        position_y_diff[:, np.newaxis] * multipliers
    )

    final_y_positions_straight_left = np.stack(
        [
            # Top
            np.linspace(
                np.ones_like(y_distance_traversed_straight)
                * 2
                * ball_radius,
                size_y - y_distance_traversed_straight - 4 * ball_radius,
                2 * num_pos_endpoints,
                endpoint=True,
                axis=-1,
            ),
            # Bottom
            np.linspace(
                size_y - 2 * ball_radius,
                y_distance_traversed_straight + 4 * ball_radius,
                2 * num_pos_endpoints,
                endpoint=True,
                axis=-1,
            ),
        ]
    )

    # This is shape [2 x 2 x num_vel x num_pos_endpoints x 2*num_pos_endpoints]
    # [left/right, top/bottom, each vel, num x positions, num y pos per x pos]
    final_y_positions_straight = dict_meta_straight[
        "final_y_positions"
    ] = np.stack(
        [
            final_y_positions_straight_left,
            final_y_positions_straight_left[:, :, ::-1],
        ]
    )

    # Precompute colors
    final_color_straight = pyutils.repeat_sequence(
        np.array(DEFAULT_COLORS),
        num_trials_straight,
        shuffle=False,
        roll=True,
        shift=1,
    ).tolist()

    # Keep track of color counts
    dict_meta_straight["final_color_counts"] = np.unique(
        final_color_straight,
        return_counts=True,
        axis=0,
    )

    # Precompute the statistics
    pccnvc_straight = pyutils.repeat_sequence(
        pccnvc_linspace,
        # np.tile(pccnvc_linspace, num_pccovc),
        num_trials_straight,
        shuffle=False,
        roll=True,
        # roll=False if num_pccovc % num_pccnvc else True,
    ).tolist()
    pccovc_straight = pyutils.repeat_sequence(
        pccovc_linspace,
        # np.tile(pccovc_linspace, num_pccnvc),
        num_trials_straight,
        shuffle=False,
    ).tolist()

    # Keep track of pcc counts
    dict_meta_straight["pccnvc_counts"] = np.unique(
        pccnvc_straight,
        return_counts=True,
    )
    dict_meta_straight["pccovc_counts"] = np.unique(
        pccovc_straight,
        return_counts=True,
    )
    dict_meta_straight["pccnvc_pccovc_counts"] = np.unique(
        [x for x in zip(*(pccnvc_straight, pccovc_straight))],
        return_counts=True,
        axis=0,
    )

    final_position_straight = []
    final_velocity_straight = []
    meta_straight = []

    for idx in range(num_trials_straight):
        # Get an index of time
        idx_time = indices_time_in_grayzone_straight[idx]

        # Choose the sides to enter the grayzone
        side_left_right = sides_left_right_straight[idx]
        side_top_bottom = sides_top_bottom_straight[idx]

        # Get the y velocity index for this trial
        idx_velocity_y = indices_velocity_y_magnitude[idx]

        # Final velocities
        final_velocity_x = (
            final_velocity_x_magnitude * velocity_x_sign_straight[idx]
        ).item()
        final_velocity_y = (
            final_velocity_y_magnitude_linspace[idx_velocity_y]
            * velocity_y_sign_straight[idx]
        ).item()
        final_velocity_straight.append((final_velocity_x, final_velocity_y))

        # Grab the final positions
        final_position_x = x_grayzone_linspace_sides[
            side_left_right, idx_time
        ].item()
        final_position_y = np.random.choice(
            final_y_positions_straight[
                side_left_right,
                side_top_bottom,
                idx_velocity_y,
                idx_time,
            ]
        ).item()
        final_position_straight.append((final_position_x, final_position_y))

        meta_straight.append(
            {
                "idx": idx,
                "trial": "straight",
                "idx_time": idx_time,
                "side_left_right": side_left_right,
                "side_top_bottom": side_top_bottom,
                "idx_velocity_y": idx_velocity_y,
                "idx_x_position": -1,
                "idx_y_position": -1,
                "length": video_lengths_f_straight[idx],
            }
        )

    # Keep track of position counts
    dict_meta_straight["x_grayzone_position_counts"] = np.unique(
        [x for x in zip(*final_position_straight)][0],
        return_counts=True,
    )

    # Put straight parameters together
    trials_straight = list(
        zip(
            final_position_straight,
            final_velocity_straight,
            final_color_straight,
            pccnvc_straight,
            pccovc_straight,
            [
                pvc,
            ]
            * num_trials_straight,
            meta_straight,
        )
    )

    if print_stats:
        htaskutils.print_type_stats(
            trials_straight,
            "straight",
            duration,
            use_logger=use_logger,
        )

    return trials_straight, dict_meta_straight
