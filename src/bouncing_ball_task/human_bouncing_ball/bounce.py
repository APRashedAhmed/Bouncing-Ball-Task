from loguru import logger
import numpy as np
from bouncing_ball_task.utils import pyutils, htaskutils
from bouncing_ball_task.constants import DEFAULT_COLORS


def generate_bounce_trials(
    num_trials_bounce,
    dict_meta,
    video_lengths_f_bounce,        
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
    x_grayzone_linspace = dict_meta["x_grayzone_linspace"]
    bounce_offset = dict_meta["bounce_offset"]
    
    dict_meta_bounce = {"num_trials": num_trials_bounce}

    # Determine bounce positions
    left_bounce_x_positions = x_grayzone_linspace - diff * bounce_offset
    assert mask_start < left_bounce_x_positions.min()
    right_bounce_x_positions = x_grayzone_linspace + diff * bounce_offset
    assert right_bounce_x_positions.max() < mask_end

    # How many final positions are there
    final_position_index_bounce = dict_meta_bounce[
        "final_position_index_bounce"
    ] = np.arange(sum(range(1, num_pos_endpoints + 1)))

    # Create indices to x and y positions from these
    # The first x position when entering the grayzone only has one coord,
    # the second has two, until the `num_pos_endpoint`th pos which has
    # `num_pos_endpoint` unique coords
    final_x_index_bounce = dict_meta_bounce[
        "final_x_index_bounce"
    ] = np.repeat(
        np.arange(num_pos_endpoints),
        np.arange(1, num_pos_endpoints + 1),
    ).astype(int)
    # This is reversed for y - there are `num_pos_endpoints` unqiue coords
    # for the first y coordinate (defined as being the point closest to the
    # top or bottom), and that decreases by one until the last one
    final_y_index_bounce = dict_meta_bounce[
        "final_y_index_bounce"
    ] = [j for i in range(num_pos_endpoints + 1) for j in range(i)]
    

    # Create all the indices for the bounce trials
    indices_final_position_index_bounce = pyutils.repeat_sequence_imbalanced(
        final_position_index_bounce,
        final_x_index_bounce,
        num_trials_bounce,
        roll=True,
    )

    # Keep track of coordinate counts
    dict_meta_bounce[
        "indices_final_position_index_bounce_counts"
    ] = np.unique(
        indices_final_position_index_bounce,
        return_counts=True,
    )

    # How long does it take to get between each x position
    time_steps_between_x = diff / (final_velocity_x_magnitude * dt)
    # How long from a bounce to the first end position
    time_steps_bounce = time_steps_between_x * bounce_offset
    # How long from a bounce to all possible endpoints
    time_steps_bounce_all_pos = (
        np.arange(num_pos_endpoints) * time_steps_between_x
        + time_steps_bounce
    )

    # How far does the ball travel in each of those cases
    y_distance_traversed_bounce = dict_meta_bounce[
        "y_distance_traversed_bounce"
    ] = (
        final_velocity_y_magnitude_linspace[:, np.newaxis]
        * time_steps_bounce_all_pos
        * dt
    )
    # What positions do those correspond to
    final_y_positions_bounce = dict_meta_bounce[
        "final_y_positions_bounce"
    ] = np.stack(
        [
            y_distance_traversed_bounce + ball_radius,  # top
            size_y - y_distance_traversed_bounce - ball_radius,  # bottom
        ]
    )

    # Binary arrays for whether the ball enters the grayzone from the left or
    # right and if it is going towards the top or bottom
    sides_left_right_bounce = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_endpoints),
        num_trials_bounce,
    ).astype(int)
    sides_top_bottom_bounce = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_endpoints),
        num_trials_bounce,
    ).astype(int)

    # Compute the signs of the velocities using the sides
    velocity_x_sign_bounce = 2 * sides_left_right_bounce - 1
    velocity_y_sign_bounce = 2 * sides_top_bottom_bounce - 1

    # Precompute indices to sample the velocities from
    indices_velocity_y_magnitude = pyutils.repeat_sequence(
        np.array(list(range(num_y_velocities)) * num_pos_endpoints),
        num_trials_bounce,
    ).astype(int)

    # Keep track of velocities
    dict_meta_bounce["indices_velocity_y_magnitude_counts"] = np.unique(
        indices_velocity_y_magnitude,
        return_counts=True,
    )

    # Precompute colors
    final_color_bounce = pyutils.repeat_sequence(
        np.array(DEFAULT_COLORS),
        num_trials_bounce,
        shuffle=False,
        roll=True,
    ).tolist()

    # Keep track of color counts
    dict_meta_bounce["final_color_counts"] = np.unique(
        final_color_bounce,
        return_counts=True,
        axis=0,
    )

    # Precompute the statistics
    pccnvc_bounce = pyutils.repeat_sequence(
        pccnvc_linspace,
        # np.tile(pccnvc_linspace, num_pccovc),
        num_trials_bounce,
        shuffle=False,
    ).tolist()
    pccovc_bounce = pyutils.repeat_sequence(
        pccovc_linspace,
        # np.tile(pccovc_linspace, num_pccnvc),
        num_trials_bounce,
        shuffle=False,
    ).tolist()

    # Keep track of pcc counts
    dict_meta_bounce["pccnvc_counts"] = np.unique(
        pccnvc_bounce,
        return_counts=True,
    )
    dict_meta_bounce["pccovc_counts"] = np.unique(
        pccovc_bounce,
        return_counts=True,
    )
    dict_meta_bounce["pccnvc_pccovc_counts"] = np.unique(
        [x for x in zip(*(pccnvc_bounce, pccovc_bounce))],
        return_counts=True,
        axis=0,
    )

    final_x_position_indices = []
    final_y_position_indices = []
    final_position_bounce = []
    final_velocity_bounce = []
    meta_bounce = []

    for idx in range(num_trials_bounce):
        # Get an index of position
        idx_position = indices_final_position_index_bounce[idx]

        # Choose the sides to enter the grayzone
        side_left_right = sides_left_right_bounce[idx]
        side_top_bottom = sides_top_bottom_bounce[idx]

        # Get the y velocity index for this trial
        idx_velocity_y = indices_velocity_y_magnitude[idx]

        # Final velocities
        final_velocity_x = (
            final_velocity_x_magnitude * velocity_x_sign_bounce[idx]
        ).item()
        final_velocity_y = (
            final_velocity_y_magnitude_linspace[idx_velocity_y]
            * velocity_y_sign_bounce[idx]
        ).item()
        final_velocity_bounce.append((final_velocity_x, final_velocity_y))

        # Get the bounce indices
        final_x_position_index = final_x_index_bounce[idx_position]
        final_x_position_indices.append(final_x_position_index)
        final_y_position_index = final_y_index_bounce[idx_position]
        final_y_position_indices.append(final_y_position_index)

        # Grab the final positions
        final_position_x = x_grayzone_linspace_sides[
            side_left_right, final_x_position_index
        ].item()
        final_position_y = final_y_positions_bounce[
            side_top_bottom,
            idx_velocity_y,
            final_y_position_index,
        ].item()
        final_position_bounce.append((final_position_x, final_position_y))

        meta_bounce.append(
            {
                "idx": idx,
                "trial": "bounce",
                "idx_time": final_x_position_index,
                "idx_position": idx_position,
                "side_left_right": side_left_right,
                "side_top_bottom": side_top_bottom,
                "idx_velocity_y": idx_velocity_y,
                "length": video_lengths_f_bounce[idx],
                "idx_x_position": final_x_position_index,
                "idx_y_position": final_y_position_index,                    
            }
        )

    # Keep track of position counts
    dict_meta_bounce["x_grayzone_position_indices_counts"] = np.unique(
        final_x_position_indices,
        return_counts=True,
    )
    dict_meta_bounce["y_grayzone_position_indices_counts"] = np.unique(
        final_y_position_indices,
        return_counts=True,
    )
    dict_meta_bounce["x_grayzone_position_counts"] = np.unique(
        [x for x in zip(*final_position_bounce)][0],
        return_counts=True,
    )
    dict_meta_bounce["y_grayzone_position_counts"] = np.unique(
        [y for y in zip(*final_position_bounce)][1],
        return_counts=True,
    )

    # Put bounce parameters together
    trials_bounce = list(
        zip(
            final_position_bounce,
            final_velocity_bounce,
            final_color_bounce,
            pccnvc_bounce,
            pccovc_bounce,
            [
                pvc,
            ]
            * num_trials_bounce,
            meta_bounce,
        )
    )

    if print_stats:
        htaskutils.print_type_stats(
            trials_bounce,
            "bounce",
            duration,
            use_logger=use_logger,
        )

    return trials_bounce, dict_meta_bounce
