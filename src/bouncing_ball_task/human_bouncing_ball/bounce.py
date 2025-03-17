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
    
    dict_meta = {"num_trials": num_trials}

    # Determine bounce positions
    left_x_positions = x_grayzone_linspace - diff * bounce_offset
    assert mask_start < left_x_positions.min()
    right_x_positions = x_grayzone_linspace + diff * bounce_offset
    assert right_x_positions.max() < mask_end

    # How many final positions are there
    final_position_index = dict_meta[
        "final_position_index"
    ] = np.arange(sum(range(1, num_pos_endpoints + 1)))

    # Create indices to x and y positions from these
    # The first x position when entering the grayzone only has one coord,
    # the second has two, until the `num_pos_endpoint`th pos which has
    # `num_pos_endpoint` unique coords
    final_x_index = dict_meta[
        "final_x_index"
    ] = np.repeat(
        np.arange(num_pos_endpoints),
        np.arange(1, num_pos_endpoints + 1),
    ).astype(int)
    # This is reversed for y - there are `num_pos_endpoints` unqiue coords
    # for the first y coordinate (defined as being the point closest to the
    # top or bottom), and that decreases by one until the last one
    final_y_index = dict_meta[
        "final_y_index"
    ] = [j for i in range(num_pos_endpoints + 1) for j in range(i)]
    

    # Create all the indices for the bounce trials
    indices_final_position_index = pyutils.repeat_sequence_imbalanced(
        final_position_index,
        final_x_index,
        num_trials,
        roll=True,
    )

    # Keep track of coordinate counts
    dict_meta[
        "indices_final_position_index_counts"
    ] = np.unique(
        indices_final_position_index,
        return_counts=True,
    )

    # How long does it take to get between each x position
    time_steps_between_x = diff / (final_velocity_x_magnitude * dt)
    # How long from a bounce to the first end position
    time_steps = time_steps_between_x * bounce_offset
    # How long from a bounce to all possible endpoints
    time_steps_all_pos = (
        np.arange(num_pos_endpoints) * time_steps_between_x
        + time_steps
    )

    # How far does the ball travel in each of those cases
    y_distance_traversed = dict_meta[
        "y_distance_traversed"
    ] = (
        final_velocity_y_magnitude_linspace[:, np.newaxis]
        * time_steps_all_pos
        * dt
    )
    # What positions do those correspond to
    final_y_positions = dict_meta[
        "final_y_positions"
    ] = np.stack(
        [
            y_distance_traversed + ball_radius,  # top
            size_y - y_distance_traversed - ball_radius,  # bottom
        ]
    )

    # Binary arrays for whether the ball enters the grayzone from the left or
    # right and if it is going towards the top or bottom
    sides_left_right = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_endpoints),
        num_trials,
    ).astype(int)
    sides_top_bottom = pyutils.repeat_sequence(
        np.array([0, 1] * num_pos_endpoints),
        num_trials,
    ).astype(int)

    # Compute the signs of the velocities using the sides
    velocity_x_sign = 2 * sides_left_right - 1
    velocity_y_sign = 2 * sides_top_bottom - 1

    # Precompute indices to sample the velocities from
    indices_velocity_y_magnitude = pyutils.repeat_sequence(
        np.array(list(range(num_y_velocities)) * num_pos_endpoints),
        num_trials,
    ).astype(int)

    # Keep track of velocities
    dict_meta["indices_velocity_y_magnitude_counts"] = np.unique(
        indices_velocity_y_magnitude,
        return_counts=True,
    )

    # Precompute colors
    final_color = pyutils.repeat_sequence(
        np.array(DEFAULT_COLORS),
        num_trials,
        shuffle=False,
        roll=True,
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

    final_x_position_indices = []
    final_y_position_indices = []
    final_position = []
    final_velocity = []
    meta = []

    for idx in range(num_trials):
        # Get an index of position
        idx_position = indices_final_position_index[idx]

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

        # Get the bounce indices
        final_x_position_index = final_x_index[idx_position]
        final_x_position_indices.append(final_x_position_index)
        final_y_position_index = final_y_index[idx_position]
        final_y_position_indices.append(final_y_position_index)

        # Grab the final positions
        final_position_x = x_grayzone_linspace_sides[
            side_left_right, final_x_position_index
        ].item()
        final_position_y = final_y_positions[
            side_top_bottom,
            idx_velocity_y,
            final_y_position_index,
        ].item()
        final_position.append((final_position_x, final_position_y))

        meta.append(
            {
                "idx": idx,
                "trial": "bounce",
                "idx_time": final_x_position_index,
                "idx_position": idx_position,
                "side_left_right": side_left_right,
                "side_top_bottom": side_top_bottom,
                "idx_velocity_y": idx_velocity_y,
                "length": video_lengths_f[idx],
                "idx_x_position": final_x_position_index,
                "idx_y_position": final_y_position_index,
            }
        )

    # Keep track of position counts
    dict_meta["x_grayzone_position_indices_counts"] = np.unique(
        final_x_position_indices,
        return_counts=True,
    )
    dict_meta["y_grayzone_position_indices_counts"] = np.unique(
        final_y_position_indices,
        return_counts=True,
    )
    dict_meta["x_grayzone_position_counts"] = np.unique(
        [x for x in zip(*final_position)][0],
        return_counts=True,
    )
    dict_meta["y_grayzone_position_counts"] = np.unique(
        [y for y in zip(*final_position)][1],
        return_counts=True,
    )

    # Put bounce parameters together
    trials = list(
        zip(
            final_position,
            final_velocity,
            final_color,
            pccnvc,
            pccovc,
            [pvc,] * num_trials,
            [[],] * num_trials,
            [[],] * num_trials,
            meta,
        )
    )

    if print_stats:
        htaskutils.print_type_stats(
            trials,
            "bounce",
            duration,
            use_logger=use_logger,
        )

    return trials, dict_meta
