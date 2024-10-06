import argparse
import copy
import itertools
import pickle
import random
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from bouncing_ball_task import index
from bouncing_ball_task.constants import CONSTANT_COLOR, DEFAULT_COLORS
from bouncing_ball_task.bouncing_ball import BouncingBallTask
from bouncing_ball_task.utils import gif, logutils, pyutils


def generate_video_parameters(
    size_frame: Iterable[int] = (256, 256),
    ball_radius: int = 10,
    dt: float = 0.1,
    video_length_min_s: Optional[float] = 8.0,  # seconds
    fixed_video_length: Optional[int] = None,
    duration: Optional[int] = 45,
    total_dataset_length: Optional[int] = 35,  # minutes
    mask_center: float = 0.5,
    mask_fraction: float = 1 / 3,
    num_pos_endpoints: int = 5,
    velocity_lower: float = 1 / 12.5,
    velocity_upper: float = 1 / 7.5,
    num_y_velocities: int = 2,
    p_catch_trials: float = 0.05,
    pccnvc_lower: float = 0.01,
    pccnvc_upper: float = 0.045,
    pccovc_lower: float = 0.1,
    pccovc_upper: float = 0.9,
    num_pccnvc: int = 2,
    num_pccovc: int = 3,
    pvc: float = 0.0,
    border_tolerance_outer: float = 2.0,
    border_tolerance_inner: float = 1.0,
    bounce_straight_split: float = 0.5,
    bounce_offset: float = 4 / 5,
    total_videos: Optional[int] = None,
    exp_scale: float = 1.0,  # seconds
):
    """Generates parameters for video simulations of a bouncing ball in a
    bounded frame.

    Parameters
    ----------
    size_frame : Iterable[int], default=(256, 256)
        Dimensions of the video frame (width, height).

    ball_radius : int, default=10
        Radius of the ball in pixels.

    dt : float, default=0.1
        Time step for the simulation in seconds.

    video_length_min_s : Optional[float], default=8.0
        Length of each video in seconds.

    duration : Optional[int], default=45
        Duration of each video segment within the total video in frames.

    total_dataset_length : Optional[int], default=35
        Total length of the dataset in minutes.

    mask_center : float, default=0.5
        Center position of the mask as a fraction of frame height.

    mask_fraction : float, default=1/3
        Fraction of the frame height covered by the mask.

    mask_center : float, default=0.5
        The center x position of the mask.

    num_pos_endpoints : int, default=5
        Number of positive endpoints for the ball trajectory.

    velocity_lower : float, default=1/12.5
        Lower limit for the ball's velocity.

    velocity_upper : float, default=1/7.5
        Upper limit for the ball's velocity.

    num_y_velocities : int, default=2
        Number of distinct velocities in the y-direction.

    p_catch_trials : float, default=0.05
        Probability of trials where the ball catches (stops moving).

    pccnvc_lower : float, default=0.01
        Lower probability of a color change not accompanying a velocity change.

    pccnvc_upper : float, default=0.045
        Upper probability of a color change not accompanying a velocity change.

    pccovc_lower : float, default=0.1
        Lower probability of a color change on a velocity change.

    pccovc_upper : float, default=0.9
        Upper probability of a color change on a velocity change.

    num_pccnvc : int, default=2
        Number of instances where a color change does not accompany a velocity
        change.

    num_pccovc : int, default=3
        Number of instances where a color change accompanies a velocity change.

    pvc : float, default=0.0
        Probability of a velocity change occurring independently.

    border_tolerance_outer : float, default=2.0
        Tolerance for the ball touching the outer border of the frame.

    border_tolerance_inner : float, default=1.0
        Tolerance for the ball touching the inner border of the frame.

    bounce_straight_split : float, default=0.5
        Probability of the ball bouncing straight back upon collision.

    bounce_offset : float, default=0.8
        Offset applied to the ball's bounce direction, expressed as a fraction
        of the ball radius.

    total_videos : Optional[int], default=None
        Total number of videos to generate, if specified.

    exp_scale : float, default=1.0
        Scale factor applied to exponential distributions for time-related
        parameters.

    Returns
    -------
    tuple : (trials_catch, trials_straight, trials_bounce, meta)
        A tuple containing a list of video parameters and a dictionary
        containing metadata about the dataset.
    """
    # Convenience
    size_x, size_y = size_frame
    exp_scale_ms = exp_scale * 1000
    if fixed_video_length is not None:
        video_length_min_f = fixed_video_length
        video_length_min_ms = video_length_min_f * duration

    elif video_length_min_s is not None:
        video_length_min_desired_ms = int(np.array(video_length_min_s) * 1000)
        video_length_min_f = np.rint(
            video_length_min_desired_ms / duration
        ).astype(int)
        video_length_min_ms = video_length_min_f * duration

    # Compute the number of videos we can make given the contraints if not
    # explicitly provided with a total number of videos
    if total_videos is None:
        max_dataset_length_ms = total_dataset_length * 60 * 1000  # * s * ms

        # Compute the lengths of catch trials
        length_ms_rough_catch = max_dataset_length_ms * p_catch_trials
        num_trials_catch = int(length_ms_rough_catch / video_length_min_ms)
        length_ms_catch = num_trials_catch * video_length_min_ms
        video_lengths_f_catch = np.rint(
            np.array(num_trials_catch * [video_length_min_f])
        ).astype(int)

        # Get the remaining lengths of the dataset
        effective_dataset_length_ms = max_dataset_length_ms - length_ms_catch
        length_ms_bounce = effective_dataset_length_ms * bounce_straight_split
        length_ms_straight = effective_dataset_length_ms - length_ms_bounce

        # Estimate the number of videos and then sample lengths
        estimated_num_samples = (
            int(effective_dataset_length_ms / video_length_min_ms) + 1
        )
        estimated_video_lengths_ms = (
            np.random.exponential(exp_scale_ms, estimated_num_samples)
            + video_length_min_ms
        )
        estimated_video_lengths_f = np.rint(
            estimated_video_lengths_ms / duration
        ).astype(int)
        cumsum_video_lengths = np.cumsum(estimated_video_lengths_ms)

        # Grab a subset for the bounce trials
        length_idx_bounce = (
            np.where(cumsum_video_lengths < length_ms_bounce)[0][-1] + 1
        )
        length_ms_bounce = cumsum_video_lengths[length_idx_bounce]
        video_lengths_f_bounce = estimated_video_lengths_f[:length_idx_bounce]
        num_trials_bounce = len(video_lengths_f_bounce)

        # Get another subset for the straight trials
        length_idx_straight = (
            np.where(
                cumsum_video_lengths[length_idx_bounce + 1 :] - length_ms_bounce
                < length_ms_straight
            )[0][-1]
            + 1
        )
        length_ms_straight = (
            cumsum_video_lengths[length_idx_bounce + length_idx_straight + 1]
            - length_ms_bounce
        )
        video_lengths_f_straight = estimated_video_lengths_f[
            length_idx_bounce + 1 : length_idx_straight + length_idx_bounce + 1
        ]
        num_trials_straight = len(video_lengths_f_straight)

        # Get the overall lengths and do a quick sanity check
        total_videos = (
            num_trials_catch + num_trials_bounce + num_trials_straight
        )
        length_trials_total_ms = duration * (
            sum(video_lengths_f_catch)
            + sum(video_lengths_f_bounce)
            + sum(video_lengths_f_straight)
        )
        assert length_trials_total_ms < max_dataset_length_ms

    else:
        # Split according to the desired number of trials
        num_trials_catch = int(total_videos * p_catch_trials)
        video_lengths_f_catch = np.rint(
            np.array(num_trials_catch * [video_length_min_f])
        ).astype(int)

        num_trials_noncatch = total_videos - num_trials_catch
        num_trials_bounce = int(num_trials_noncatch * bounce_straight_split)
        if fixed_video_length is None:
            video_lengths_f_bounce = np.rint(
                (
                    np.random.exponential(exp_scale_ms, num_trials_bounce)
                    + video_length_min_ms
                )
                / duration
            ).astype(int)
        else:
            video_lengths_f_bounce = np.array(
                [fixed_video_length] * num_trials_bounce
            ).astype(int)

        num_trials_straight = num_trials_noncatch - num_trials_bounce
        if fixed_video_length is None:
            video_lengths_f_straight = np.rint(
                (
                    np.random.exponential(exp_scale_ms, num_trials_straight)
                    + video_length_min_ms
                )
                / duration
            ).astype(int)
        else:
            video_lengths_f_straight = np.array(
                [fixed_video_length] * num_trials_straight
            ).astype(int)

        length_trials_total_ms = duration * (
            sum(video_lengths_f_catch)
            + sum(video_lengths_f_bounce)
            + sum(video_lengths_f_straight)
        )

    # Useful quantities
    video_length_max_f = max(
        [
            max(lengths)
            for lengths in (
                video_lengths_f_catch,
                video_lengths_f_bounce,
                video_lengths_f_straight,
            )
            if lengths.size > 0
        ]
    )
    video_length_max_ms = video_length_max_f * duration

    # Containers
    trials_catch, trials_straight, trials_bounce = [], [], []
    meta = {
        "num_trials": total_videos,
        "length_trials_ms": length_trials_total_ms,
        "video_length_min_ms": video_length_min_ms,
        "video_length_max_ms": video_length_max_ms,
        "video_length_min_f": video_length_min_f,
        "video_length_max_f": video_length_max_f,
        "ball_radius": ball_radius,
        "dt": dt,
        "duration": duration,
        "exp_scale": exp_scale,
    }

    print_task_summary(meta)

    # Velocity Linspaces
    final_velocity_x_magnitude = meta["final_velocity_x_magnitude"] = (
        np.mean([velocity_lower, velocity_upper]) * size_x
    )
    final_velocity_y_magnitude_linspace = meta[
        "final_velocity_y_magnitude_linspace"
    ] = np.linspace(
        velocity_lower * size_y,
        velocity_upper * size_y,
        num_y_velocities,
        endpoint=True,
    )

    # Probability Linspaces
    pccnvc_linspace = meta["pccnvc_linspace"] = np.linspace(
        pccnvc_lower,
        pccnvc_upper,
        num_pccnvc,
        endpoint=True,
    )
    pccovc_linspace = meta["pccovc_linspace"] = np.linspace(
        pccovc_lower,
        pccovc_upper,
        num_pccovc,
        endpoint=True,
    )

    # Mask positions
    meta["mask_center"] = mask_center
    meta["mask_fraction"] = mask_fraction
    mask_size = meta["mask_size"] = int(np.round(size_x * mask_fraction))
    mask_start = meta["mask_start"] = int(
        np.round((mask_center * size_x) - mask_size / 2)
    )
    mask_end = meta["mask_end"] = size_x - mask_start

    # x positions Non-Grayzone
    nongrayzone_left_x_range = (
        border_tolerance_outer * ball_radius,
        mask_start,
    )
    nongrayzone_right_x_range = (
        mask_end,
        size_x - border_tolerance_outer * ball_radius,
    )

    if num_trials_catch > 0:
        meta["catch"] = {}
        meta["catch"]["num_trials"] = num_trials_catch

        # Keep track of possible catch x positions
        meta["catch"]["nongrayzone_left_x_range"] = nongrayzone_left_x_range
        meta["catch"]["nongrayzone_right_x_range"] = nongrayzone_right_x_range

        # Catch Trial Positions
        final_x_position_catch = pyutils.alternating_ab_sequence(
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
            num_trials_catch,
        ).tolist()

        final_y_position_catch = pyutils.repeat_sequence(
            np.linspace(
                border_tolerance_outer * ball_radius,
                size_y - border_tolerance_outer * ball_radius,
                2 * num_pos_endpoints,
                endpoint=True,
            ),
            num_trials_catch,
        ).tolist()
        final_position_catch = zip(
            final_x_position_catch, final_y_position_catch
        )

        # Catch trial velocities
        final_velocity_catch = zip(
            [final_velocity_x_magnitude.item()] * num_trials_catch,
            pyutils.repeat_sequence(
                final_velocity_y_magnitude_linspace,
                num_trials_catch,
            ).tolist(),
        )

        # Catch colors
        final_color_catch = pyutils.repeat_sequence(
            np.array(DEFAULT_COLORS),
            num_trials_catch,
        ).tolist()

        # Catch probabilities
        pccnvc_catch = pyutils.repeat_sequence(
            pccnvc_linspace,
            num_trials_catch,
        ).tolist()
        pccovc_catch = pyutils.repeat_sequence(
            pccovc_linspace,
            num_trials_catch,
        ).tolist()

        # Put parameters together
        trials_catch = list(
            zip(
                final_position_catch,
                final_velocity_catch,
                final_color_catch,
                pccnvc_catch,
                pccovc_catch,
                [
                    pvc,
                ]
                * num_trials_catch,
                [
                    {
                        "idx": idx,
                        "trial": "catch",
                        "length": video_lengths_f_catch[idx],
                    }
                    for idx in range(num_trials_catch)
                ],
            )
        )

        print_type_stats(trials_catch, "catch", duration)

    # Normal trials
    # x position Grayzone linspace
    x_grayzone_linspace = np.linspace(
        mask_start + border_tolerance_inner * ball_radius,
        mask_end - border_tolerance_inner * ball_radius,
        num_pos_endpoints + 2,
        endpoint=True,
    )[1:-1]

    # Final x position linspaces that correspond to approaching from each side
    x_grayzone_linspace_reversed = x_grayzone_linspace[::-1]
    x_grayzone_linspace_sides = meta["x_grayzone_linspace_sides"] = np.vstack(
        [
            x_grayzone_linspace,
            x_grayzone_linspace_reversed,
        ]
    )

    # Get the final y positions depending on num of end x
    diff = meta["diff"] = np.diff(x_grayzone_linspace).mean()
    multipliers = np.arange(1, num_pos_endpoints + 1)
    position_y_diff = final_velocity_y_magnitude_linspace * dt * diff

    # Straight conditions
    if num_trials_straight > 0:
        meta["straight"] = {}
        meta["straight"]["num_trials"] = num_trials_straight

        indices_time_in_grayzone_straight = pyutils.repeat_sequence(
            np.arange(num_pos_endpoints),
            num_trials_straight,
            shuffle=False,
        )

        # Binary arrays for whether the ball enters the grayzone from the left or
        # right and if it is going towards the top or bottom
        sides_left_right_straight = pyutils.repeat_sequence(
            np.array([0, 1] * num_pos_endpoints),
            num_trials_straight,
        )
        sides_top_bottom_straight = pyutils.repeat_sequence(
            np.array([0, 1] * num_pos_endpoints),
            num_trials_straight,
        )

        # Compute the signs of the velocities using the sides
        velocity_x_sign_straight = 2 * sides_left_right_straight - 1
        velocity_y_sign_straight = (
            2 * np.logical_not(sides_top_bottom_straight) - 1
        )

        # Precompute indices to sample the velocities from
        indices_velocity_y_magnitude = pyutils.repeat_sequence(
            np.array(list(range(num_y_velocities)) * num_pos_endpoints),
            num_trials_straight,
        )

        # Keep track of velocities
        meta["straight"]["indices_velocity_y_magnitude_counts"] = np.unique(
            indices_velocity_y_magnitude,
            return_counts=True,
        )

        # Straight y positions
        y_distance_traversed_straight = (
            position_y_diff[:, np.newaxis] * multipliers
        )

        # This is shape [2 x num_vel x num_pos_endpoints x 2*num_pos_endpoints]
        # [top/bottom, each vel, num x positions, num y pos per x pos]
        final_y_positions_straight = meta["straight"][
            "final_y_positions"
        ] = np.stack(
            [
                # Top
                np.linspace(
                    np.ones_like(y_distance_traversed_straight)
                    * 2
                    * ball_radius,
                    size_y - y_distance_traversed_straight - 2 * ball_radius,
                    2 * num_pos_endpoints,
                    endpoint=True,
                    axis=-1,
                ),
                # Bottom
                np.linspace(
                    size_y - 2 * ball_radius,
                    y_distance_traversed_straight + 2 * ball_radius,
                    2 * num_pos_endpoints,
                    endpoint=True,
                    axis=-1,
                ),
            ]
        )

        # Final y position linspaces
        y_grayzone_positions = pyutils.repeat_sequence(
            np.linspace(
                border_tolerance_outer * ball_radius,
                size_y - border_tolerance_outer * ball_radius,
                2 * num_pos_endpoints,
                endpoint=True,
            ),
            num_trials_straight,
        )

        # keep track of grayzone positions
        meta["straight"]["y_grayzone_position_counts"] = np.unique(
            y_grayzone_positions,
            return_counts=True,
            axis=0,
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
        meta["straight"]["final_color_counts"] = np.unique(
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
        meta["straight"]["pccnvc_counts"] = np.unique(
            pccnvc_straight,
            return_counts=True,
        )
        meta["straight"]["pccovc_counts"] = np.unique(
            pccovc_straight,
            return_counts=True,
        )
        meta["straight"]["pccnvc_pccovc_counts"] = np.unique(
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
                    "length": video_lengths_f_straight[idx],
                }
            )

        # Keep track of position counts
        meta["straight"]["x_grayzone_position_counts"] = np.unique(
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

        print_type_stats(trials_straight, "straight", duration)

    # Bounce conditions
    if num_trials_bounce > 0:
        meta["bounce"] = {}
        meta["bounce"]["num_trials"] = num_trials_bounce

        # Determine bounce positions
        left_bounce_x_positions = x_grayzone_linspace - diff * bounce_offset
        assert mask_start < left_bounce_x_positions.min()
        right_bounce_x_positions = x_grayzone_linspace + diff * bounce_offset
        assert right_bounce_x_positions.max() < mask_end

        # How many final positions are there
        final_position_index_bounce = meta["bounce"][
            "final_position_index_bounce"
        ] = np.arange(sum(range(1, num_pos_endpoints + 1)))

        # Create indices to x and y positions from these
        # The first x position when entering the grayzone only has one coord,
        # the second has two, until the `num_pos_endpoint`th pos which has
        # `num_pos_endpoint` unique coords
        final_x_index_bounce = meta["bounce"][
            "final_x_index_bounce"
        ] = np.repeat(
            np.arange(num_pos_endpoints),
            np.arange(1, num_pos_endpoints + 1),
        )
        # This is reversed for y - there are `num_pos_endpoints` unqiue coords
        # for the first y coordinate (defined as being the point closest to the
        # top or bottom), and that decreases by one until the last one
        final_y_index_bounce = meta["bounce"][
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
        meta["bounce"][
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
        y_distance_traversed_bounce = meta["bounce"][
            "y_distance_traversed_bounce"
        ] = (
            final_velocity_y_magnitude_linspace[:, np.newaxis]
            * time_steps_bounce_all_pos
            * dt
        )
        # What positions do those correspond to
        final_y_positions_bounce = meta["bounce"][
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
        )
        sides_top_bottom_bounce = pyutils.repeat_sequence(
            np.array([0, 1] * num_pos_endpoints),
            num_trials_bounce,
        )

        # Compute the signs of the velocities using the sides
        velocity_x_sign_bounce = 2 * sides_left_right_bounce - 1
        velocity_y_sign_bounce = 2 * sides_top_bottom_bounce - 1

        # Precompute indices to sample the velocities from
        indices_velocity_y_magnitude = pyutils.repeat_sequence(
            np.array(list(range(num_y_velocities)) * num_pos_endpoints),
            num_trials_bounce,
        )

        # Keep track of velocities
        meta["bounce"]["indices_velocity_y_magnitude_counts"] = np.unique(
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
        meta["bounce"]["final_color_counts"] = np.unique(
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
        meta["bounce"]["pccnvc_counts"] = np.unique(
            pccnvc_bounce,
            return_counts=True,
        )
        meta["bounce"]["pccovc_counts"] = np.unique(
            pccovc_bounce,
            return_counts=True,
        )
        meta["bounce"]["pccnvc_pccovc_counts"] = np.unique(
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
        meta["bounce"]["x_grayzone_position_indices_counts"] = np.unique(
            final_x_position_indices,
            return_counts=True,
        )
        meta["bounce"]["y_grayzone_position_indices_counts"] = np.unique(
            final_y_position_indices,
            return_counts=True,
        )
        meta["bounce"]["x_grayzone_position_counts"] = np.unique(
            [x for x in zip(*final_position_bounce)][0],
            return_counts=True,
        )
        meta["bounce"]["y_grayzone_position_counts"] = np.unique(
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

        print_type_stats(trials_bounce, "bounce", duration)

    return trials_catch, trials_straight, trials_bounce, meta


def print_task_summary(meta):
    length_trials = meta["length_trials_ms"] / 60000
    length_trials_min = int(length_trials)
    length_trials_s = np.round((length_trials % 1) * 60, 1)
    logger.info("Dataset Generation Summary")
    logger.info(
        f'  Num Total Trials: {meta["num_trials"]} ({length_trials_min} min {length_trials_s} sec)'
    )
    max_key_len = max([len(key) for key in meta.keys()])

    for key, val in meta.items():
        if key == "num_trials":
            continue
        key += ":"
        logger.info(f"    {key:<{max_key_len + 2}}{val}")


def print_type_stats(trials, trial_type, duration, return_str=False):
    position, velocity, color, pccnvc, pccovc, pvc, meta = zip(*trials)
    stats_comb = [f"{nvc}-{ovc}" for nvc, ovc in zip(pccnvc, pccovc)]
    list_messages = []

    trial_type_lengths_s = (
        np.array([m["length"] for m in meta]) * duration
    ) / 1000
    trial_type_lengths_min = trial_type_lengths_s / 60
    length_trial_total_min = int(sum(trial_type_lengths_min))
    length_trial_total_s = np.round((sum(trial_type_lengths_min) % 1) * 60, 1)

    length_trial_s_min = np.round(min(trial_type_lengths_s), 1)
    length_trial_s_max = np.round(max(trial_type_lengths_s), 1)

    msg = f"  Num {trial_type.title()} Trials: {len(trials)} ({length_trial_total_min} min {length_trial_total_s} sec)"
    if return_str:
        list_messages.append(msg)
    else:
        logger.info(msg)

    trial_type_stats = [
        ("Min Video Length (s):", length_trial_s_min),
        ("Max Video Length (s):", length_trial_s_max),
        ("Color Splits:", Counter(np.argmax(color, axis=1))),
        ("pccnvc Splits:", Counter(pccnvc)),
        ("pccovc Splits:", Counter(pccovc)),
        ("Stat Comb Splits:", Counter(stats_comb)),
    ]

    if trial_type.lower() != "catch":
        trial_type_stats += [
            ("End time Splits:", Counter(m["idx_time"] for m in meta)),
            ("Left/Right Splits:", Counter(m["side_left_right"] for m in meta)),
            ("Top/Bottom Splits:", Counter(m["side_top_bottom"] for m in meta)),
            ("Velocity Splits:", Counter(m["idx_velocity_y"] for m in meta)),
        ]

    if trial_type.lower() == "bounce":
        trial_type_stats += [
            ("Final y position splits", Counter(m["idx_y_position"] for m in meta)),
        ]
        df = pd.DataFrame(meta)
        for idx_x, df_idx in df.groupby("idx_time"):
            trial_type_stats += [
                (
                    f"x={idx_x} - Final y position splits",
                    Counter(df_idx["idx_y_position"]),
                )
            ]
        # import ipdb; ipdb.set_trace()

    max_desc_len = max([len(desc) for (desc, _) in trial_type_stats])
    for (description, value) in trial_type_stats:
        if isinstance(value, Counter):
            total = value.total()
            value_str = f"{[np.round(val[1] / total, 2) for val in sorted(value.items(), key=lambda pair: pair[0])]}"
        else:
            value_str = f"{value}"
        msg = f"    {description:<{max_desc_len + 2}}{value_str}"
        if return_str:
            list_messages.append(msg)
        else:
            logger.debug(msg)

    if return_str:
        return list_messages


def generate_block_parameters(
    params,
    num_blocks,
    shuffle_block=True,
    shuffle_block_elements=True,
):
    """Distributes elements from a tuple of iterables into blocks with an equal
    or nearly equal number of elements, then optionally shuffles the elements
    within each block and the order of the blocks.

    Parameters
    ----------
    params : tuple of iterables
        A tuple containing iterables, potentially of different lengths.

    num_blocks : int
        The number of blocks to distribute elements into.

    shuffle_block_elements : bool, optional
        If True, shuffles the order of elements within each block. Default is
        True.

    shuffle_block : bool, optional
        If True, shuffles the order of the blocks. Default is True.

    Returns
    -------
    list of blocks
        A list of blocks, where each block is a list containing a distributed
        portion of the original elements split as evenly as possible and
        optionally shuffled.
    """
    # So we dont change the passed in list
    params = copy.deepcopy(params)

    # Shuffle the elements before grouping
    if shuffle_block_elements:
        for param in params:
            random.shuffle(param)

    # Flatten params to get a single list of all elements
    params_flattened = list(itertools.chain.from_iterable(params))
    num_params = len(params_flattened)

    # Create blocks with as even distribution as possible
    blocks = [[] for _ in range(num_blocks)]
    for i, param in enumerate(params_flattened):
        blocks[i % num_blocks].append(param)

    # Shuffle the order of elements within each block if requested
    if shuffle_block_elements:
        for block in blocks:
            random.shuffle(block)

    # Shuffle the order of the blocks if requested
    if shuffle_block:
        random.shuffle(blocks)

    # Build the meta data for each block
    meta_blocks = [
        {
            "num_trials": len(block),
            "length_trials_ms": sum([meta["length"] for *_, meta in block]),
        }
        for block in blocks
    ]

    return list(zip(blocks, meta_blocks))


def print_block_stats(params_blocks, duration):
    blocks, meta_blocks = zip(*params_blocks)
    logger.info(
        f"  Num blocks: {len(blocks)} - Lengths = {[len(block) for block in blocks]}"
    )

    for i, block in enumerate(blocks):
        position, velocity, color, pccnvc, pccovc, pvc, meta = zip(*block)
        stats_comb = [f"{nvc}-{ovc}" for nvc, ovc in zip(pccnvc, pccovc)]

        block_lengths_s = (
            np.array([m["length"] for m in meta]) * duration
        ) / 1000
        block_lengths_min = block_lengths_s / 60
        length_block_total_min = int(sum(block_lengths_min))
        length_block_total_s = np.round((sum(block_lengths_min) % 1) * 60, 1)

        length_block_s_min = np.round(min(block_lengths_s), 1)
        length_block_s_max = np.round(max(block_lengths_s), 1)

        logger.debug(
            f"    Block {i+1} - {len(block)} videos ({length_block_total_min} min {length_block_total_s} sec)"
        )

        block_stats = [
            ("Min Video Length (s):", length_block_s_min),
            ("Max Video Length (s):", length_block_s_max),
            ("Trial type Counts:", Counter([m["trial"] for m in meta])),
            ("Color Counts:", Counter(np.argmax(color, axis=1))),
            ("pccnvc Counts:", Counter(pccnvc)),
            ("pccovc Counts:", Counter(pccovc)),
            ("stats comb Counts:", Counter(stats_comb)),
        ]
        max_desc_len = max([len(desc) for (desc, _) in block_stats])
        for (description, value) in block_stats:
            if isinstance(value, Counter):
                total = value.total()
                value_str = f"{[np.round(val[1] / total, 2) for val in sorted(value.items(), key=lambda pair: pair[0])]}"
                # value_str = f"{sorted(value.items(), key=lambda pair: pair[0])}"
            else:
                value_str = f"{value}"
            logger.trace(f"      {description:<{max_desc_len + 2}}{value_str}")


def plot_params(
    params,
    size_frame: Iterable[int] = (256, 256),
    sequence_length=60,
    duration=40,
    target_future_timestep=1,
    ball_radius: int = 10,
    dt: float = 0.1,
    mask_center: float = 0.5,
    mask_fraction: float = 1 / 3,
    mask_color=(127, 127, 127),
    min_t_color_change=15,
    sample_mode="parameter_array",
    target_mode="parameter_array",
    save_target=True,
    sequence_mode="reverse",
    debug=True,
    save_animation=False,
    display_animation=True,
    mode="combined",
    multiplier=2,
    include_timestep=True,
    as_mp4=True,
):

    batch_size = len(params)
    (
        initial_position,
        initial_velocity,
        initial_color,
        pccnvc,
        pccovc,
        pvc,
        meta,
    ) = zip(*params)
    task = BouncingBallTask(
        size_frame=size_frame,
        sequence_length=sequence_length,
        ball_radius=ball_radius,
        target_future_timestep=target_future_timestep,
        dt=dt,
        batch_size=batch_size,
        sample_mode=sample_mode,
        target_mode=target_mode,
        mask_center=mask_center,
        mask_fraction=mask_fraction,
        mask_color=mask_color,
        sequence_mode=sequence_mode,
        debug=debug,
        probability_velocity_change=pvc,
        probability_color_change_no_velocity_change=pccnvc,
        probability_color_change_on_velocity_change=pccovc,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        initial_color=initial_color,
        min_t_color_change=min_t_color_change,
    )
    initial_params = np.concatenate(
        [np.array(initial_position), np.stack(initial_color)],
        axis=1,
    )
    assert np.all(
        np.isclose(np.array(initial_position), task.sequence[-1][1][:, :2])
    )
    assert np.all(
        np.isclose(np.stack(initial_color), task.sequence[-1][1][:, 2:])
    )
    samples, targets = zip(*[x for x in task])
    assert np.all(
        np.isclose(np.array(initial_position), task.sequence[-1][1][:, :2])
    )
    assert np.all(
        np.isclose(np.stack(initial_color), task.sequence[-1][1][:, 2:])
    )

    samples = np.array(samples)
    targets = np.array(targets)

    for i in range(batch_size):
        print(np.array(params[i][0]) * multiplier, params[i][1:])
        assert np.all(
            np.isclose(np.array(initial_position[i]), targets[-1][i, :2])
        )
        assert np.all(
            np.isclose(np.stack(initial_color[i]), targets[-1][i, 2:])
        )

        _, _ = task.animate(
            arrays=(targets[:, i, :]),
            path_dir=None,
            name="",
            duration=duration,
            mode=mode,
            multiplier=multiplier,
            save_target=save_target,
            save_animation=save_animation,
            display_animation=display_animation,
            num_sequences=1,
            as_mp4=as_mp4,
            include_timestep=include_timestep,
            return_path=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human Bouncing Ball Data Generation Script"
    )

    parser.add_argument(
        "--size_frame",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Frame size as two integers (width, height)",
    )
    parser.add_argument(
        "--ball_radius", type=int, default=10, help="Radius of the ball"
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Time delta")
    parser.add_argument(
        "--num_blocks", type=int, default=20, help="Number of blocks"
    )
    parser.add_argument(
        "--video_length_min_s",
        type=float,
        default=6.5,
        help="Video length in seconds",
    )
    parser.add_argument("--duration", type=int, default=35, help="Duration")
    parser.add_argument(
        "--total_dataset_length",
        type=int,
        default=35,
        help="Total dataset length",
    )
    parser.add_argument(
        "--mask_center", type=float, default=0.5, help="Mask center"
    )
    parser.add_argument(
        "--mask_fraction", type=float, default=1 / 3, help="Mask fraction"
    )
    parser.add_argument(
        "--num_pos_endpoints",
        type=int,
        default=5,
        help="Number of positive endpoints",
    )
    parser.add_argument(
        "--velocity_lower",
        type=float,
        default=1 / 12.5,
        help="Lower velocity limit",
    )
    parser.add_argument(
        "--velocity_upper",
        type=float,
        default=1 / 7.5,
        help="Upper velocity limit",
    )
    parser.add_argument(
        "--num_y_velocities", type=int, default=2, help="Number of y velocities"
    )
    parser.add_argument(
        "--p_catch_trials",
        type=float,
        default=0.05,
        help="Probability of catch trials",
    )
    parser.add_argument("--dryrun", action="store_true", help="Enable dry run")

    parser.add_argument(
        "--pccnvc_lower", type=float, default=0.01, help="PCCNVC lower limit"
    )
    parser.add_argument(
        "--pccnvc_upper", type=float, default=0.045, help="PCCNVC upper limit"
    )
    parser.add_argument(
        "--pccovc_lower", type=float, default=0.1, help="PCCOVC lower limit"
    )
    parser.add_argument(
        "--pccovc_upper", type=float, default=0.9, help="PCCOVC upper limit"
    )
    parser.add_argument(
        "--num_pccnvc", type=int, default=2, help="Number of PCCNVC"
    )
    parser.add_argument(
        "--num_pccovc", type=int, default=3, help="Number of PCCOVC"
    )
    parser.add_argument("--pvc", type=float, default=0.0, help="PVC")
    parser.add_argument(
        "--border_tolerance_outer",
        type=float,
        default=2.0,
        help="Border tolerance outer",
    )
    parser.add_argument(
        "--border_tolerance_inner",
        type=float,
        default=1.0,
        help="Border tolerance inner",
    )
    parser.add_argument(
        "--bounce_straight_split",
        type=float,
        default=0.5,
        help="Bounce straight split",
    )

    parser.add_argument(
        "--sequence_length", type=int, default=None, help="Sequence length"
    )
    parser.add_argument(
        "--target_future_timestep",
        type=int,
        default=0,
        help="Target future timestep",
    )
    parser.add_argument(
        "--mask_color",
        type=int,
        nargs=3,
        default=(127, 127, 127),
        help="Mask color as three integers (R, G, B)",
    )
    parser.add_argument(
        "--min_t_color_change",
        type=int,
        default=20,
        help="Minimum time for color change",
    )
    parser.add_argument(
        "--sample_mode", type=str, default="parameter_array", help="Sample mode"
    )
    parser.add_argument(
        "--target_mode", type=str, default="parameter_array", help="Target mode"
    )
    parser.add_argument(
        "--sequence_mode", type=str, default="reverse", help="Sequence mode"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--display_animation", action="store_true", help="Display animation"
    )
    parser.add_argument("--mode", type=str, default="original", help="Mode")
    parser.add_argument("--multiplier", type=int, default=2, help="Multiplier")
    parser.add_argument(
        "--include_timestep",
        action="store_true",
        help="Include timestep in the output",
    )
    parser.add_argument(
        "--total_videos", type=int, default=None, help="Total number of videos"
    )
    parser.add_argument(
        "--dir_base",
        type=Path,
        default=index.dir_data / "hmdcpd",
        help="Base directory for output",
    )
    parser.add_argument(
        "--name_dataset",
        type=str,
        default="hbb_dataset_" + datetime.now().strftime("%y%m%d_%H%M%S"),
        help="Dataset name",
    )
    parser.add_argument(
        "--exp_scale",
        type=float,
        default=1.0,
        help="Exponential scaling factor",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Setup the logger
    logger = logutils.configure_logger(verbose=args.verbose, trace=args.debug)

    # Calculate the directory for storing dataset based on provided base
    # directory and dataset name
    dir_base = Path(args.dir_base)
    dir_dataset = dir_base / args.name_dataset
    dir_all_videos = dir_dataset / "videos"

    # Generate video parameters based on the arguments provided
    *params, meta_dataset = generate_video_parameters(
        size_frame=args.size_frame,
        ball_radius=args.ball_radius,
        dt=args.dt,
        video_length_min_s=args.video_length_min_s,
        duration=args.duration,
        total_dataset_length=args.total_dataset_length,
        mask_center=args.mask_center,
        mask_fraction=args.mask_fraction,
        num_pos_endpoints=args.num_pos_endpoints,
        velocity_lower=args.velocity_lower,
        velocity_upper=args.velocity_upper,
        num_y_velocities=args.num_y_velocities,
        p_catch_trials=args.p_catch_trials,
        num_pccnvc=args.num_pccnvc,
        num_pccovc=args.num_pccovc,
        border_tolerance_outer=args.border_tolerance_outer,
        border_tolerance_inner=args.border_tolerance_inner,
        bounce_straight_split=args.bounce_straight_split,
        total_videos=args.total_videos,
        exp_scale=args.exp_scale,
    )

    # Add some extra parameters to the metadata
    meta_dataset["name"] = args.name_dataset
    meta_dataset["min_t_color_change"] = args.min_t_color_change

    num_params = sum(len(param) for param in params)
    if args.total_videos is not None:
        assert num_params == args.total_videos

    if args.sequence_length is None:
        max_length = meta_dataset["video_length_max_f"]
    else:
        max_length = args.sequence_length

    params_blocks = generate_block_parameters(
        params,
        args.num_blocks,
        shuffle_block=True,
        shuffle_block_elements=True,
    )

    # Basic Checks
    assert len(params_blocks) == args.num_blocks
    block_lengths = [len(block) for block, _ in params_blocks]
    assert sum(block_lengths) == num_params
    assert max(block_lengths) - min(block_lengths) <= 1
    print_block_stats(params_blocks, args.duration)

    block_list = [
        f"block_{i+1}"
        for i, (block, _) in enumerate(params_blocks)
        for param in block
    ]
    flattened_params = [param for block, _ in params_blocks for param in block]
    positions, velocities, colors, pccnvcs, pccovcs, pvcs, meta_trials = (
        list(param) for param in zip(*flattened_params)
    )

    task = BouncingBallTask(
        size_frame=args.size_frame,
        sequence_length=max_length,
        ball_radius=args.ball_radius,
        target_future_timestep=args.target_future_timestep,
        dt=args.dt,
        batch_size=len(positions),
        sample_mode=args.sample_mode,
        target_mode=args.target_mode,
        mask_center=args.mask_center,
        mask_fraction=args.mask_fraction,
        mask_color=args.mask_color,
        sequence_mode=args.sequence_mode,
        debug=True,
        probability_velocity_change=pvcs,
        probability_color_change_no_velocity_change=pccnvcs,
        probability_color_change_on_velocity_change=pccovcs,
        initial_position=positions,
        initial_velocity=velocities,
        initial_color=colors,
        min_t_color_change=args.min_t_color_change,
    )

    initial_params = np.concatenate(
        [np.array(positions), np.stack(colors)],
        axis=1,
    )
    assert np.all(np.isclose(np.array(positions), task.sequence[-1][1][:, :2]))
    assert np.all(np.isclose(np.stack(colors), task.sequence[-1][1][:, 2:]))

    # samples, targets = zip(*[x for x in task])
    samples = task.samples # np.array(samples).transpose(1, 0, 2)
    targets = task.targets #np.array(targets).transpose(1, 0, 2)

    # ERROR HERE - NEED TO ADAPT TO "SOURCE" TRAGET CHANGE MODE
    color_changes = [param["color_change"] for param in task.all_parameters[1:]]
    color_changes = np.array(
        2 * [np.zeros_like(color_changes[0])] + color_changes[:-1]
    ).transpose(1, 0)
    columns = ["x", "y", "r", "g", "b"]
    colors = ["red", "green", "blue"]

    row_list = []
    last_block_num = None
    idx_block_video = None

    msg = f"Saving dataset to {dir_dataset} (dir_dataset)"
    if args.dryrun:
        msg = f"Dryrun - {msg}"
    logger.info(msg)

    for idx_param, (param, sample, target, color_change, block) in enumerate(
        zip(
            flattened_params,
            samples,
            targets,
            color_changes,
            block_list,
        )
    ):
        # Set the path for the block
        dir_block = dir_all_videos / block
        block_num = int(dir_block.stem.split("_")[-1])

        # Grab the relevant params
        position, velocity, color, pccnvc, pccovc, pvc, meta_trial = param
        length = meta_trial["length"]

        # Shorten the videos to the specified length
        sample = sample[max_length - length :]
        target = target[max_length - length :]
        color_change = color_change[max_length - length :]

        # Grab the final color
        target_color = colors[np.argmax(target[-1, 2:])]
        timestamps = np.arange(length) * args.duration

        # Create the df for the targets and color changes
        target_df = pd.DataFrame(target, index=timestamps, columns=columns)
        target_df.index.name = "Timestamp"
        color_change_df = pd.DataFrame(
            color_change,
            index=timestamps,
            columns=["Color Changed"],
        )
        color_change_df.index.name = "Timestamp"

        # Create the index
        if last_block_num == block_num:
            idx_block_video += 1
        else:
            last_block_num = block_num
            idx_block_video = 1

        # Create the relevant paths
        dir_video = dir_block / f"video_{idx_block_video}"

        msg = f"Generating video files in /dir_videos/{dir_block.stem}/video_{idx_block_video}"
        if args.dryrun:
            msg = f"  Dryrun - {msg}"
        elif not dir_video.exists():
            dir_video.mkdir(parents=True)
        logger.debug(msg)

        path_target_df = dir_video / f"video_{idx_block_video}_parameters.csv"
        path_color_change_df = (
            dir_video / f"video_{idx_block_video}_color_change.csv"
        )
        if args.dryrun:
            logger.trace(
                f"    Dryrun - Saving target df as {path_target_df.stem}.csv"
            )
            logger.trace(
                f"    Dryrun - Saving color_change df as {path_color_change_df.stem}.csv"
            )
        else:
            target_df.to_csv(str(path_target_df))
            color_change_df.to_csv(str(path_color_change_df))

        video_name = f"video_{idx_block_video}_{target_color}"
        if args.dryrun:
            path_video = dir_video / f"{video_name}.mp4"
            logger.trace(f"    Dryrun - Saving video in {path_video.stem}.mp4")
        else:
            path_video, _ = task.animate(
                target,
                path_dir=dir_video,
                name=video_name,
                duration=args.duration,
                mode=args.mode,
                multiplier=args.multiplier,
                save_target=True,
                save_animation=True,
                display_animation=args.display_animation,
                num_sequences=1,
                as_mp4=True,
                include_timestep=args.include_timestep,
                return_path=True,
            )

        meta_trial.update(
            {
                "Block": block_num,
                "Final Color": target_color,
                "Final X Position": sample[-1, 0],
                "Final Y Position": sample[-1, 1],
                "Final X Velocity": -velocity[0],
                "Final Y Velocity": -velocity[1],
                "PCCNVC": pccnvc,
                "PCCOVC": pccovc,
                "PVC": pvc,
                "Dir Base": dir_dataset,
                "Dir Block": dir_block,
                "Dir Video": dir_video,
                "Path Video": path_video,
                "Path Color Changes": Path(path_color_change_df),
                "Path Parameters": Path(path_target_df),
            }
        )

        row_list.append(meta_trial)

    df_meta = pd.DataFrame(row_list)    
    path_df_trial_meta = dir_dataset / "trial_meta.csv"
    msg = f"Saving trial metadata to to {path_df_trial_meta}"
    if args.dryrun:
        msg = f"Dryrun - {msg}"
        logger.info(msg)
    else:
        logger.info(msg)
        df_meta.to_csv(str(path_df_trial_meta))

    path_dataset_meta = dir_dataset / "dataset_meta.pkl"
    msg = f"Saving dataset metadata to to {path_dataset_meta}"
    if args.dryrun:
        msg = f"Dryrun - {msg}"
        logger.info(msg)
    else:
        logger.info(msg)
        with open(str(path_dataset_meta), "wb") as handle:
            pickle.dump(meta_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
