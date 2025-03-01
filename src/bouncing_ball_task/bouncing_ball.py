import argparse
import sys
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw

from bouncing_ball_task import index
from bouncing_ball_task.constants import CONSTANT_COLOR, DEFAULT_COLORS
from bouncing_ball_task.utils import gif, logutils, pyutils

np.set_printoptions(precision=2, linewidth=150)

class BouncingBallTask:

    valid_output_modes = {
        "parameter",  # Returns tuples of parameters (position, color)
        "array",  # Returns a np array of the image generated
        "parameter_array",  # Returns the parameters as a np array instead of a tuple
        "parameter_array_batch",  # Returns the parameters as a np array instead of a tuple
    }
    valid_color_samplings = {
        "fixed",  # Samples next color in a fixed order
        "random",  # Samples next color randomly with replacement
    }
    valid_return_change_modes = {
        "any",  # Return int for if any change is detected - len 1
        "feature",  # Return int for changes in each feature - len 2
        "source",  # Return int for changes split by the source of the change - len 4
    }
    valid_sequence_modes = {
        "static",  # Sequence is repeated on each iter
        "ic",  # Initial conditions are repeated on each iter
        "reset",  # Fully reset the state on each iter
        "reverse",  # Outputs a static sequence but in reverse
        "preset",  # Outputs a static provided sequence
    }
    valid_color_mask_modes = {
        "outer",  # Applies the mask when the outer circle edge touches the grayzone
        "inner",  # Applies the mask when the inner circle edge touches the grayzone
        "fade",  # Fades the color of the grayzone as a function of diameter inside the grayzone
    }
    valid_transitioning_change_modes = { # Behavior for transitions into the grayzone
        "all", # Random changes are fully allowed as the ball transitions
        "half", # Transition changes only allowed if >half the ball is visible
        None, # Transitions changes entirely not allowed
    }

    def __init__(
        self,
        size_frame: Iterable[int] = (256, 256),
        sequence_length: int = 500,
        ball_radius: int = 10,
        dt: float = 0.01,
        probability_velocity_change: Union[
            float, Callable[[int], np.ndarray]
        ] = 0.001,
        probability_color_change_no_velocity_change: Union[
            float, Callable[[int], np.ndarray]
        ] = 0.01,
        probability_color_change_on_velocity_change: Union[
            float, Callable[[int], np.ndarray]
        ] = 1.0,
        initial_position: Optional[Iterable[float]] = None,
        same_xy_velocity: bool = False,
        batch_size: int = 2,
        batch_first: bool = True,
        initial_velocity: Optional[Iterable[float]] = None,
        velocity_x_lower_multiplier: float = 1 / 12.5,
        velocity_x_upper_multiplier: float = 1 / 7.5,
        velocity_y_lower_multiplier: float = 1 / 12.5,
        velocity_y_upper_multiplier: float = 1 / 7.5,
        sample_velocity_discretely: bool = False,
        num_x_velocities: Optional[int] = 1,
        num_y_velocities: Optional[int] = 2,
        color_sampling: Optional[str] = "fixed",
        probability_initial_colors: Optional[tuple[float]] = None,
        initial_color: Optional[Iterable[int]] = None,
        valid_colors: Optional[Union[str, Iterable[Iterable[int]]]] = "default",
        num_colors: Optional[int] = None,
        mask_center: Optional[float] = 0.5,
        mask_fraction: Optional[float] = 1
        / 3,  # Has to be None for no grayzone
        mask_color: Iterable[int] = (127, 127, 127),
        sample_mode: str = "parameter_array",
        target_mode: str = "parameter_array",
        target_future_timestep: int = 0,
        return_change: bool = False,
        return_change_mode: str = "any",
        sequence_mode: str = "static",
        reset_after_iter: bool = False,
        min_t_color_change: int = 20,
        forced_velocity_bounce_x: Optional[list[int]] = None,
        forced_velocity_bounce_y: Optional[list[int]] = None,
        forced_velocity_resamples: Optional[list[int]] = None,
        forced_color_changes: Optional[list[int]] = None,
        seed: Optional[int] = None,
        initial_velocity_points_away_from_grayzone: bool = True,
        debug: bool = False,
        pccnvc_lower: Optional[float] = None,
        pccnvc_upper: Optional[float] = None,
        pccovc_lower: Optional[float] = None,
        pccovc_upper: Optional[float] = None,
        num_pccnvc: int = None,
        num_pccovc: int = None,
        color_mask_mode: str = "inner",
        initial_timestep_is_changepoint: bool = True,
        color_change_bounce_delay: int = 0,
        color_change_random_delay: int = 0,
        transitioning_change_mode: Optional[str] = None,
        samples=None,
        targets=None,
        *args,
        **kwargs,
    ):
        self._initialized = False
        self.size_frame = self.size_x, self.size_y = np.array(
            size_frame
        )  # Convenience
        self.sequence_length = sequence_length
        self.ball_radius = ball_radius
        self.ball_diameter = 2 * self.ball_radius
        self.dt = dt
        self.min_t_color_change = min_t_color_change
        self._color_mask_mode = None  # Setter initialization
        self.color_mask_mode = color_mask_mode
        self.color_change_bounce_delay = color_change_bounce_delay
        self.color_change_random_delay = color_change_random_delay
        self.color_change_delay = np.array(
            [
                self.color_change_bounce_delay,
                self.color_change_random_delay,
            ]
        ).astype(int)
        self.color_change_target_indices = (
            np.arange(self.sequence_length)[:, None] + self.color_change_delay
        )

        # Which timestep in the future should we return as the target
        self.target_future_timestep = target_future_timestep
        self.task_deque = deque()        

        self.batch_size = batch_size
        self.batch_first = batch_first
        self.transitioning_change_mode = transitioning_change_mode

        self.velocity_x_lower_multiplier = velocity_x_lower_multiplier
        self.velocity_x_upper_multiplier = velocity_x_upper_multiplier
        self.velocity_y_lower_multiplier = velocity_y_lower_multiplier
        self.velocity_y_upper_multiplier = velocity_y_upper_multiplier
        self.initial_velocity_points_away_from_grayzone = (
            initial_velocity_points_away_from_grayzone
        )

        self.sample_velocity_discretely = sample_velocity_discretely
        self.num_x_velocities = num_x_velocities
        self.num_y_velocities = num_y_velocities

        # Initialize attributes to None
        self.probability_velocity_change = None
        self.probability_color_change_no_velocity_change = None
        self.probability_color_change_on_velocity_change = None

        # Store callables internally
        self._set_callables(
            probability_velocity_change,
            probability_color_change_no_velocity_change,
            probability_color_change_on_velocity_change,
        )

        self.pccnvc_lower = pccnvc_lower
        self.pccnvc_upper = pccnvc_upper
        self.pccovc_lower = pccovc_lower
        self.pccovc_upper = pccovc_upper

        self.num_pccnvc = num_pccnvc
        self.num_pccovc = num_pccovc

        self.preset_samples = samples
        self.preset_targets = targets

        self.sequence = []  # Only used in static mode

        # Initialize with None to use the setter in initialization
        self._sequence_mode = None
        self.sequence_mode = sequence_mode  # Use setter during initialization

        if self.preset_samples is not None and self.preset_targets is not None:
            if self.sequence_mode != "preset":
                logger.warning(
                    f"Preset samples and targets provided, but sequence mode is not set to 'preset'; currently set to {self.sequence_mode}"
                )

        # Use the provided batch_size to initialize attributes
        if self.sequence_mode != "preset":
            self.resample_change_probabilities(batch_size)

        self.mask_center = mask_center
        self.mask_fraction = mask_fraction
        self.mask_color = np.array([127, 127, 127], dtype=np.single)

        self.set_mask_parameters()

        self.test_mode = False

        self.initial_position = (
            np.array(initial_position)
            if initial_position is not None
            else self.sample_position()
        )

        self.same_xy_velocity = same_xy_velocity
        if self.same_xy_velocity:
            logger.warning(
                "Setting same_xy_velocity to True has not been thoroughly tested and will likely lead to errors or unexpected behavior."
            )

        self.initial_velocity = (
            np.array(initial_velocity)
            if initial_velocity is not None
            else self.sample_velocity(
                initial_sample=self.initial_velocity_points_away_from_grayzone,
            )
        )

        # Initialize with None to use the setter in initialization
        self._color_sampling = None
        self.color_sampling = color_sampling  # Use setter during initialization

        self.probability_initial_colors = probability_initial_colors

        (
            self.initial_color,
            self.valid_colors,
            self.num_colors,
        ) = self.set_color_parameters(
            initial_color,
            valid_colors,
            num_colors,
        )

        self.sample_mode = sample_mode
        self.target_mode = target_mode

        # For generating arrays
        if self.sample_mode == "array" or self.target_mode == "array":
            self.background = self.build_background(
                self.size_frame,
                self.mask_start,
                self.mask_end,
                self.mask_color,
            )

        self.seed = seed
        if self.seed is not False:  # Skip seeding if its done elsewhere
            pyutils.set_global_seed(self.seed) # Set the seed before all operations

        self.return_change = return_change

        self._return_change_mode = None
        # Use setter during initialization
        self.return_change_mode = return_change_mode

        self.debug = debug
        self.all_parameters = []  # Only in debug mode

        # This clunky way of handling transformations was a result of needing to
        # make the clss picklable when used in a downstream pipeline, meaning
        # lambda and nested funcs had to be avoided
        self.sample_transformation_dict = {
            "parameter": self.sample_parameter_transformation,
            "array": self.sample_array_transformation,
            "parameter_array": self.sample_parameter_array_transformation,
            "parameter_array_batch": self.sample_parameter_array_batch_transformation,
        }

        self.target_transformation_dict = {
            "parameter": self.target_parameter_transformation,
            "array": self.target_array_transformation,
            "parameter_array": self.target_parameter_array_transformation,
            "parameter_array_batch": self.target_parameter_array_batch_transformation,
        }

        # Where to force a velocity x bounce
        self.forced_velocity_bounce_x = (
            []
            if forced_velocity_bounce_x is None
            else list(forced_velocity_bounce_x)
        )

        forced_velocity_bounce_x_array = np.zeros(
            (self.sequence_length, self.batch_size), dtype=bool
        )
        if self.forced_velocity_bounce_x and isinstance(
            self.forced_velocity_bounce_x[0], list
        ):
            for batch_idx, batch in enumerate(self.forced_velocity_bounce_x):
                for time_idx in batch:

                    forced_velocity_bounce_x_array[time_idx, batch_idx] = True
        else:
            forced_velocity_bounce_x_array[self.forced_velocity_bounce_x] = True

        # Forced y bounce
        self.forced_velocity_bounce_y = (
            []
            if forced_velocity_bounce_y is None
            else list(forced_velocity_bounce_y)
        )

        forced_velocity_bounce_y_array = np.zeros(
            (self.sequence_length, self.batch_size), dtype=bool
        )

        if self.forced_velocity_bounce_y and isinstance(
            self.forced_velocity_bounce_y[0], list
        ):
            for batch_idx, batch in enumerate(self.forced_velocity_bounce_y):
                for time_idx in batch:
                    forced_velocity_bounce_y_array[time_idx, batch_idx] = True
        else:
            forced_velocity_bounce_y_array[self.forced_velocity_bounce_y] = True

        # Forced bounce array
        self.forced_velocity_bounce_array = np.stack(
            (forced_velocity_bounce_x_array, forced_velocity_bounce_y_array),
            axis=-1,
        )

        # Where to force a velocity resample
        self.forced_velocity_resamples = (
            []
            if forced_velocity_resamples is None
            else list(forced_velocity_resamples)
        )
        self.forced_velocity_resamples_array = np.zeros(
            (self.sequence_length, self.batch_size), dtype=bool
        )
        self.forced_velocity_resamples_array[
            self.forced_velocity_resamples
        ] = True

        # Where to force a color resample
        self.forced_color_changes = (
            [] if forced_color_changes is None else list(forced_color_changes)
        )
        self.forced_color_changes_array = np.zeros(
            (self.sequence_length, self.batch_size), dtype=bool
        )
        self.forced_color_changes_array[self.forced_color_changes] = True

        self.reset_tracking()

        # # Initialize the color change count to zeros
        # self.color_change_count = np.zeros((self.batch_size, 2))
        # # Initialize the velocity change count to zeros
        # self.velocity_change_count = np.zeros((self.batch_size, 2))

        # Sequence agnostic variables
        self.position_lower_bound = np.array(
            [self.ball_radius, self.ball_radius]
        )
        self.position_upper_bound = np.array(
            [
                self.size_x - self.ball_radius,
                self.size_y - self.ball_radius,
            ]
        )

        # Initial change conditions
        self.initial_timestep_is_changepoint = initial_timestep_is_changepoint
        self.initial_changes = (
            np.ones((self.batch_size, 2)) * self.initial_timestep_is_changepoint
        )

        # Run through the sequence once for specific modes upon initialization
        if self.sequence_mode in {
            "static",
            "reverse",
        }:
            _ = [_ for _ in self]  # Run through the sequence once

        self._initialized = True

    @property
    def color_sampling(self) -> str:
        return self._color_sampling

    @color_sampling.setter
    def color_sampling(self, value: str):
        logger.debug("Running color_sampling setter")
        value = value.lower()  # Convert input to lowercase
        if value not in self.valid_color_samplings:
            raise ValueError(
                f"Invalid color_sampling: {value}. Must be one of {self.valid_color_samplings}."
            )

        # Adjust color_sampling based on batch_size if necessary
        if value == "fixed" and self.batch_size > 1:
            value = "fixed_vectorized"
        elif value == "random" and self.batch_size > 1:
            raise NotImplementedError(
                "Vectorized random color sampling isn't implemented"
            )

        self._color_sampling = value

    @property
    def sequence_mode(self) -> str:
        return self._sequence_mode

    @sequence_mode.setter
    def sequence_mode(self, mode: str):
        logger.debug("Running sequence_mode setter")
        mode = mode.lower()  # Convert input to lowercase
        if mode not in self.valid_sequence_modes:
            raise ValueError(
                f"Invalid sequence_mode: {mode}. Must be one of {self.valid_sequence_modes}."
            )
        self._sequence_mode = mode

        if self.sequence_mode == "preset":
            assert self.preset_samples is not None
            assert self.preset_targets is not None
            self.sequence = list(zip(
                self.preset_samples.transpose(1, 0, 2),
                self.preset_targets.transpose(1, 0, 2),
            ))
            self.sequence_length = len(self.sequence) + self.target_future_timestep

        # Reset all the relevant downstream parameters and run through the
        # sequence once for specific modes but dont run it when the obj is
        # first being initialized
        elif self.sequence_mode in {"static", "reverse"}:
            if self._initialized:
                # Depends on the sequence mode, so resample them
                self.resample_change_probabilities(self.batch_size)

                # Run through the sequence once
                _ = [_ for _ in self]

    @property
    def return_change_mode(self) -> str:
        return self._return_change_mode

    @return_change_mode.setter
    def return_change_mode(self, mode: str):
        logger.debug("Running return_change_mode setter")
        mode = mode.lower()  # Convert input to lowercase
        if mode not in self.valid_return_change_modes:
            raise ValueError(
                f"Invalid return_change_mode: {mode}. Must be one of {self.valid_return_change_modes}."
            )
        self._return_change_mode = mode

    @property
    def transitioning_change_mode(self) -> str:
        return self._transitioning_change_mode

    @return_change_mode.setter
    def transitioning_change_mode(self, mode: Optional[str]):
        logger.debug("Running transitioning_change_mode setter")
        if mode is not None:
            mode = mode.lower()  # Convert input to lowercase
        if mode not in self.valid_transitioning_change_modes:
            raise ValueError(
                f"Invalid transitioning_change_mode: {mode}. Must be one of "
                f"{self.valid_transitioning_change_modes}."
            )
        self._transitioning_change_mode = mode
        if mode == "all":
            self.transition_value = np.Inf
        elif mode == "half":
            self.transition_value = self.ball_radius
        elif mode == None:
            self.transition_value = 0

    @property
    def color_mask_mode(self) -> str:
        return self._color_mask_mode

    @color_mask_mode.setter
    def color_mask_mode(self, mode: str):
        logger.debug("Running color_mask_mode setter")
        mode = mode.lower()  # Convert input to lowercase
        if mode not in self.valid_color_mask_modes:
            raise ValueError(
                f"Invalid color_mask_mode: {mode}. Must be one of {self.valid_color_mask_modes}."
            )
        self._color_mask_mode = mode

    def _set_callables(
        self,
        probability_velocity_change,
        probability_color_change_no_velocity_change,
        probability_color_change_on_velocity_change,
    ):
        """Store the callable or float values."""
        self._callable_probability_velocity_change = probability_velocity_change
        self._callable_probability_color_change_no_velocity_change = (
            probability_color_change_no_velocity_change
        )
        self._callable_probability_color_change_on_velocity_change = (
            probability_color_change_on_velocity_change
        )

    def resample_change_probabilities(self, batch_size: int):
        """Update attribute values based on callables or set them if they are
        floats."""
        logger.debug("Running resample_change_probabilities")
        # Process probability_velocity_change
        if callable(self._callable_probability_velocity_change):
            self.probability_velocity_change = (
                self._callable_probability_velocity_change(batch_size)
            )
        else:
            self.probability_velocity_change = (
                self._callable_probability_velocity_change
            )

        # Process probability_color_change_no_velocity_change
        if callable(self._callable_probability_color_change_no_velocity_change):
            self.probability_color_change_no_velocity_change = (
                self._callable_probability_color_change_no_velocity_change(
                    batch_size
                )
            )
        elif self.pccnvc_lower is not None and self.pccnvc_upper is not None:
            # Warning: untested and likely has bugs
            (
                pccnvc,
                self.pccnvc_bin_indices,
                self.pccnvc_bins,
            ) = self.sample_from_range(
                batch_size,
                self.pccnvc_lower,
                self.pccnvc_upper,
                mode="repeat",
                sequence_mode=self.sequence_mode,
                num_bins=self.num_pccovc,
            )
            self.probability_color_change_no_velocity_change = pccnvc
        else:
            self.probability_color_change_no_velocity_change = (
                self._callable_probability_color_change_no_velocity_change
            )

        # Process probability_color_change_on_velocity_change
        if callable(self._callable_probability_color_change_on_velocity_change):
            self.probability_color_change_on_velocity_change = (
                self._callable_probability_color_change_on_velocity_change(
                    batch_size
                )
            )
        elif self.pccovc_lower is not None and self.pccovc_upper is not None:
            # Warning: untested and likely has bugs
            (
                pccovc,
                self.pccovc_bin_indices,
                self.pccovc_bins,
            ) = self.sample_from_range(
                batch_size,
                self.pccovc_lower,
                self.pccovc_upper,
                mode="tile",
                sequence_mode=self.sequence_mode,
                num_bins=self.num_pccovc,
            )
            self.probability_color_change_on_velocity_change = pccovc
        else:
            self.probability_color_change_on_velocity_change = (
                self._callable_probability_color_change_on_velocity_change
            )

        # Update color related probabilities
        self.probability_color_change_given_velocity = np.array(
            [
                self.probability_color_change_on_velocity_change,
                self.probability_color_change_no_velocity_change,
            ]
        )
        if len(self.probability_color_change_given_velocity.shape) > 1:
            self.probability_color_change_given_velocity = (
                self.probability_color_change_given_velocity.T
            )

    def sample_from_range(
        self,
        batch_size,
        value_min,
        value_max,
        mode="repeat",
        sequence_mode="reset",
        num_bins=None,
    ):
        """Extends the functionality to create a [num_bins x num_bins] matrix by
        either uniformly sampling vectors of length num_bins from each bin's
        range or using linspace. Returns a vector indicating which bin each
        element in the flattened vector corresponds to, the method to flatten
        the matrix, and the bins used for sampling.

        Parameters
        ----------
        batch_size : int
            The desired total number of elements (approximate) in the final
                flattened vector.

        value_min : float
            The minimum value of the range to sample from or start of linspace.

        value_max : float
            The maximum value of the range to sample from or end of linspace.

        mode : str
            The method to flatten the matrix ('repeat' for direct flatten or
                'tile' for transpose then flatten).

        sequence_mode : str
            The mode to fill the matrix ('reset' for uniform sampling, 'static'
                for linspace).

        Returns
        -------
        np.ndarray
            The final flattened vector.

        np.ndarray
            A vector indicating the bin index for each element in the flattened
                vector.

        np.ndarray
            The bins used for sampling or linspace.
        """
        # Determine the number of bins
        if num_bins is None:
            num_bins = int(np.floor(np.sqrt(batch_size)))

        # Create bins for uniform sampling or linspace
        bins = np.linspace(value_min, value_max, num_bins + 1)

        if sequence_mode == "reset":
            # Uniformly sample values within each bin to create a matrix
            sampled_matrix = np.array(
                [
                    np.random.uniform(bins[i], bins[i + 1], num_bins)
                    for i in range(num_bins)
                ]
            )
        elif sequence_mode in "static":
            # Use linspace to fill the matrix with static sequences
            sampled_matrix = np.array(
                [
                    np.linspace(bins[i], bins[i + 1], num_bins, endpoint=False)
                    for i in range(num_bins)
                ]
            )
        else:
            raise ValueError("sequence_mode must be 'reset' or 'static'")

        # Create a matrix of bin indices
        bin_indices_matrix = np.array(
            [[i for _ in range(num_bins)] for i in range(num_bins)]
        )

        # Flatten the matrices based on the mode
        if mode == "repeat":
            flattened_vector = sampled_matrix.flatten()
            flattened_bin_indices = bin_indices_matrix.flatten()
        elif mode == "tile":
            flattened_vector = sampled_matrix.T.flatten()
            flattened_bin_indices = bin_indices_matrix.T.flatten()
        else:
            raise ValueError("Mode must be 'repeat' or 'tile'")

        return flattened_vector, flattened_bin_indices, bins

    def sample_parameter_transformation(self, position, masked_color, color):
        return position, masked_color

    def sample_array_transformation(self, position, masked_color, color):
        return self.array_transformation(position, masked_color)

    def sample_parameter_array_transformation(
        self, position, masked_color, color
    ):
        return np.concatenate([position, masked_color], axis=1, dtype=np.single)

    def sample_parameter_array_batch_transformation(
        self, position, masked_color, color
    ):
        return np.concatenate([position, masked_color], axis=1, dtype=np.single)

    def target_parameter_transformation(
        self, position, masked_color, color, velocity_change, color_change
    ):
        return color

    def target_array_transformation(
        self, position, masked_color, color, velocity_change, color_change
    ):
        return self.array_transformation(position, color)

    def target_parameter_array_transformation(
        self, position, masked_color, color, velocity_change, color_change
    ):
        return np.concatenate(
            [position, color]
            + self.get_change_arrays(velocity_change, color_change),
            axis=1,
            dtype=np.single,
        )

    def target_parameter_array_batch_transformation(
        self, position, masked_color, color, velocity_change, color_change
    ):
        return np.concatenate(
            [position, color]
            + self.get_change_arrays(velocity_change, color_change),
            axis=1,
            dtype=np.single,
        )

    def get_change_arrays(self, velocity_change, color_change):
        if not self.return_change:
            return []

        if self.return_change_mode == "source":
            return [velocity_change, color_change]

        any_velocity_change = velocity_change.any(axis=-1, keepdims=True)
        any_color_change = color_change.any(axis=-1, keepdims=True)

        if self.return_change_mode == "feature":
            return [any_velocity_change, any_color_change]

        elif self.return_change_mode == "any":
            return [np.logical_or(any_velocity_change, any_color_change)]

    @property
    def sample_mode(self) -> str:
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str):
        mode = mode.lower()  # Convert input to lowercase
        if not hasattr(self, "_sample_mode") or mode != self._sample_mode:
            if mode not in self.valid_output_modes:
                raise ValueError(
                    f"Variable 'sample_mode' must be one of {self.valid_output_modes}"
                )
            self._sample_mode = mode

    @property
    def target_mode(self) -> str:
        return self._target_mode

    @target_mode.setter
    def target_mode(self, mode: str):
        mode = mode.lower()  # Convert input to lowercase
        if not hasattr(self, "_target_mode") or mode != self._target_mode:
            if mode not in self.valid_output_modes:
                raise ValueError(
                    f"Variable 'target_mode' must be one of {self.valid_output_modes}"
                )
            self._target_mode = mode

    def sample_position(self) -> np.ndarray:
        """Sample positions of a ball in the arena avoiding gray zone.

        The ball is sampled from either side of the gray zone, ensuring that it does
        not overlap with the gray zone boundaries by a margin of 1.5 times the
        ball's radius.

        Returns
        -------
        np.ndarray
            A 2D array of shape (batch_size, 2) where each row represents the
            (x, y) coordinates of a sampled ball position.
        """
        # Sample x-positions:
        # For each sample in the batch, we randomly choose a position from
        # either the left or the right of the gray zone, ensuring it's distant
        # from the gray zone boundaries.
        position_x = np.array(
            [
                np.random.choice(
                    [
                        # Sample from the left side of the gray zone
                        np.random.uniform(
                            self.ball_radius,
                            self.mask_start - 1.5 * self.ball_radius,
                        ),
                        # Sample from the right side of the gray zone
                        np.random.uniform(
                            self.mask_end + 1.5 * self.ball_radius,
                            self.size_x - self.ball_radius,
                        ),
                    ]
                )
                for i in range(self.batch_size)
            ]
        )

        # Sample y-positions ensuring the ball stays within the vertical
        # boundaries of the arena
        position_y = np.random.uniform(
            self.ball_radius,
            self.size_y - self.ball_radius,
            size=self.batch_size,
        )

        # Stack the x and y positions to get a 2D array of coordinates
        return np.stack([position_x, position_y], axis=-1)

    def sample_velocity(
        self,
        timesteps: int = 1,
        batch_size: Optional[int] = None,
        initial_sample: bool = False,
    ) -> np.ndarray:
        """Samples velocities for the bouncing balls where the magnitude scales
        according to the size of the stage. Velocity signs are determined
        randomly or based on the initial condition to ensure visibility before
        entering a masked area (gray zone).

        Parameters
        ----------
        timesteps : int, optional
            The number of timesteps for which velocities need to be sampled.
            Defaults to 1, which means velocities are sampled only for the
            initial timestep.

        batch_size : int, optional
            The number of velocities to sample. Defaults to None, which will use
            the class's batch_size attribute.

        initial_sample : bool, optional
            If True, sample velocities ensuring they initially point away from
            the center, useful for initial state setups. Defaults to False.

        Returns
        -------
        np.ndarray
            A numpy array of sampled velocities. The shape of the array is
            (timesteps, batch_size, 2) or (batch_size, 2) depending on
            `timesteps` and whether velocities for x and y are sampled to be the
            same.
        """
        # Use the class's batch_size if none is specified
        batch_size = self.batch_size if batch_size is None else batch_size

        # Determine number of velocity components based on whether x and y
        # velocities should be the same
        num_v = 1 if self.same_xy_velocity else 2

        # Generate signs as 1 or -1. For initial samples, point away from the
        # center.
        if not initial_sample:
            sign = np.sign(
                np.random.randn(batch_size, num_v)
                if timesteps <= 1
                else np.random.randn(timesteps, batch_size, num_v),
            )
        else:
            # Set the sign to always point towards a wall
            sign = np.sign(self.initial_position[:, 0] - self.size_x / 2)
            if num_v == 2:
                sign = np.stack(
                    [sign, np.sign(np.random.randn(batch_size))], -1
                )
            if timesteps > 1:
                sign = np.stack(
                    [
                        sign,
                        np.sign(
                            np.random.randn(timesteps - 1, batch_size, num_v)
                        ),
                    ]
                )

        # Velocity magnitude bounds based on stage size
        self.velocity_lower_bound = np.array(
            [
                self.size_x * self.velocity_x_lower_multiplier,
                self.size_y * self.velocity_y_lower_multiplier,
            ]
        )[:num_v]
        self.velocity_upper_bound = np.array(
            [
                self.size_x * self.velocity_x_upper_multiplier,
                self.size_y * self.velocity_y_upper_multiplier,
            ]
        )[:num_v]

        # Discrete velocity sampling using linspace if enabled
        if self.sample_velocity_discretely:
            mean_vel = np.mean(
                (self.velocity_lower_bound, self.velocity_upper_bound), axis=0
            )

            vel_x_linspace = np.linspace(
                self.velocity_lower_bound[0]
                if self.num_x_velocities != 1
                else mean_vel[0],
                self.velocity_upper_bound[0]
                if self.num_x_velocities != 1
                else mean_vel[0],
                self.num_x_velocities,
                endpoint=True,
            )

            if not self.same_xy_velocity:
                vel_y_linspace = np.linspace(
                    self.velocity_lower_bound[1]
                    if self.num_y_velocities != 1
                    else mean_vel[1],
                    self.velocity_upper_bound[1]
                    if self.num_y_velocities != 1
                    else mean_vel[1],
                    self.num_y_velocities,
                    endpoint=True,
                )
                vel_linspaces = (vel_x_linspace, vel_y_linspace)
            else:
                vel_linspaces = (vel_x_linspace,)

            if timesteps > 1:
                vel_magnitude = np.stack(
                    [
                        np.random.choice(
                            vel_linspace,
                            replace=True,
                            size=(timesteps, batch_size),
                        )
                        for vel_linspace in vel_linspaces
                    ],
                    axis=-1,
                )[..., :num_v]
            else:
                # Force all discrete values to appear as often as possible
                vel_magnitude = np.stack(
                    [
                        pyutils.repeat_sequence(
                            vel_linspace,
                            batch_size,
                        )
                        for vel_linspace in vel_linspaces
                    ],
                    axis=-1,
                )[..., :num_v]

        # Continuously sample velocity magnitudes within specified bounds
        else:
            vel_magnitude = np.random.uniform(
                self.velocity_lower_bound,
                self.velocity_upper_bound,
                size=(batch_size, num_v)
                if timesteps <= 1
                else (timesteps, batch_size, num_v),
            )

        # Combine signs and magnitudes
        vel = sign * vel_magnitude

        # If x and y velocities should be the same, repeat the values along axis 1
        if self.same_xy_velocity:
            vel = np.repeat(vel, 2, axis=1)

        return vel

    def set_color_parameters(self, initial_color, valid_colors, num_colors):
        if isinstance(valid_colors, str):
            if valid_colors.lower() == "default":
                valid_colors = list(DEFAULT_COLORS)
            elif valid_colors.lower() == "constant":
                valid_colors = list(CONSTANT_COLOR)
            else:
                raise ValueError(f"Invalid str color input, '{valid_color}'")

        # Go off passed valid colors
        if valid_colors is not None:
            if initial_color is not None and len(initial_color) == 3:
                if np.array(initial_color).ndim > 1:
                    for color in initial_color:
                        if (
                            not (initial_color[0] == valid_colors)
                            .all(axis=1)
                            .any()
                        ):
                            valid_colors = [color] + valid_colors

                elif initial_color not in valid_colors:
                    valid_colors = [initial_color] + valid_colors

            num_colors = len(valid_colors)

        # Generate colors randomly
        else:
            if num_colors is None:
                num_colors = 3
            # Future: For 3 colors, generate equidistant colors
            valid_colors = [
                np.random.randint(0, 256, 3) for _ in range(num_colors)
            ]
            if initial_color is not None:
                valid_colors[0] = initial_color

        valid_colors = np.array(valid_colors)

        if self.sequence_mode == "reverse":
            valid_colors = valid_colors[::-1]

        if initial_color is None:
            self._index = np.random.choice(
                num_colors,
                p=self.probability_initial_colors,
                size=self.batch_size,
            )
            initial_color = valid_colors[self._index]
        else:
            initial_color = np.array(initial_color)
            self._index = np.zeros((initial_color.shape[0]), dtype=int)
            for i, valid_color in enumerate(valid_colors):
                self._index[np.all(initial_color == valid_color, axis=1)] = i

        return initial_color, valid_colors, num_colors

    def color_sampler(self):

        yield self.initial_color

        if self.color_sampling == "random":
            next_index = self.color_sampler_random

        elif self.color_sampling == "fixed":
            next_index = self.color_sampler_fixed

        elif self.color_sampling == "fixed_vectorized":
            next_index = self.color_sampler_fixed_vectorized

        while True:
            self._index = next_index()
            yield self.valid_colors[self._index]

    def color_sampler_random(self):
        return np.random.choice(self.num_colors, size=self.batch_size)

    def color_sampler_fixed(self):
        return (self._index + 1) % self.num_colors

    def color_sampler_fixed_vectorized(self):
        return (self._index + self.color_change_indices) % self.num_colors

    def set_mask_parameters(self):
        if self.mask_fraction is None:
            self.mask_size = 0
            self.mask_start = 0
            self.mask_end = 0

        else:
            self.mask_center = self.mask_center or 0.5
            self.mask_size = int(np.round(self.size_x * self.mask_fraction))
            self.mask_start = int(
                np.round((self.mask_center * self.size_x) - self.mask_size / 2)
            )
            self.mask_end = int(self.size_x - self.mask_start)

    def color_mask(self, positions, colors):
        """Apply a color mask according to the current inputted positions.

        If `color_mask_mode` is set to "fade", the mask is applied with a fading
        effect based on the proportion of overlap with the grayzone. If "outer",
        the mask is applied the moment the outer edge of the ball overlaps with
        the grayzone. If set to "inner" then the mask is applied when the inner
        edge of the ball overlaps with the grayzone.

        Parameters
        ----------
        positions : numpy.ndarray
            Array of positions with shape (B, 2), where B is the batch size and
                second dim corresponds to (x, y) coordinates.

        colors : numpy.ndarray
            Array of colors with shape (B, 3), where B is the batch size and
                second dim correspnds to the RGB color.

        Returns
        -------
        masked_colors : numpy.ndarray
            Array of masked colors with the same shape as the input colors.
        """
        x_positions = positions[:, 0]

        if self.color_mask_mode == "fade":
            # Compute overlaps with either side of the grayzone
            overlap_left = np.maximum(
                x_positions - self.ball_radius, self.mask_start
            )
            overlap_right = np.minimum(
                x_positions + self.ball_radius, self.mask_end
            )
            overlap_width = np.clip(
                overlap_right - overlap_left, 0, None
            )  # Replace np.maximum with np.clip

            # Directly compute the masked color using broadcasting
            overlap_proportion = overlap_width / self.ball_diameter
            return np.where(
                overlap_width[:, None]
                > 0,  # Use np.where to blend only where needed
                np.round(
                    overlap_proportion[:, None] * self.mask_color
                    + (1 - overlap_proportion[:, None]) * colors
                ).astype(int),
                colors,
            )

        elif self.color_mask_mode == "outer":
            # Find mask locations within the specified range
            mask_locations = np.logical_and(
                self.mask_start - self.ball_radius <= x_positions,
                x_positions <= self.mask_end + self.ball_radius,
            )
            # Apply the mask color to the selected locations
            masked_colors = np.copy(colors)
            masked_colors[mask_locations] = self.mask_color
            return masked_colors

        elif self.color_mask_mode == "inner":
            # Find mask locations within the specified range
            mask_locations = np.logical_and(
                self.mask_start + self.ball_radius <= x_positions,
                x_positions <= self.mask_end - self.ball_radius,
            )
            # Apply the mask color to the selected locations
            masked_colors = np.copy(colors)
            masked_colors[mask_locations] = self.mask_color
            return masked_colors

        else:
            raise ValueError(
                f"color_mask_mode must be one of '{self.valid_color_mask_modes}'"
                f"but is set to '{self.color_mask_mode}'"
            )

    def __len__(self):
        return self.sequence_length - self.target_future_timestep

    def track_parameters(
        self,
        t,
        position,
        velocity,
        masked_color,
        color,
        velocity_change,
        color_change,
    ):
        self.all_parameters.append(
            {
                "time": t,
                "position": position,
                "velocity": velocity,
                "color_masked": masked_color,
                "color": color,
                "velocity change": velocity_change,
                "color_change": color_change,
            }
        )
        # Increment the change counters
        self.color_change_count += color_change.astype(int)
        self.velocity_change_count += velocity_change.astype(int)

    def output_transformation(self, sample_parameters, target_parameters):
        # (position, velocity, masked_color, color, velocity_change, color_change)
        position, _, masked_color, color, _, _ = sample_parameters
        sample_out = self.sample_transformation_dict[self.sample_mode](
            position, masked_color, color
        )

        (
            position,
            _,
            masked_color,
            color,
            velocity_change,
            color_change,
        ) = target_parameters
        target_out = self.target_transformation_dict[self.target_mode](
            position,
            masked_color,
            color,
            velocity_change,
            color_change,
        )
        self.sequence.append((sample_out, target_out))
        return sample_out, target_out

    def array_transformation(self, position, color, thickness=-1):
        return gif.draw_circle(
            position,
            color,
            self.background,
            self.ball_radius,
            self.mask_color,
            thickness=-1,
        )

    def reset_tracking(self):
        self.velocity_changes = []
        self.color_changes = []
        self.all_parameters = []

        if self.debug:
            # Reset the color change count to zeros
            self.color_change_count = np.zeros((self.batch_size, 2))
            # Reset the velocity change count to zeros
            self.velocity_change_count = np.zeros((self.batch_size, 2))

    def bouncing_ball_sequence(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        color_sequence: Iterator[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a sequence of bouncing ball positions, velocities, and
        colors.

        Simulates the trajectory of bouncing ball in the arena taking into
        account the ball's position, velocity, and color changes. The velocity
        and color can change either due to interactions with the boundaries or
        probabilistically over time.

        Parameters
        ----------
        position : np.ndarray
            Initial position of the ball

        velocity : np.ndarray
            Initial velocity of the ball

        color_sequence : Iterator[np.ndarray]
            An iterator yielding color values for each time step

        Yields
        ------
        position : np.ndarray
            Current position of the ball

        velocity : np.ndarray
            Current velocity of the ball

        color : np.ndarray
            Current color of the ball

        velocity_changes_combined : np.ndarray
            Boolean array indicating sequences with velocity changes due to a
                bounce or random velocity resamplings

        color_changes_combined : np.ndarray
            Boolean array indicating sequences with color changes due to a
                velocity change or a random color change
        """
        color = next(color_sequence)

        # Pre-allocate arrays for random samples to reduce function calls
        rand_for_velocity = np.random.uniform(
            size=(self.sequence_length, self.batch_size)
        )
        rand_for_velocity_resample = self.sample_velocity(
            self.sequence_length, self.batch_size
        )
        rand_for_color = np.random.uniform(
            size=(self.sequence_length, self.batch_size, 2)
        )

        # Color change array to hold changes
        color_change_array = np.zeros(
            (
                self.sequence_length + self.color_change_delay.max(),
                self.batch_size,
                2,
            ),
            dtype=bool,
        )

        # Send out the initial conditions
        yield (
            position.copy(),
            velocity.copy(),
            color,
            self.initial_changes,
            self.initial_changes,
        )

        for t in range(1, self.sequence_length):
            # Check for sequences where the position is OOB
            positions_out_of_bounds = np.logical_or(
                position > self.position_upper_bound,
                position < self.position_lower_bound,
            )

            # Get all bounce indices including forced bounces
            indices_positions_with_bounce = np.logical_or(
                positions_out_of_bounds,
                self.forced_velocity_bounce_array[t],
            )

            # Update the velocity component that brings the ball in bounds
            velocity[indices_positions_with_bounce] *= -1

            # Which sequences had a bounce velocity change in either component
            indices_velocity_bounce = np.logical_or(
                *indices_positions_with_bounce.T
            )

            # Which ones did not
            indices_velocity_no_bounce = np.logical_not(indices_velocity_bounce)

            # Of those with no bounce, probabilistically resample velocity or
            # force a resample if it is a forced resampling index
            indices_velocity_resamples = np.logical_and(
                indices_velocity_no_bounce,
                np.logical_or(
                    rand_for_velocity[t] <= self.probability_velocity_change,
                    self.forced_velocity_resamples_array[t],
                ),
            )

            # Apply a velocity change to the indices that are resampled
            velocity[indices_velocity_resamples] = rand_for_velocity_resample[
                t, indices_velocity_resamples
            ]

            # Combine the velocity changes into one array
            velocity_changes_combined = np.stack(
                [indices_velocity_bounce, indices_velocity_resamples],
                axis=-1,
            )

            # Compile all bounce and resampled velocity changes into one vector
            velocity_changes = velocity_changes_combined.any(axis=-1)

            # Vector of sequences without a velocity change of either kind
            no_velocity_changes = np.logical_not(velocity_changes)
            
            # Compute the combined indices where the color will change according
            # to whether the velocity changed or not
            color_changes_combined = np.logical_and(
                np.stack([velocity_changes, no_velocity_changes], axis=-1),
                rand_for_color[t]
                <= self.probability_color_change_given_velocity,
            )

            # Apply color changes to the timestep that gets affected by
            # statistics at current timestep. No color_change_delay means it
            # affects current timestep, otherwise it will change a future color
            color_change_array[
                self.color_change_target_indices[t], :, [0, 1]
            ] = color_changes_combined.T

            # Find indices where the ball is gray-transitioning
            transitioning_overlap = np.clip(
                np.minimum(
                    position[:, 0] + self.ball_radius,
                    self.mask_end,
                ) -
                np.maximum(
                    position[:, 0] - self.ball_radius,
                    self.mask_start,
                ),
                0,
                None,
            )
            transitioning_mask = np.logical_and(
                transitioning_overlap > self.transition_value,
                transitioning_overlap < self.ball_radius * 2,
            )

            # Impose color changes cannot happen when transitioning into and out
            # of the grayzone
            color_change_array[t, :, 1] = np.logical_and(
                color_change_array[t, :, 1],
                ~transitioning_mask,
            )

            # Select indices for where color will change
            self.color_change_indices = color_changes = np.logical_or(
                color_change_array[t].any(axis=-1),
                self.forced_color_changes_array[t],
            )
            
            # Set chance for random (no vel change) color changes to be 0 for
            # sequences where the color changed for min_t_color_change time steps
            # or that are transitioning into or out of the grayzone
            rand_for_color[
                t + 1 : t + self.min_t_color_change + 1, color_changes, 1
            ] = 1.0

            # Step the color forward in time
            color = next(color_sequence)

            # Step the position forward in time
            position += velocity * self.dt

            yield (
                position.copy(),
                velocity.copy(),
                color,
                velocity_changes_combined,
                color_change_array[t],
            )

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over the bouncing ball sequences based on the sequence mode.

        This generator yields sequences of bouncing ball (sample, target) pairs.
        The generated sequences can either be static (unchanging across
        iterations), or they can be regenerated based on the "reset" mode. The
        ball's sample color is masked according to whether it's in th gray zone
        or not.

        Yields
        ------
        tuple[sample : np.ndarray, target : np.ndarray]:
            The tuple containing the sample and target for every timestep. The
            sample has components of [x_pos, y_pos, masked_c1, masked_c2,
            masked_c3], where `x_pos` and `y_pos` are the positions of the ball
            on the arena, and `masked_c{n}` are the different color channels,
            which can be masked if the  ball is in the gray zone. The target is
            the same but the color channels always have the true color, and
            optionally return the different change conditions.
        """
        # Check if the mode is static and a pre-defined sequence exists
        if self.sequence_mode in {"static", "reverse", "preset"} and len(
            self.sequence
        ) == len(self):
            yield from self.sequence
            return

        # For reset mode, re-sample initial conditions
        elif self.sequence_mode == "reset":
            self.initial_position = self.sample_position()
            self.initial_velocity = self.sample_velocity(
                initial_sample=self.initial_velocity_points_away_from_grayzone
            )
            self.initial_color, _, _ = self.set_color_parameters(
                None,
                self.valid_colors,
                None,
            )
            self.resample_change_probabilities(self.batch_size)

        # Reset state tracking for the new sequence
        self.reset_tracking()
        self.task_deque = deque()
        self.sequence = []

        # Define the iterator
        ball_sequence = self.bouncing_ball_sequence(
            self.initial_position.copy(),
            self.initial_velocity.copy(),
            self.color_sampler(),
        )

        # Reverse it for that sequence mode
        if self.sequence_mode == "reverse":
            ball_sequence = self.reverse_ball_sequence(ball_sequence)

        # Iterate over the generated bouncing ball sequence
        for t, (
            position,
            velocity,
            color,
            velocity_change,
            color_change,
        ) in enumerate(ball_sequence):
            # Apply the mask to the color based on the ball's position
            masked_color = self.color_mask(position, color)
            parameters = (
                position,
                velocity,
                masked_color,
                color,
                velocity_change,
                color_change,
            )

            # Store the current parameters for future reference
            if self.debug:
                self.track_parameters(t, *parameters)

            # If we have not reached the target timestep, continue iterating
            # until we do so before starting to yield (sample, target) pairs
            if t <= self.target_future_timestep:
                self.task_deque.append(parameters)
                continue

            # Yield the transformed output for the current timestep
            yield self.output_transformation(
                self.task_deque[0],
                self.task_deque[-1],
            )

            # Rotate the deque and update its last element
            self.task_deque.rotate(-1)
            self.task_deque[-1] = parameters

        # Yield the transformed output for the last timestep
        yield self.output_transformation(
            self.task_deque[0], self.task_deque[-1]
        )

    def reverse_ball_sequence(self, ball_sequence):
        ball_sequence = list(reversed(list(ball_sequence)))

        # Unpack the sequence (each is a tuple now)
        pos, vel, col, vel_ch, col_ch = zip(*ball_sequence)

        # For color, we want the first element repeated at the start and then
        # all but the last element.
        col_new = (col[0],) + col[:-1]

        # Precompute a zeros array for the change dims.
        zero_vel = np.zeros_like(vel_ch[-1])
        zero_col = np.zeros_like(col_ch[-1])

        # For the velocity and color change arrays, we want:
        # - first element: last element from the original,
        # - second element: zeros,
        # - then all but the last two elements.
        vel_ch_new = (vel_ch[-1], zero_vel) + vel_ch[:-2]
        col_ch_new = (col_ch[-1], zero_col) + col_ch[:-2]

        # Return a zipped iterator of the new tuples.
        return zip(pos, vel, col_new, vel_ch_new, col_ch_new)        
            
    @classmethod
    def target_to_sample(
        cls, target, mask_start, mask_end, mask_color, ball_radius
    ):
        x_positions = target[:, 0]
        colors = target[:, 2:]
        ball_diameter = 2 * ball_radius

        # Compute overlaps directly without intermediate array storage
        overlap_left = np.maximum(x_positions - ball_radius, mask_start)
        overlap_right = np.minimum(x_positions + ball_radius, mask_end)
        overlap_width = np.clip(
            overlap_right - overlap_left, 0, None
        )  # Replace np.maximum with np.clip

        # Directly compute the masked color using broadcasting
        overlap_proportion = overlap_width / ball_diameter
        return np.concatenate(
            [
                target[:, :2],
                np.where(
                    overlap_width[:, None]
                    > 0,  # Use np.where to blend only where needed
                    np.round(
                        overlap_proportion[:, None] * mask_color
                        + (1 - overlap_proportion[:, None]) * colors
                    ).astype(int),
                    colors,
                ),
            ],
            axis=1,
        )

    @classmethod
    def build_background(cls, size_frame, mask_start, mask_end, mask_color):
        background = np.zeros((*size_frame[::-1], 3), dtype=np.uint8)
        background[:, mask_start:mask_end, :] = mask_color
        return background

    @classmethod
    def parameter_to_image(
        cls,
        sample_parameters: tuple[torch.Tensor],
        size_frame,
        mask_start,
        mask_end,
        mask_color,
        ball_radius,
        target_parameters: Optional[tuple[torch.Tensor]] = None,
        background: Optional[np.ndarray] = None,
        mode: str = "original",
        multiplier: int = 1,
        sample_thickness: int = 2,
        return_mode: str = "image",
    ):
        valid_modes = {
            "original",  # Creates image arrays separately
            "concat",  # Concatenates sample and target. Requires target
            "combined",  # Combines sample and target. Requires target
        }
        valid_return_modes = {
            "image",  # PIL Image
            "array",  # np arrays
        }

        if (mode := mode.lower()) not in valid_modes:
            raise ValueError(f"Argument 'mode' must be one of {valid_modes}")
        if mode in {"concat", "combined"} and target_parameters is None:
            raise ValueError(
                f"Mode cannot be {mode} if no target parameters were passed."
            )
        if (return_mode := return_mode.lower()) not in valid_return_modes:
            raise ValueError(
                f"Argument 'return_mode' must be one of {valid_return_modes}"
            )

        size_frame = [size * multiplier for size in size_frame]
        mask_start *= multiplier
        mask_end *= multiplier
        ball_radius *= multiplier
        if mode == "combined":
            sample_thickness *= int(multiplier)
        elif mode == "original" and target_parameters is not None:
            sample_thickness *= int(multiplier)
            # sample_thickness = -1 / sample_thickness

        if background is None:
            background = cls.build_background(
                size_frame, mask_start, mask_end, mask_color
            )

        sample_arrays = [
            gif.draw_frame(
                (param[:2] * multiplier).tolist(),
                param[2:5].tolist(),
                ball_radius,
                mask_color.tolist(),
                size_frame,
                mask_start,
                mask_end,
                circle_border_thickness=sample_thickness,
            )
            for param in sample_parameters
        ]

        if target_parameters is not None:
            if mode == "combined":
                target_arrays = [
                    gif.draw_circle(
                        (param[:2] * multiplier).tolist(),
                        param[2:].tolist(),
                        sample,
                        ball_radius,
                        mask_color.tolist(),
                    )
                    for param, sample in zip(target_parameters, sample_arrays)
                ]

            else:
                target_arrays = [
                    gif.draw_circle(
                        (param[:2] * multiplier).tolist(),
                        param[2:].tolist(),
                        background,
                        ball_radius,
                        mask_color.tolist(),
                    )
                    for param in target_parameters
                ]

        if mode == "concat":
            output = [
                np.concatenate((sample, target), 1)
                for sample, target in zip(sample_arrays, target_arrays)
            ]

            if return_mode == "array":
                return output
            elif return_mode == "image":
                return [Image.fromarray(out) for out in output]

        elif mode == "original":
            if return_mode == "array":
                if target_parameters is None:
                    return sample_arrays
                return sample_arrays, target_arrays

            elif return_mode == "image":
                sample_images = [
                    Image.fromarray(array) for array in sample_arrays
                ]

                if target_parameters is None:
                    return sample_images

                target_images = [
                    Image.fromarray(array) for array in target_arrays
                ]
                return sample_images, target_images

        elif mode == "combined":
            if return_mode == "array":
                return target_arrays
            elif return_mode == "image":
                return [Image.fromarray(out) for out in target_arrays]

    def animate(
        self,
        arrays: Optional[Union[tuple[torch.Tensor], torch.Tensor]] = None,
        path_dir: Union[Path, str] = index.dir_data / "temp/bouncing_ball/",
        name: str = "animation",
        save_animation: bool = False,
        save_target: bool = False,
        mode: str = "original",
        multiplier: int = 2,
        display_animation: bool = True,
        duration=0,
        loop=0,
        num_sequences=1,
        include_timestep: bool = True,
        sample_thickness=2,
        as_mp4=False,
        return_path=False,
        animate_as_sample=False,
    ):
        # To set them back when we are done
        original_sample_mode = self.sample_mode
        original_target_mode = self.target_mode

        if arrays is None:
            self.sample_mode = "parameter_array"
            self.target_mode = "parameter_array"
            sample_sequence, target_sequence = zip(*[outs for outs in self])
        elif len(arrays) == 2:
            sample_sequence, target_sequence = arrays
        else:
            sample_sequence, target_sequence = arrays, None

        has_target = target_sequence is not None

        if isinstance(sample_sequence, (np.ndarray, torch.Tensor)):
            if len(sample_sequence.shape) > 2:
                batch_size = sample_sequence.shape[2]
            else:
                batch_size = 1
        elif isinstance(
            element := sample_sequence[0], (np.ndarray, torch.Tensor)
        ):
            if len(element.shape) > 2:
                batch_size = element.shape[0]
            else:
                batch_size = 1
        else:
            batch_size = 1

        if batch_size > 1:
            if isinstance(sample_sequence, tuple):
                sample_sequence = [sample for sample in zip(*sample_sequence)][
                    :num_sequences
                ]
                if has_target:
                    target_sequence = [
                        target for target in zip(*target_sequence)
                    ][:num_sequences]
            else:
                sample_sequence = sample_sequence[:, :num_sequences, :]
                if has_target:
                    target_sequence = target_sequence[:, :num_sequences, :]

            if not has_target:
                target_sequence = [target_sequence]

        else:
            sample_sequence = [sample_sequence]
            target_sequence = [target_sequence]

        for i, (samples, targets) in enumerate(
            zip(sample_sequence, target_sequence)
        ):
            sample_images = self.parameter_to_image(
                samples,
                self.size_frame,
                self.mask_start,
                self.mask_end,
                self.mask_color,
                self.ball_radius,
                target_parameters=targets,
                mode=(mode := mode.lower()),
                return_mode="image" if display_animation is False else "array",
                multiplier=multiplier,
                sample_thickness=sample_thickness
                if (has_target or animate_as_sample)
                else -1,
            )
            if targets is not None and mode == "original":
                sample_images, target_images = sample_images

            sample_path, target_path = None, None

            if save_animation:
                sample_path = gif.save_gif(
                    sample_images,
                    path_dir=str(path_dir),
                    name=name,
                    duration=duration,
                    loop=0,
                    include_timestep=include_timestep,
                    as_mp4=as_mp4,
                    return_path=return_path,
                )

                if has_target and mode == "original":
                    if name.endswith(".gif"):
                        name = name.rsplit(".", 1)[
                            0
                        ]  # reverse split, 1 occurence
                    target_name = "_".join([name, "target"]) + ".gif"

                    target_path = gif.save_gif(
                        target_images,
                        path_dir=str(path_dir),
                        name=target_name,
                        duration=duration,
                        loop=0,
                        include_timestep=include_timestep,
                        as_mp4=as_mp4,
                        return_path=return_path,
                    )

            if display_animation:
                if has_target and mode == "original":
                    output = [
                        np.concatenate((sample, target), 1)
                        for sample, target in zip(sample_images, target_images)
                    ]
                else:
                    output = sample_images

                fig = plt.figure()
                fig_outputs = [
                    [plt.imshow(out, animated=True)] for out in output
                ]
                sequence_animation = animation.ArtistAnimation(
                    fig,
                    fig_outputs,
                    interval=duration,
                    blit=True,
                    repeat_delay=500,
                )
                plt.show()

        self.sample_mode = original_sample_mode
        self.target_mode = original_target_mode

        if return_path:
            return sample_path, target_path

    @property
    def sequence(self) -> list[tuple[np.ndarray]]:
        return self._sequence

    @sequence.setter
    def sequence(self, value):
        self._sequence = value
        if len(self._sequence) == 0:
            self._samples = None
            self._targets = None

    @property
    def samples(self) -> np.ndarray:
        if self.sequence and (
            self._samples is None
            or self._samples.shape[1] != len(self.sequence)
        ):
            samples, targets = zip(*self.sequence)
            self._samples = np.array(samples).transpose(1, 0, 2)
            self._targets = np.array(targets).transpose(1, 0, 2)
        return self._samples

    @property
    def targets(self) -> np.ndarray:
        if self.sequence and (
            self._targets is None
            or self._targets.shape[1] != len(self.sequence)
        ):
            samples, targets = zip(*self.sequence)
            self._samples = np.array(samples).transpose(1, 0, 2)
            self._targets = np.array(targets).transpose(1, 0, 2)
        return self._targets

    # def __repr__(self):
    #     params = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
    #     return f"{self.__class__.__name__}({params})"

    def __str__(self):
        # Basic overall task parameters
        description = (
            f"BouncingBallTask(\n"
            f"    size_frame={self.size_frame},\n"
            f"    sequence_length={self.sequence_length},\n"
            f"    ball_radius={self.ball_radius},\n"
            f"    dt={self.dt},\n"
            f"    seed={self.seed},\n"
            f"    batch_size={self.batch_size},\n"
            f"    target_future_timestep={self.target_future_timestep},\n"
            f"    min_t_color_change={self.min_t_color_change},\n"
            f"    sequence_mode='{self.sequence_mode}',\n"
            f"    sample_mode='{self.sample_mode}',\n"
            f"    target_mode='{self.target_mode}',\n"
        )

        # Add the change mode and conditionally add mode itself if its true
        description += f"    return_change={self.return_change},\n"
        if self.return_change:
            description += (
                f"    return_change_mode='{self.return_change_mode}',\n"
            )

        # Conditionally add the mask parameters if its not the default
        if self.mask_center != 0.5 or np.round(
            self.mask_fraction, 3
        ) != np.round(1 / 3, 3):
            description += (
                f"    mask_center={self.mask_center},\n"
                f"    mask_fraction={self.mask_fraction},\n"
            )

        # Add in the the discrete velocity sampling variables if using discrete
        # sampling
        description += f"    sample_velocity_discretely={self.sample_velocity_discretely},\n"
        if self.sample_velocity_discretely:
            description += (
                f"    num_x_velocities={self.num_x_velocities},\n"
                f"    num_y_velocities={self.num_y_velocities},\n"
                f"    velocity_x_lower_multiplier={self.velocity_x_lower_multiplier},\n"
                f"    velocity_x_upper_multiplier={self.velocity_x_upper_multiplier},\n"
                f"    velocity_y_lower_multiplier={self.velocity_y_lower_multiplier},\n"
                f"    velocity_y_upper_multiplier={self.velocity_y_upper_multiplier},\n"
            )

        # Add in the probability parameters
        description += (
            f"    probability_velocity_change={self.probability_velocity_change if isinstance(self.probability_velocity_change, (float, int)) else (np.unique(self.probability_velocity_change), len(self.probability_velocity_change))},\n"
            f"    probability_color_change_no_velocity_change={self.probability_color_change_no_velocity_change if isinstance(self.probability_color_change_no_velocity_change, (float, int)) else (np.unique(self.probability_color_change_no_velocity_change), len(self.probability_color_change_no_velocity_change))},\n"
            f"    probability_color_change_on_velocity_change={self.probability_color_change_on_velocity_change if isinstance(self.probability_color_change_on_velocity_change, (float, int)) else (np.unique(self.probability_color_change_on_velocity_change), len(self.probability_color_change_on_velocity_change))},\n"
        )

        description += ")"

        return description


if __name__ == "__main__":

    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(
        description="Script for the bouncing ball task. Can be used to generate videos for humans or models."
    )

    # Add arguments based on the function signature
    parser.add_argument(
        "--size_frame",
        nargs=2,
        type=int,
        default=(256, 256),
        help="Frame size as two integers (width, height)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=500,
        help="Length of the sequence",
    )
    parser.add_argument(
        "--ball_radius", type=int, default=10, help="Radius of the ball"
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Time delta for updates"
    )
    parser.add_argument(
        "--probability_velocity_change",
        type=float,
        default=0.0,
        help="Probability of velocity change per step",
    )
    parser.add_argument(
        "--probability_color_change_no_velocity_change",
        type=float,
        default=0.01,
        help="Probability of color change without velocity change",
    )
    parser.add_argument(
        "--probability_color_change_on_velocity_change",
        type=float,
        default=1.0,
        help="Probability of color change on velocity change",
    )
    parser.add_argument(
        "--initial_position",
        nargs="+",
        action="append",
        type=float,
        help="Initial position of the object, as a float iterable",
    )
    parser.add_argument(
        "--same_xy_velocity",
        action="store_true",
        help="Flag to use the same velocity for X and Y directions",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for processing"
    )
    parser.add_argument(
        "--batch_first",
        action="store_true",
        help="Whether the batch dimension comes first",
    )
    parser.add_argument(
        "--initial_velocity",
        nargs="+",
        type=float,
        help="Initial velocity of the object, as a float iterable",
    )
    parser.add_argument(
        "--color_sampling",
        type=str,
        default="fixed",
        help="Method of sampling colors",
    )
    parser.add_argument(
        "--probability_initial_colors",
        nargs="+",
        type=float,
        help="Probability of initial colors as a tuple",
    )
    parser.add_argument(
        "--initial_color",
        nargs="+",
        type=int,
        help="Initial color of the object, as int iterable",
    )
    parser.add_argument(
        "--valid_colors",
        type=str,
        default="default",
        help="Valid colors for the object, could be 'default' or a list of colors",
    )
    parser.add_argument("--num_colors", type=int, help="Number of valid colors")
    parser.add_argument(
        "--mask_center", type=float, default=0.5, help="Center of the mask zone"
    )
    parser.add_argument(
        "--mask_fraction",
        type=float,
        default=1 / 3,
        help="Fraction of the frame to be masked",
    )
    parser.add_argument(
        "--mask_color",
        nargs=3,
        type=int,
        default=(127, 127, 127),
        help="Color of the mask zone as three integers (R, G, B)",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="parameter_array",
        help="Mode of sampling parameters",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="parameter_array",
        help="Mode for setting target parameters",
    )
    parser.add_argument(
        "--target_future_timestep",
        type=int,
        default=0,
        help="Target future timestep for predictions",
    )
    parser.add_argument(
        "--return_change",
        action="store_true",
        help="Flag to return changes in the output",
    )
    parser.add_argument(
        "--return_change_mode",
        type=str,
        default="any",
        help="Mode of return change",
    )
    parser.add_argument(
        "--sequence_mode",
        type=str,
        default="static",
        help="Mode of sequence generation",
    )
    parser.add_argument(
        "--reset_after_iter",
        action="store_true",
        help="Flag to reset state after each iteration",
    )
    parser.add_argument(
        "--min_t_color_change",
        type=int,
        default=20,
        help="Minimum time between color changes",
    )
    parser.add_argument(
        "--forced_velocity_bounce_x",
        nargs="+",
        type=int,
        help="Forced X velocities on bounce as a list",
    )
    parser.add_argument(
        "--forced_velocity_bounce_y",
        nargs="+",
        type=int,
        help="Forced Y velocities on bounce as a list",
    )
    parser.add_argument(
        "--forced_velocity_resamples",
        nargs="+",
        type=int,
        help="Forced resampling of velocities as a list",
    )
    parser.add_argument(
        "--forced_color_changes",
        nargs="+",
        type=int,
        help="Forced color changes as a list",
    )
    parser.add_argument(
        "--velocity_x_lower_multiplier",
        type=float,
        default=1 / 12.5,
        help="Lower bound multiplier for the velocity sampling",
    )
    parser.add_argument(
        "--velocity_x_upper_multiplier",
        type=float,
        default=1 / 7.5,
        help="Upper bound multiplier for the velocity sampling",
    )
    parser.add_argument(
        "--velocity_y_lower_multiplier",
        type=float,
        default=1 / 12.5,
        help="Lower bound multiplier for the velocity sampling",
    )
    parser.add_argument(
        "--velocity_y_upper_multiplier",
        type=float,
        default=1 / 7.5,
        help="Upper bound multiplier for the velocity sampling",
    )
    parser.add_argument(
        "--sample_velocity_discretely",
        action="store_true",
        help="Samples velocity values discretely",
    )
    parser.add_argument(
        "--num_x_velocities",
        type=int,
        default=1,
        help="Number of velocities to sample in x",
    )
    parser.add_argument(
        "--num_y_velocities",
        type=int,
        default=2,
        help="Number of velocities to sample in y",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for initialization"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    parser.add_argument(
        "--pccnvc_lower", type=float, help="Lower limit of PCCNVC"
    )
    parser.add_argument(
        "--pccnvc_upper", type=float, help="Upper limit of PCCNVC"
    )
    parser.add_argument(
        "--pccovc_lower", type=float, help="Lower limit of PCCOVC"
    )
    parser.add_argument(
        "--pccovc_upper", type=float, help="Upper limit of PCCOVC"
    )
    parser.add_argument(
        "--num_pccnvc", type=int, help="Number of PCCNVC instances"
    )
    parser.add_argument(
        "--num_pccovc", type=int, help="Number of PCCOVC instances"
    )
    parser.add_argument(
        "--color_change_bounce_delay",
        type=int,
        default=0,
        help="Timesteps to delay a color change after a bounce",
    )
    parser.add_argument(
        "--color_change_random_delay",
        type=int,
        default=0,
        help="Timesteps to delay a random color change",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Time (ms) between frames for viewing",
    )
    parser.add_argument("--multiplier", type=int, default=2, help="Multiplier")
    parser.add_argument(
        "--save_animation", action="store_true", help="Saves the animation"
    )
    parser.add_argument(
        "--animate_target", action="store_true", help="Animates just the target"
    )
    parser.add_argument(
        "--animate_sample", action="store_true", help="Animates just the sample"
    )
    parser.add_argument(
        "--as_mp4", action="store_true", help="Saves the animation as an mp4"
    )
    parser.add_argument(
        "--animate_as_sample",
        action="store_true",
        help="Animates the array as if its a sample video",
    )
    parser.add_argument(
        "--skip_animation",
        action="store_false",
        help="Skips showing the animation",
    )
    parser.add_argument(
        "--no_timestep",
        action="store_false",
        help="Removes the timestep from the videos",
    )

    # Parse the arguments from the command line
    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    logger = logutils.configure_logger(verbose=args.debug)

    # Instantiate BouncingBallTask with all arguments
    task = BouncingBallTask(
        size_frame=args.size_frame,
        sequence_length=args.sequence_length,
        ball_radius=args.ball_radius,
        dt=args.dt,
        probability_velocity_change=args.probability_velocity_change,
        probability_color_change_no_velocity_change=args.probability_color_change_no_velocity_change,
        probability_color_change_on_velocity_change=args.probability_color_change_on_velocity_change,
        initial_position=args.initial_position,
        same_xy_velocity=args.same_xy_velocity,
        batch_size=args.batch_size,
        batch_first=args.batch_first,
        initial_velocity=args.initial_velocity,
        velocity_x_lower_multiplier=args.velocity_x_lower_multiplier,
        velocity_x_upper_multiplier=args.velocity_x_upper_multiplier,
        velocity_y_lower_multiplier=args.velocity_y_lower_multiplier,
        velocity_y_upper_multiplier=args.velocity_y_upper_multiplier,
        color_sampling=args.color_sampling,
        probability_initial_colors=args.probability_initial_colors,
        initial_color=args.initial_color,
        valid_colors=args.valid_colors,
        num_colors=args.num_colors,
        mask_center=args.mask_center,
        mask_fraction=args.mask_fraction,
        mask_color=args.mask_color,
        sample_mode=args.sample_mode,
        target_mode=args.target_mode,
        target_future_timestep=args.target_future_timestep,
        return_change=args.return_change,
        return_change_mode=args.return_change_mode,
        sequence_mode=args.sequence_mode,
        reset_after_iter=args.reset_after_iter,
        min_t_color_change=args.min_t_color_change,
        forced_velocity_bounce_x=args.forced_velocity_bounce_x,
        forced_velocity_bounce_y=args.forced_velocity_bounce_y,
        forced_velocity_resamples=args.forced_velocity_resamples,
        forced_color_changes=args.forced_color_changes,
        seed=args.seed,
        debug=args.debug,
        pccnvc_lower=args.pccnvc_lower,
        pccnvc_upper=args.pccnvc_upper,
        pccovc_lower=args.pccovc_lower,
        pccovc_upper=args.pccovc_upper,
        num_pccnvc=args.num_pccnvc,
        num_pccovc=args.num_pccovc,
        sample_velocity_discretely=args.sample_velocity_discretely,
        num_x_velocities=args.num_x_velocities,
        num_y_velocities=args.num_y_velocities,
        color_change_bounce_delay=args.color_change_bounce_delay,
        color_change_random_delay=args.color_change_random_delay,
    )

    print(task)

    samples, targets = zip(*[x for x in task])
    samples = np.array(samples).transpose(1, 0, 2)
    targets = np.array(targets).transpose(1, 0, 2)

    if args.animate_sample:
        output_to_animate = samples[0]
        args.animate_as_sample = True
    elif args.animate_target:
        output_to_animate = targets[0]
    else:
        output_to_animate = (samples[0], targets[0])

    paths = task.animate(
        output_to_animate,
        duration=args.duration,
        multiplier=args.multiplier,
        save_target=False,
        save_animation=args.save_animation,
        display_animation=args.skip_animation,
        num_sequences=1,
        include_timestep=args.no_timestep,
        return_path=args.save_animation,
        as_mp4=args.as_mp4,
        animate_as_sample=args.animate_as_sample,
    )
    if args.save_animation:
        print(f"Saving videos to {paths}")
