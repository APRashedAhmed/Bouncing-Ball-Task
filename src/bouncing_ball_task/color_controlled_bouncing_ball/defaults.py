"""Default parameters for color-controlled bouncing ball dataset generation.

The task parameters SUBCLASS the human task's ``TaskParameters`` so that the
color-controlled stimulus is parameter-identical to the real task (``dt``,
``sequence_length``, ``color_mask_mode="outer"``, warmup / min_t /
transition_tol, ``initial_timestep_is_changepoint=False``) — the LSTM loader
reconstructs a ``BouncingBallTask`` from these values, so any divergence would
be a different stimulus wearing the same schema.
"""
from dataclasses import dataclass
from typing import Optional
from bouncing_ball_task.utils import pyutils as _pyutils
from bouncing_ball_task.human_bouncing_ball.defaults import (
    TaskParameters as _HumanTaskParameters,
)


# Display and file format defaults
name_dataset: str = "color_controlled_dataset"
multiplier: int = 2
mode: str = "original"
include_timestep: bool = False
display_animation: bool = False
duration: int = 50  # Milliseconds PER FRAME (the real task's default)


@dataclass
class ColorControlledTaskParameters(_HumanTaskParameters):
    """Task parameters for color-controlled dataset generation.

    Everything not listed here is inherited from the human task (see module
    docstring). Only fields that are genuine ``BouncingBallTask`` kwargs live
    here; the former non-task fields (``gravity_mag``, ``elasticity``,
    ``mask_start``, ``mask_end``) were swallowed by ``**kwargs`` and are gone.
    The grayzone is derived from ``mask_center`` / ``mask_fraction`` exactly as
    in the real task.
    """
    # PVC = 0: the only velocity changes are wall bounces, so a trajectory is a
    # pure function of its initial conditions (operator decision 2026-09-04).
    probability_velocity_change: float = 0.0

    # The base task's own color changes are discarded (colors are re-authored by
    # the variants), so keep them off.
    probability_color_change_no_velocity_change: float = 0.0
    probability_color_change_on_velocity_change: float = 0.0

    # Frame 0 is natively zero in the change vector, as in the real task.
    initial_timestep_is_changepoint: bool = False


@dataclass
class ColorControlledDatasetParameters:
    """Dataset parameters for color-controlled dataset generation.

    Initial conditions are explicit-else-sampled: ``initial_position`` /
    ``initial_velocity`` are optional ``(N, 2)`` arrays (``None`` => sampled
    with the task's own samplers). ``control_end`` is an optional ``(N,)`` bool
    array (``None`` => all False): ``False`` keeps the trajectory so the trial
    STARTS at the requested position (``sequence_mode="static"``); ``True``
    reverses it so the trial ENDS there (``sequence_mode="reverse"``).
    """
    # Core dataset parameters
    num_base_sequences: int = 10  # Number of base sequences to generate
    seed: Optional[int] = None    # Random seed for reproducibility
    duration: int = 50            # Milliseconds PER FRAME (length_ms = length * duration)
    variable_length: bool = False # Whether to use variable length sequences

    # Controlled initial conditions (None => sampled)
    initial_position: Optional[tuple] = None
    initial_velocity: Optional[tuple] = None
    control_end: Optional[tuple] = None

    # Color change parameters (all zero for controlled experiments)
    pccnvc_lower: float = 0.0  # No color changes without velocity change
    pccnvc_upper: float = 0.0
    pccovc_lower: float = 0.0  # No color changes on velocity change
    pccovc_upper: float = 0.0


# Register defaults to enable .keys property on dataclasses
_pyutils.register_defaults(globals())
