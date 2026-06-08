"""Default parameters for controlled bouncing ball dataset generation.

This module defines the parameter classes for generating controlled datasets
using the change vector manipulation approach.
"""
from dataclasses import dataclass
from typing import Optional
from bouncing_ball_task.utils import pyutils as _pyutils


# Display and file format defaults
name_dataset: str = "controlled_dataset"
multiplier: int = 2
mode: str = "original"
include_timestep: bool = False
display_animation: bool = False
duration: int = 1000  # Default duration in milliseconds


@dataclass
class ControlledTaskParameters:
    """Task parameters for controlled dataset generation.
    
    These parameters control the bouncing ball physics and rendering.
    """
    # Core task parameters
    sequence_length: int = 150
    size_frame: tuple[int, int] = (256, 256)
    ball_radius: int = 10
    dt: float = 0.01
    
    # Physics parameters
    gravity_mag: float = 9.8
    elasticity: float = 1.0
    
    # Rendering parameters
    mask_start: int = 56
    mask_end: int = 200
    mask_color: tuple[int, int, int] = (127, 127, 127)
    
    # Advanced options
    sequence_mode: str = "reverse"
    target_future_timestep: int = 0
    sample_velocity_discretely: bool = True
    return_change: bool = True
    return_change_mode: str = "source"
    
    # Color change probabilities (enforced to 0 for controlled experiments)
    probability_color_change_no_velocity_change: float = 0.0
    probability_color_change_on_velocity_change: float = 0.0
    probability_velocity_change: float = 0.075


@dataclass 
class ControlledDatasetParameters:
    """Dataset parameters for controlled dataset generation.
    
    These parameters control dataset size and generation options.
    """
    # Core dataset parameters
    num_base_sequences: int = 10  # Number of base sequences to generate
    seed: Optional[int] = None    # Random seed for reproducibility
    duration: int = 1000          # Duration in milliseconds
    variable_length: bool = False # Whether to use variable length sequences
    
    # Color change parameters (all zero for controlled experiments)
    pccnvc_lower: float = 0.0  # No color changes without velocity change
    pccnvc_upper: float = 0.0
    pccovc_lower: float = 0.0  # No color changes on velocity change
    pccovc_upper: float = 0.0


# Register defaults to enable .keys property on dataclasses
_pyutils.register_defaults(globals())
