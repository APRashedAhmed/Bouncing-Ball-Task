"""Controlled bouncing ball dataset generation module.

This module generates controlled datasets for the bouncing ball task with precise
parameter control for experimental variants. Currently supports the no_change
variant (hazard_rate=0, contingency=0).

Example:
    from bouncing_ball_task.controlled_bouncing_ball import generate_controlled_dataset
    from bouncing_ball_task.controlled_bouncing_ball.defaults import (
        ControlledDatasetParameters, 
        ControlledTaskParameters
    )
"""

from .dataset import generate_controlled_dataset
from .defaults import ControlledDatasetParameters, ControlledTaskParameters

__all__ = [
    'generate_controlled_dataset',
    'ControlledDatasetParameters', 
    'ControlledTaskParameters',
]