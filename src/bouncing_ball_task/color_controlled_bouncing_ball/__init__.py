"""Color-controlled bouncing ball dataset generation module.

This module generates color-controlled datasets for the bouncing ball task with precise
parameter control for experimental variants. Currently supports the no_change
variant (hazard_rate=0, contingency=0).

Example:
    from bouncing_ball_task.color_controlled_bouncing_ball import generate_color_controlled_dataset
    from bouncing_ball_task.color_controlled_bouncing_ball.defaults import (
        ColorControlledDatasetParameters,
        ColorControlledTaskParameters
    )
"""

from .dataset import generate_color_controlled_dataset
from .defaults import ColorControlledDatasetParameters, ColorControlledTaskParameters

__all__ = [
    'generate_color_controlled_dataset',
    'ColorControlledDatasetParameters',
    'ColorControlledTaskParameters',
]
