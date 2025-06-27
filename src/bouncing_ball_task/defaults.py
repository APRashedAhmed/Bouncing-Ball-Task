"""Top level default variables"""
from dataclasses import dataclass
from typing import Optional, Union
from bouncing_ball_task.utils import pyutils as _pyutils


@dataclass
class TaskParameters:
    size_frame: tuple[int, int] = (256, 256)
    sequence_length: int = 600
    ball_radius: int = 10
    target_future_timestep: int = 1
    dt: float = 0.1
    batch_size: Optional[int] = 128
    sequence_mode: str = "static"
    
    sample_mode: str = "parameter_array"
    target_mode: str = "parameter_array"
    
    mask_center: float = 0.5
    mask_fraction: float = 1/3
    mask_color: list[Union[int, float]] = (127, 127, 127)
    
    return_change: bool = True
    return_change_mode: str = "source"

    min_t_color_change_after_bounce: int = 10
    min_t_color_change_after_random: int = 10
    
    min_t_velocity_change_after_bounce: int = 30
    min_t_velocity_change_after_random: int = 10
    
    warmup_t_no_rand_velocity_change: int = 30
    warmup_t_no_rand_color_change: int = 2    
    
    color_sampling: str = "fixed"
    color_mask_mode: str = "inner"
    
    initial_timestep_is_changepoint: bool = False
    initial_velocity_points_away_from_grayzone: bool = True
    
    sample_velocity_discretely: bool = False
    same_xy_velocity: bool = False
    
    num_x_velocities: int = 1
    num_y_velocities: int = 2
    
    color_change_bounce_delay: int = 0
    color_change_random_delay: int = 0
    transitioning_change_mode: Optional[str] = None
    transition_tol: int = 1
    
    debug: bool = False

    
_pyutils.register_defaults(globals())
