"""Human task defaults"""
from dataclasses import dataclass
from typing import Optional, Union
from bouncing_ball_task.utils import pyutils as _pyutils
from bouncing_ball_task.defaults import TaskParameters as _BaseTaskParameters


name_dataset: str = "hbb_dataset"
multiplier: int = 2
mode: str = "original"
include_timestep: bool = False
display_animation: bool = False


@dataclass
class TaskParameters(_BaseTaskParameters):
    batch_size: Optional[int] = None
    sequence_mode: str = "reverse"
    target_future_timestep: int = 0
    sample_velocity_discretely: bool = True
    initial_velocity_points_away_from_grayzone: bool = True
    initial_timestep_is_changepoint: bool = False
    min_t_color_change_after_random: int = 15
    min_t_color_change_after_bounce: int = 15
    min_t_velocity_change_after_random: int = 15
    min_t_velocity_change_after_bounce: int = 15
    warmup_t_no_rand_velocity_change: int = 20
    warmup_t_no_rand_color_change: int = 5


@dataclass
class HumanDatasetParameters:
    size_frame: int = TaskParameters.size_frame
    ball_radius: int = TaskParameters.ball_radius
    dt: float = TaskParameters.dt
    mask_center: int = TaskParameters.mask_center
    mask_fraction: float = TaskParameters.mask_fraction

    total_dataset_length: int = 35
    num_blocks: int = 25
    variable_length: bool = True
    duration: int = 30
    trial_type_split: tuple[Optional[Union[int, float]], ...] = (0.05, -1, -1, -1)
    video_length_min_s: float = 6.0
    exp_scale: float = 1.75
    standard: bool = True

    pvc: float = 0.001
    pccnvc_lower: float = 0.00575
    pccnvc_upper: float = 0.0575
    pccovc_lower: float = 0.025
    pccovc_upper: float = 0.975
    num_pccnvc: int = 2
    num_pccovc: int = 3

    num_y_velocities: int = 2
    velocity_lower: float = 0.08
    velocity_upper: float = 0.1333333333

    num_pos_x_endpoints: int = 3
    num_pos_y_endpoints: int = 8
    y_pos_multiplier: int = 8
    bounce_offset: float = 2/5
    border_tolerance_outer: float = 1.25
    border_tolerance_inner: float = 1.0

    num_pos_x_linspace_bounce: int = 5
    idx_linspace_bounce: int = 1
    bounce_timestep: int = 7    
    repeat_factor: int = 3
    
    total_videos: Optional[int] = None
    fixed_video_length: Optional[int] = None

    use_logger: bool = False
    print_stats: bool = True
    seed: Optional[int] = None


_pyutils.register_defaults(globals())
