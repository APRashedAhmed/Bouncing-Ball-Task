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
    

@dataclass
class HumanDatasetParameters:
    size_frame: int = TaskParameters.size_frame
    ball_radius: int = TaskParameters.ball_radius
    dt: float = TaskParameters.dt
    mask_center: int = TaskParameters.mask_center
    mask_fraction: float = TaskParameters.mask_fraction

    total_dataset_length: int = 30
    num_blocks: int = 10
    variable_length: bool = True
    duration: int = 50
    trial_type_split: tuple[Optional[Union[int, float]], ...] = (0.05, -2, -1, -1)
    # trial_type_split: tuple[Optional[Union[int, float]], ...] = (0.01, 0.01, 0.01, -1)
    # trial_type_split: tuple[Optional[Union[int, float]], ...] = (0.01, 0.01, -1, -1)
    video_length_min_s: float = 7.5
    exp_scale: float = 5.0
    standard: bool = True
    catch_ncc_nvc_timesteps: int = 20

    pvc: float = 0.075
    pccnvc_lower: float = 0.015
    pccnvc_upper: float = 0.05
    pccovc_lower: float = 0.125
    pccovc_upper: float = 0.875
    num_pccnvc: int = 2
    num_pccovc: int = 3

    num_y_velocities: int = 2
    velocity_lower: float = 0.0975
    velocity_upper: float = 0.1125

    num_pos_x_endpoints: int = 3
    num_pos_y_endpoints: int = 8
    y_pos_multiplier: int = 8
    bounce_offset: float = 2/5
    border_tolerance_outer: float = 1.25
    border_tolerance_inner: float = 0.1

    num_pos_x_linspace_bounce: int = 5
    idx_linspace_bounce: int = 0
    bounce_timestep: int = 5
    repeat_factor: int = 3
    
    total_videos: Optional[int] = None
    fixed_video_length: Optional[int] = None

    use_logger: bool = False
    print_stats: bool = True
    seed: Optional[int] = None


_pyutils.register_defaults(globals())
