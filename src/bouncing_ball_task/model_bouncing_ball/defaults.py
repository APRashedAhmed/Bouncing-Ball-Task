"""Human task defaults"""
from dataclasses import dataclass
from typing import Optional, Union
from bouncing_ball_task.utils import pyutils as _pyutils
from bouncing_ball_task.human_bouncing_ball.defaults import (
    TaskParameters,
    HumanDatasetParameters,
    multiplier,
    mode,
    include_timestep,
    display_animation,
)

name_dataset: str = "mbb_dataset"

@dataclass
class ModelDatasetParameters(HumanDatasetParameters):
    total_dataset_length: Optional[int] = None
    num_blocks: Optional[int] = None
    duration: int = 30
    trial_type_split: tuple[Optional[Union[int, float]], ...] = (1, 1, 1, 1, 1, 1,)
    num_pos_y_endpoints: int = 20
    standard: bool = False
    
    num_pos_x_endpoints: Optional[int] = None
    num_pos_x_linspace_bounce: Optional[int] = None
    total_videos: Optional[int] = 18000

@dataclass
class NongrayDatasetParameters(ModelDatasetParameters):
    ncc_nvc_timesteps: int = 20
    timestep_change: int = ncc_nvc_timesteps // 2
    timestep_from_wall: int = 5    
    
_pyutils.register_defaults(globals())
