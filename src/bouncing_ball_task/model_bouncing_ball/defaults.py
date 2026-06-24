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

# Effective-hazard-rate estimator sample size (E5). Decoupled from the final
# dataset size (total_videos=18000) and pinned by calibration to absolute
# tolerance eps=0.005 on the per-`Hazard Rate` mean PCCNVC_effective. Derivation:
# pilot within-group sigma_g of PCCNVC_effective -> analytic per-group bound
# n_req=(1.96*max_g sigma_g/0.005)**2 -> replicate sweep confirming worst-case
# smallest-group count clears n_req and across-replicate std of each per-group
# mean < 0.005. (Calibration evidence in the project's durable PerAnkh
# plans/ calibration learnings artifact, kept out of the repo per the repo's
# process-artifact policy.)
ESTIMATE_N: int = 1000


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
