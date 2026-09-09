"""No-change variant for the color-controlled bouncing ball dataset.

The simplest color-controlled variant: no color changes occur throughout the
trial (hazard_rate=0, contingency=0). The trajectory (and hence the bounce
ledger) is untouched — variants never author velocity channels.
"""
from loguru import logger
import numpy as np


def generate_no_change_trials(positions, bounce_frames, params):
    """Author zero color events for every trial.

    Color-only authoring contract (P4)::

        variant(positions, bounce_frames, params) -> np.ndarray (N, T, 2)

    Args:
        positions: ``(N, T, 2)`` trajectory positions — read-only.
        bounce_frames: ``(N, T)`` bool — the derived wall-bounce ledger
            (targets channel 5), read-only. Contingent color changes
            (channel 7) may fire ONLY where this is True.
        params: dict with ``controlled_dataset_parameters``,
            ``task_parameters`` (the resolved base task kwargs),
            ``initial_position``, ``initial_velocity``, ``control_end``.

    Returns:
        ``(N, T, 2)`` array of authored color events: ``[:, :, 0]`` is the
        contingent (on-bounce) color change (channel 7), ``[:, :, 1]`` the
        random color change (channel 8). All zeros for this variant.
    """
    num_trials, num_timesteps = bounce_frames.shape
    logger.info(f"Generating {num_trials} no_change trials")
    return np.zeros((num_trials, num_timesteps, 2), dtype=int)


# Optional variant-specific metadata, merged by the generator into
# dict_metadata[<variant>]["variant_metadata"] on top of the derived fields
# (variant_name, num_trials, authored event counts).
generate_no_change_trials.variant_metadata = {
    "change_strategy": "No color events authored",
    "hazard_rate": 0.0,
    "contingency": 0.0,
    "change_parameters": {
        "allow_velocity_changes": True,  # Natural wall bounces still occur
        "allow_color_changes": False,    # No color changes
    },
    "expected_changes": {
        "bounce_changes": "Natural wall bounces only",
        "random_changes": 0,
        "color_changes": 0,
        "total_color_changes": 0,
    },
    "enforced_parameters": {
        "probability_color_change_no_velocity_change": 0.0,
        "probability_color_change_on_velocity_change": 0.0,
    },
}
