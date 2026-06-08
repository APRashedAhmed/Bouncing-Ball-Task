"""No change variant implementation for controlled bouncing ball dataset.

This module implements the simplest controlled variant where no color changes
occur throughout the trial (hazard_rate=0, contingency=0).
"""
from loguru import logger
import numpy as np


def generate_no_change_trials(
    change_vectors,
    controlled_dataset_parameters,
    task_parameters,
    base_task,
):
    """Generate trials with no color changes (hazard_rate=0, contingency=0).
    
    For the no_change variant, we don't modify the change vectors at all since
    color changes have already been zeroed out in the main generation function.
    This variant represents the simplest controlled condition.
    
    Args:
        change_vectors: Array of change vectors to modify (N, T, 4)
        controlled_dataset_parameters: Dataset parameters
        task_parameters: Task parameters
        base_task: Base BouncingBallTask instance
        
    Returns:
        tuple: (change_vectors, dict_meta_type) unchanged vectors and metadata
    """
    # For no_change variant, we don't modify the change vectors
    # Color changes are already zeroed out in the main function
    # The change vectors have shape (N, T, 4) where:
    # - [:, :, 0]: Bounce velocity changes
    # - [:, :, 1]: Random velocity changes  
    # - [:, :, 2]: Color changes on velocity change (already zeroed)
    # - [:, :, 3]: Random color changes (already zeroed)
    
    num_trials = change_vectors.shape[0]
    logger.info(f"Generating {num_trials} no_change trials")
    
    # Create metadata for this variant
    dict_meta_type = {
        "num_trials": num_trials,
        "variant_name": "no_change",
        "change_strategy": "No modifications to change vectors",
        "change_parameters": {
            "allow_velocity_changes": True,  # Natural bounces still occur
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
        "validation": {
            "color_changes_zeroed": np.all(change_vectors[:, :, -2:] == 0),
            "description": "Verified no color changes in change vectors",
        }
    }
    
    # Log statistics if requested
    logger.debug(f"No-change variant: {num_trials} trials, no modifications to change vectors")
    logger.debug(f"Color change validation: {dict_meta_type['validation']['color_changes_zeroed']}")
    
    # Return unchanged vectors and metadata
    return change_vectors, dict_meta_type
