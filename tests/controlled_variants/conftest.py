"""Fixtures specific to controlled dataset variant tests."""
import pytest
from pathlib import Path

from bouncing_ball_task.controlled_bouncing_ball.dataset import generate_controlled_dataset
from bouncing_ball_task.controlled_bouncing_ball.no_change import generate_no_change_trials


@pytest.fixture
def no_change_dataset():
    """Generate a dataset with only the no_change variant."""
    controlled_params = {
        'num_base_sequences': 3,
        'seed': 12345,
        'duration': 1000,
        'variable_length': False,
    }
    
    task_params = {
        'sequence_length': 50,
        'size_frame': (256, 256),
        'ball_radius': 10,
        'probability_velocity_change': 0.075,
    }
    
    task, samples, model_samples, targets, df_data, metadata = generate_controlled_dataset(
        controlled_params,
        task_params,
        shuffle=False,
        dict_trial_type_generation_funcs={'no_change': generate_no_change_trials}
    )
    
    return {
        'task': task,
        'samples': samples,
        'model_samples': model_samples,
        'targets': targets,
        'df_data': df_data,
        'metadata': metadata,
        'controlled_params': controlled_params,
        'task_params': task_params,
    }


# Future variant fixtures would go here:
# @pytest.fixture
# def sudden_change_dataset():
#     """Generate a dataset with only the sudden_change variant."""
#     pass
#
# @pytest.fixture
# def gradual_change_dataset():
#     """Generate a dataset with only the gradual_change variant."""
#     pass
