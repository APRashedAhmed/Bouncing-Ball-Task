"""Fixtures specific to controlled dataset variant tests."""
import pytest
from pathlib import Path

from bouncing_ball_task.color_controlled_bouncing_ball.dataset import generate_color_controlled_dataset
from bouncing_ball_task.color_controlled_bouncing_ball.no_change import generate_no_change_trials


@pytest.fixture
def no_change_dataset():
    """Generate a dataset with only the no_change variant."""
    color_controlled_params = {
        'num_base_sequences': 3,
        'seed': 12345,
        'duration': 50,
        'variable_length': False,
    }
    
    task_params = {
        'sequence_length': 50,
        'size_frame': (256, 256),
        'ball_radius': 10,
    }
    
    task, samples, model_samples, targets, df_data, metadata = generate_color_controlled_dataset(
        color_controlled_params,
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
        'color_controlled_params': color_controlled_params,
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
