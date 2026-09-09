"""Shared pytest fixtures for bouncing ball task tests."""
import pytest
import tempfile
from pathlib import Path
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bouncing_ball_task.color_controlled_bouncing_ball.dataset import (
    generate_color_controlled_dataset,
    generate_color_controlled_dataset_with_videos
)
from bouncing_ball_task.color_controlled_bouncing_ball.no_change import generate_no_change_trials
from bouncing_ball_task.utils.pyutils import set_global_seed
from bouncing_ball_task.color_controlled_bouncing_ball.defaults import (
    ColorControlledDatasetParameters,
    ColorControlledTaskParameters,
)


@pytest.fixture(scope="session")
def session_temp_dir():
    """Create a temporary directory for the entire test session."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for each test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def color_controlled_dataset_small():
    """Generate a small color controlled dataset for testing."""
    color_controlled_params = {
        'num_base_sequences': 5,
        'seed': 42,
        'duration': 50,
        'variable_length': False,
    }

    task_params = {
        'sequence_length': 100,
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


@pytest.fixture
def color_controlled_dataset_with_videos(temp_output_dir):
    """Generate color controlled dataset with video files."""
    color_controlled_params = {
        'num_base_sequences': 3,
        'seed': 123,
        'duration': 50,
        'variable_length': False,
    }

    task_params = {
        'sequence_length': 50,
        'size_frame': (256, 256),
        'ball_radius': 10,
    }

    task, samples, model_samples, targets, df_data, metadata = generate_color_controlled_dataset_with_videos(
        color_controlled_params,
        task_params,
        output_dir=temp_output_dir,
        shuffle=False,
        dict_trial_type_generation_funcs={'no_change': generate_no_change_trials}
    )
    
    # Find the generated dataset directory
    dataset_dirs = list(temp_output_dir.glob("color_controlled_*"))
    assert len(dataset_dirs) == 1, f"Expected 1 dataset directory, found {len(dataset_dirs)}"
    dataset_dir = dataset_dirs[0]
    
    return {
        'dataset_dir': dataset_dir,
        'task': task,
        'samples': samples,
        'model_samples': model_samples,
        'targets': targets,
        'df_data': df_data,
        'metadata': metadata,
        'output_dir': temp_output_dir,
    }


@pytest.fixture
def multiple_variants_dataset():
    """Generate dataset with multiple variants for testing."""
    # Define a test variant
    def generate_test_variant_trials(positions, bounce_frames, params):
        """Color-only test variant: a contingent color change on every bounce.

        Authors channels [7 (cc_bounce), 8 (cc_random)] only, as (N, T, 2).
        """
        color_events = np.zeros(bounce_frames.shape + (2,), dtype=int)
        color_events[:, :, 0] = bounce_frames
        return color_events

    generate_test_variant_trials.variant_metadata = {'description': 'Test variant'}

    color_controlled_params = {
        'num_base_sequences': 4,
        'seed': 999,
        'duration': 50,
        'variable_length': False,
    }

    task_params = {
        'sequence_length': 80,
        'size_frame': (256, 256),
        'ball_radius': 10,
    }

    task, samples, model_samples, targets, df_data, metadata = generate_color_controlled_dataset(
        color_controlled_params,
        task_params,
        shuffle=False,
        dict_trial_type_generation_funcs={
            'no_change': generate_no_change_trials,
            'test_variant': generate_test_variant_trials,
        }
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


@pytest.fixture
def default_controlled_params():
    """Get default controlled dataset parameters."""
    return ColorControlledDatasetParameters()


@pytest.fixture
def default_task_params():
    """Get default controlled task parameters."""
    return ColorControlledTaskParameters()


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for reproducibility.

    Uses ``set_global_seed`` rather than ``np.random.seed`` alone: tasks that are
    constructed without an explicit ``seed=`` call ``set_global_seed(None)``,
    which draws its seed from the stdlib ``random`` module. Seeding only numpy
    therefore left unseeded tests nondeterministic.
    """
    set_global_seed(42)
    yield
    # Cleanup if needed
