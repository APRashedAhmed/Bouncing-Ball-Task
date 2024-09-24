from math import isclose

import numpy as np
import pytest

from bouncing_ball_task.bouncing_ball import BouncingBallTask

# @pytest.fixture(autouse=True)
# def set_seed():
#     np.random.seed(0)  # Set a fixed seed for NumPy's random number generator


@pytest.fixture
def default_task():
    return BouncingBallTask(sequence_mode="reset")


# Test instantiation and basic properties
def test_initialization(default_task):
    assert isinstance(
        default_task, BouncingBallTask
    ), "Should successfully create an instance"


@pytest.mark.parametrize(
    "mode,setter,exception",
    [
        ("invalid_mode", "sequence_mode", ValueError),
        ("invalid_mode", "sample_mode", ValueError),
        ("invalid_mode", "target_mode", ValueError),
        ("invalid_mode", "color_sampling", ValueError),
        ("invalid_mode", "return_change_mode", ValueError),
    ],
)
def test_invalid_modes(default_task, mode, setter, exception):
    with pytest.raises(exception):
        setattr(default_task, setter, mode)


@pytest.mark.parametrize(
    "setter, valid_modes_dict",
    [
        ("sample_mode", {"Parameter_array": "parameter_array"}),
        ("target_mode", {"Parameter_array": "parameter_array"}),
        ("return_change_mode", {"Any": "any"}),
        ("color_sampling", {"Fixed": "fixed_vectorized"}),
    ],
)
def test_valid_modes(default_task, setter, valid_modes_dict):
    for valid_mode, return_mode in valid_modes_dict.items():
        setattr(default_task, setter, valid_mode)
        assert (
            getattr(default_task, setter) == return_mode
        ), f"{setter} should be settable to {valid_mode}"


def test_sample_target_iteration_and_basic_content(default_task):
    # Setup the task with predefined parameters if necessary
    samples_targets = list(zip(*[x for x in default_task]))

    # Check we got samples and targets
    assert (
        len(samples_targets) == 2
    ), f"Each tuple should contain two elements (sample, target) but got {len(samples_targets)} elements."

    # Split into samples and targets
    samples_targets_array = [
        np.array(outputs).transpose(1, 0, 2) for outputs in samples_targets
    ]

    for name, array in zip(("sample", "target"), samples_targets_array):
        # Check the shapes of the outputs are as expected
        batch_size, sequence_length, features = array.shape
        assert (
            batch_size == default_task.batch_size
        ), f"The batch size ({batch_size}) should match the task specified batch size ({default_task.batch_size})."
        assert (
            sequence_length == default_task.sequence_length
        ), f"The number of target iterations ({sequence_length}) should match the sequence length ({default_task.sequence_length})."
        assert (
            features == 5
        ), f"The number of features ({features}) should match the number of task features (5)."


def test_sample_target_has_no_invalid_values(default_task):
    # Setup the task with predefined parameters if necessary
    samples_targets = list(zip(*[x for x in default_task]))

    # Split into samples and targets
    samples_targets_array = [
        np.array(outputs).transpose(1, 0, 2) for outputs in samples_targets
    ]

    for name, array in zip(("sample", "target"), samples_targets_array):

        # Assert no invalid values
        invalid_condition = np.isnan(array) | np.isinf(array)
        if array.dtype == object:
            invalid_condition |= arr == None
        assert not np.any(
            invalid_condition
        ), "Array contains invalid values (None, NaN, or Inf)."


def test_sample_target_has_correct_min_max_values(default_task):
    # Setup the task with predefined parameters if necessary
    samples_targets = list(zip(*[x for x in default_task]))

    # Split into samples and targets
    samples_targets_array = [
        np.array(outputs).transpose(1, 0, 2) for outputs in samples_targets
    ]

    # Useful to define here
    ball_radius = default_task.ball_radius
    dt = default_task.dt
    velocity_mag = np.abs(default_task.initial_velocity)
    size_frame = np.array(default_task.size_frame)

    for name, array in zip(("sample", "target"), samples_targets_array):
        # Check the shapes of the outputs are as expected
        batch_size, sequence_length, features = array.shape

        # Check the max and min values are correct
        min_vals = np.concatenate(
            [ball_radius - velocity_mag * dt, np.zeros((batch_size, 3))],
            axis=-1,
        )
        assert np.all(min_vals <= array.min(axis=1))

        max_vals = np.concatenate(
            [
                size_frame - ball_radius + velocity_mag * dt,
                np.ones((batch_size, 3)) * 255,
            ],
            axis=-1,
        )
        assert np.all(array.max(axis=1) <= max_vals)


@pytest.mark.parametrize(
    "size_x,size_y,batch_size,sample_velocity_discretely,num_x_velocities,num_y_velocities,velocity_x_lower_multiplier,velocity_x_upper_multiplier,velocity_y_lower_multiplier,velocity_y_upper_multiplier",
    [
        (128, 128, 10, False, 10, 10, 0.01, 0.1, 0.01, 0.1),
        (256, 256, 7, False, 7, 1, 0.02, 0.2, 0.02, 0.2),
        (64, 64, 15, True, 3, 15, 0.03, 0.3, 0.03, 0.3),
        (128, 64, 200, True, 200, 25, 0.04, 0.4, 0.04, 0.4),
        (256, 256, 1024, False, 1, 2, 1 / 12.5, 1 / 7.5, 1 / 12.5, 1 / 7.5),
        (256, 256, 1024, True, 1, 2, 1 / 12.5, 1 / 7.5, 1 / 12.5, 1 / 7.5),
    ],
)
def test_sample_velocity(
    size_x,
    size_y,
    batch_size,
    sample_velocity_discretely,
    num_x_velocities,
    num_y_velocities,
    velocity_x_lower_multiplier,
    velocity_x_upper_multiplier,
    velocity_y_lower_multiplier,
    velocity_y_upper_multiplier,
):
    task = BouncingBallTask(
        batch_size=batch_size,
        sample_velocity_discretely=sample_velocity_discretely,
        num_x_velocities=num_x_velocities,
        num_y_velocities=num_y_velocities,
        velocity_x_lower_multiplier=velocity_x_lower_multiplier,
        velocity_x_upper_multiplier=velocity_x_upper_multiplier,
        velocity_y_lower_multiplier=velocity_y_lower_multiplier,
        velocity_y_upper_multiplier=velocity_y_upper_multiplier,
        size_x=size_x,
        size_y=size_y,
    )

    # Run the task to ensure it works fine with the input combinations
    _ = [x for x in task]

    # Grab the velocities
    velocities = task.initial_velocity
    velocities_magnitude = np.abs(velocities)

    # Check the shape of the resulting velocities array
    assert velocities.shape == (
        batch_size,
        2,
    ), f"Expected velocity shape {(batch_size, 2)}, but got {velocities.shape}"

    # Check that velocities are within expected bounds
    assert np.all(
        velocities_magnitude[:, 0] >= task.size_x * velocity_x_lower_multiplier
    ) and np.all(
        velocities_magnitude[:, 0] <= task.size_x * velocity_x_upper_multiplier
    ), "Velocity x components are out of expected bounds"
    assert np.all(
        velocities_magnitude[:, 1] >= task.size_y * velocity_y_lower_multiplier
    ) and np.all(
        velocities_magnitude[:, 1] <= task.size_y * velocity_y_upper_multiplier
    ), "Velocity y components are out of expected bounds"

    unique_vel_x = np.unique(velocities_magnitude[:, 0])
    unique_vel_y = np.unique(velocities_magnitude[:, 1])

    # Test uniform sampling
    if not sample_velocity_discretely:
        # Should get new values for each trial in the batch
        assert len(unique_vel_x) == batch_size
        assert len(unique_vel_y) == batch_size

    # Test discrete sampling
    else:
        # Correct number of unique velocities assuming the number of velocities
        # requested is smaller than the batch size
        assert len(unique_vel_x) == num_x_velocities
        assert len(unique_vel_y) == num_y_velocities

        # Returns a velocity thats the mean of the bounds if just 1 is requested
        if num_x_velocities == 1:
            assert unique_vel_x[0] == task.size_x * np.mean(
                (velocity_x_lower_multiplier, velocity_x_upper_multiplier)
            )
        if num_y_velocities == 1:
            assert unique_vel_y[0] == task.size_y * np.mean(
                (velocity_y_lower_multiplier, velocity_y_upper_multiplier)
            )


@pytest.mark.parametrize(
    "sequence_mode",
    ["invalid_mode", "Static", "IC", "Reset", "Reverse"],
)
def test_valid_sequence_modes(sequence_mode):
    valid_sequence_modes = BouncingBallTask.valid_sequence_modes

    # Test an invalid sequence_mode (should not be case sensitive)
    if sequence_mode.lower() not in valid_sequence_modes:
        with pytest.raises(ValueError):
            task = BouncingBallTask(sequence_mode=sequence_mode)

    # Test each of the valid modes
    else:
        task = BouncingBallTask(sequence_mode=sequence_mode)
        # Test the iteration works properly
        _ = zip(*[x for x in task])


@pytest.mark.parametrize(
    "sequence_mode",
    ["static", "ic", "reset", "reverse"],
)
def test_correct_sequences_for_each_sequence_modes(sequence_mode):
    sequence_length = 5
    task = BouncingBallTask(
        sequence_mode=sequence_mode,
        sequence_length=sequence_length,
    )

    # Test that sequence length gets set correctly for specific modes
    if sequence_mode in {"static", "reverse"}:
        assert len(task.sequence) == sequence_length
        assert task.sequence == [x for x in task]

    # Test it gets set correctly after each iteration
    else:
        assert len(task.sequence) == 0
        for i, (sample, target) in enumerate(task):
            assert len(task.sequence) == i + 1
            assert np.all(task.sequence[-1][0] == sample)
            assert np.all(task.sequence[-1][1] == target)


@pytest.mark.parametrize(
    "sequence_mode",
    ["static", "ic", "reset", "reverse"],
)
def test_correct_samples_and_targets_for_each_sequence_modes(sequence_mode):
    batch_size = 2
    sequence_length = 5
    features = 5
    task = BouncingBallTask(
        batch_size=batch_size,
        sequence_mode=sequence_mode,
        sequence_length=sequence_length,
    )

    # Test samples gets set correctly for specific modes
    if sequence_mode in {"static", "reverse"}:
        # Test shapes
        assert task.samples.shape == (batch_size, sequence_length, features)
        assert task.targets.shape == (batch_size, sequence_length, features)

        # Test values
        samples, targets = zip(*[x for x in task])
        samples = np.array(samples).transpose(1, 0, 2)
        targets = np.array(targets).transpose(1, 0, 2)

        assert np.all(task.samples == samples)
        assert np.all(task.targets == targets)

    # Test it gets set correctly after one iteration
    else:
        assert task.samples is None
        assert task.targets is None

        for i, (sample, target) in enumerate(task):
            # Test shapes
            assert task.samples.shape == (batch_size, i + 1, features)
            assert task.targets.shape == (batch_size, i + 1, features)

            # Test values
            assert np.all(task.samples[:, i, :] == sample)
            assert np.all(task.targets[:, i, :] == target)


@pytest.mark.parametrize(
    "sequence_mode",
    ["static", "ic", "reset", "reverse"],
)
def test_sequence_modes_correctly_reset(sequence_mode):
    batch_size = 2
    sequence_length = 10
    features = 5
    task = BouncingBallTask(
        batch_size=batch_size,
        sequence_mode=sequence_mode,
        sequence_length=sequence_length,
        probability_velocity_change=0.5,
        probability_color_change_no_velocity_change=0.5,
        probability_color_change_on_velocity_change=1.0,
    )

    # These should remain the same between iterations
    if sequence_mode in {"static", "reverse"}:
        samples = task.samples.copy()
        targets = task.targets.copy()

        for _ in range(5):
            _ = [x for x in task]
            assert np.all(samples == task.samples)
            assert np.all(targets == task.targets)

    # These should be different on every iteration
    else:
        # Run once
        _ = [x for x in task]
        samples = task.samples.copy()
        targets = task.targets.copy()
        initial_sample = samples[:, 0, :]

        for _ in range(5):
            _ = [x for x in task]
            assert np.any(samples != task.samples)
            assert np.any(targets != task.targets)

            # For IC, the first sample should be same
            if sequence_mode == "ic":
                assert np.all(task.samples[:, 0, :] == initial_sample)

            samples = task.samples.copy()
            targets = task.targets.copy()


# def test_reverse_sequence_mode_correctly_reverses_sequences()
@pytest.mark.parametrize(
    "return_change",
    [False, True],
)
def test_return_change_correctly_returns_changes(return_change):
    batch_size = 2
    sequence_length = 10
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        return_change=return_change,
    )

    if not return_change:
        # Samples and targets should be the same for no return change
        assert task.samples.shape == (batch_size, sequence_length, features)
        assert task.targets.shape == (batch_size, sequence_length, features)
    else:
        # Samples should be unaffected but targets now have extra features
        assert task.samples.shape == (batch_size, sequence_length, features)
        assert task.targets.shape != (batch_size, sequence_length, features)
        assert task.targets.shape[-1] > features


@pytest.mark.parametrize(
    "return_change_mode",
    ["Any", "Feature", "Source", "invalid_mode"],
)
def test_return_change_modes_have_correct_shapes(return_change_mode):
    batch_size = 2
    sequence_length = 10
    features = 5
    if (
        return_change_mode.lower()
        not in BouncingBallTask.valid_return_change_modes
    ):
        with pytest.raises(ValueError):
            task = BouncingBallTask(
                sequence_mode="static",
                batch_size=batch_size,
                sequence_length=sequence_length,
                return_change=True,
                return_change_mode=return_change_mode,
            )

    else:
        task = BouncingBallTask(
            sequence_mode="static",
            batch_size=batch_size,
            sequence_length=sequence_length,
            return_change=True,
            return_change_mode=return_change_mode,
        )

        # Ensure samples is unaffected
        assert task.samples.shape == (batch_size, sequence_length, features)

        if task.return_change_mode == "any":
            # For "any", changes should be combined for one extra feature
            assert task.targets.shape[-1] == features + 1
        elif task.return_change_mode == "feature":
            # For "feature", velocity and color changes are added separately,
            # resulting in two extra features
            assert task.targets.shape[-1] == features + 2

        elif task.return_change_mode == "source":
            # For "source", changes from the sources of velocity and color
            # changes are added
            assert task.targets.shape[-1] == features + 4


@pytest.mark.parametrize("initial_timestep_is_changepoint", [True, False])
@pytest.mark.parametrize("return_change_mode", ["any", "feature", "source"])
def test_initial_changes(initial_timestep_is_changepoint, return_change_mode):
    batch_size = 2
    sequence_length = 10
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        return_change=True,
        return_change_mode=return_change_mode,
        initial_timestep_is_changepoint=initial_timestep_is_changepoint,
    )

    # Should be 0 for no changepoint and 1 fo changepoint
    assert np.all(task.initial_changes == int(initial_timestep_is_changepoint))


@pytest.mark.parametrize(
    "color_mask_mode",
    ["Fade", "INNER", "outer"],
)
def test_grayzone_and_color_mask_mode(color_mask_mode):
    # Create a task instance that doesn't have any color transitions so observe
    # sample behavior as it enters and leaves the gray zone
    batch_size = 100
    sequence_length = 500
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        probability_velocity_change=0.0,
        probability_color_change_no_velocity_change=0.0,
        probability_color_change_on_velocity_change=0.0,
        color_mask_mode=color_mask_mode,
    )
    # Test all positions in the grayzone are gray
    assert (
        task.samples[
            np.logical_and(
                task.mask_start + task.ball_radius < task.samples[:, :, 0],
                task.samples[:, :, 0] < task.mask_end - task.ball_radius,
            )
        ][:, 2:]
        == task.mask_color
    ).all()

    # Check the unique colors are what we expect them to be
    colors = task.samples[:, :, 2:].reshape(batch_size * sequence_length, 3)
    unique_colors = np.unique(colors, axis=0)
    num_unique_colors = len(unique_colors)
    num_valid_colors = len(task.valid_colors) + 1  # Add one for gray

    # Subselect the positions in the border of the grayzone to inspect behvior
    # of the different modes
    transition_indices = np.logical_and(
        task.samples[:, :, 0] < task.mask_start + task.ball_radius / 4,
        task.mask_start - task.ball_radius / 4 < task.samples[:, :, 0],
    )
    transition_colors = task.samples[transition_indices][:, 2:]
    initial_colors = task.targets[transition_indices][:, 2:]

    if color_mask_mode.lower() == "fade":
        # Consider adding test for the fading colors itself
        assert num_unique_colors > num_valid_colors

        mean_color_gray = (initial_colors + task.mask_color) / 2
        sq_diff_color_mean = (transition_colors - mean_color_gray) ** 2
        sq_diff_color_color = (transition_colors - initial_colors) ** 2
        sq_diff_color_gray = (
            transition_colors
            - np.ones_like(transition_colors) * task.mask_color[0]
        ) ** 2

        # Must be more different from either the initial color or gray than to
        # the mean of the two
        assert np.logical_and(
            sq_diff_color_mean < sq_diff_color_color,
            sq_diff_color_mean < sq_diff_color_gray,
        ).all()

    elif color_mask_mode.lower() == "inner":
        assert num_unique_colors <= num_valid_colors
        # Must all still equal the initial colors
        assert (transition_colors == initial_colors).all()

    elif color_mask_mode.lower() == "outer":
        assert num_unique_colors <= num_valid_colors
        # Must all already equal the grayzone color
        assert (transition_colors == task.mask_color).all()


@pytest.mark.parametrize("probability_velocity_change", [0.0, 0.01, 0.1])
def test_pvc_causes_correct_random_velocity_changes(
    probability_velocity_change,
):
    batch_size = 4096
    sequence_length = 1000
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        probability_velocity_change=probability_velocity_change,
        probability_color_change_no_velocity_change=0.0,
        probability_color_change_on_velocity_change=1.0,
        return_change=True,
        return_change_mode="source",
        initial_timestep_is_changepoint=False,
    )
    # Subselect all instances where there was a random velocity change
    timesteps_velocity_changes_random = task.targets[task.targets[..., -3] == 1]

    # Fraction of random velocity changes / total timesteps should match pvc
    assert isclose(
        len(timesteps_velocity_changes_random) / task.targets[:, :, 0].size,
        probability_velocity_change,
        rel_tol=0.05,
    )


@pytest.mark.parametrize(
    "probability_color_change_on_velocity_change", [0.0, 0.1, 0.5, 0.9, 1.0]
)
def test_pccovc_causes_correct_color_changes(
    probability_color_change_on_velocity_change,
):
    batch_size = 4096
    sequence_length = 1000
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        probability_velocity_change=0.0,
        probability_color_change_no_velocity_change=0.0,
        probability_color_change_on_velocity_change=probability_color_change_on_velocity_change,
        return_change=True,
        return_change_mode="feature",
        initial_timestep_is_changepoint=False,
        color_change_bounce_delay=0,
    )

    # Subselect all instances where there was a velocity change
    timesteps_velocity_changes = task.targets[task.targets[..., -2] == 1][
        :, -2:
    ]

    # Number of color changes encountered on a bounce should match pccovc
    assert isclose(
        timesteps_velocity_changes[:, -1].mean(),
        probability_color_change_on_velocity_change,
        rel_tol=0.05,
    )


@pytest.mark.parametrize("pvc", [0.001, 0.01])
def test_bounce_and_random_velocity_changes_occur_at_correct_locations(pvc):
    batch_size = 1024
    sequence_length = 500
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        probability_velocity_change=pvc,
        probability_color_change_no_velocity_change=0.0,
        return_change=True,
        return_change_mode="source",
        initial_timestep_is_changepoint=False,
        color_change_bounce_delay=0,
        min_t_color_change=0,
    )

    # Subselect for timesteps where there is a velocity change
    timesteps_velocity_changes_bounce = task.targets[
        task.targets[..., -4] == 1
    ][:, :2]
    timesteps_velocity_changes_random = task.targets[
        task.targets[..., -3] == 1
    ][:, :2]

    # Define the lower bound of a bounce position in the tops of the frame and
    # the upper bound of the bottoms of the frame
    bounce_top_lower_bound = (
        task.size_frame - task.ball_radius - task.velocity_upper_bound * task.dt
    )
    bounce_bottom_upper_bound = (
        np.zeros_like(task.size_frame)
        + task.ball_radius
        + task.velocity_upper_bound * task.dt
    )

    # The positions resulting from a bounce can only within these bounds
    assert (
        np.logical_or(
            timesteps_velocity_changes_bounce < bounce_bottom_upper_bound,
            timesteps_velocity_changes_bounce > bounce_top_lower_bound,
        )
        .any(axis=-1)
        .all()
    )

    # Define the upper bound of the top position and lower bound of the bottom
    # position that can have random velocity changes
    random_top_upper_bound = (
        task.size_frame - task.ball_radius + task.velocity_upper_bound * task.dt
    )
    random_bottom_lower_bound = (
        np.zeros_like(task.size_frame)
        + task.ball_radius
        - task.velocity_upper_bound * task.dt
    )

    # The positions resulting from a bounce can only within these bounds
    assert np.logical_or(
        timesteps_velocity_changes_random < random_top_upper_bound,
        timesteps_velocity_changes_random > random_bottom_lower_bound,
    ).all()


@pytest.mark.parametrize("color_change_bounce_delay", [0, 5, 10])
def test_bounce_and_random_color_changes_occur_correctly(
    color_change_bounce_delay,
):
    batch_size = 1024
    sequence_length = 500
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        return_change=True,
        return_change_mode="source",
        initial_timestep_is_changepoint=False,
        color_change_bounce_delay=color_change_bounce_delay,
        min_t_color_change=0,
    )

    # Subselect for timesteps where there is a color change
    indices_color_changes_bounce = np.where(task.targets[..., -2] == 1)
    indices_color_changes_random = np.where(task.targets[..., -1] == 1)

    # Combine for aggregate color change testing
    indices_batch_color_changes, indices_timestep_color_changes = (
        np.concatenate([idx1, idx2])
        for idx1, idx2 in zip(
            indices_color_changes_bounce, indices_color_changes_random
        )
    )

    # Check that the colors are different between every timestep that flagged a
    # color change and the timestep before it
    assert not np.any(  # Check if any comparisons failed
        np.all(  # Check for equal channels values
            task.targets[  # Colors of the timestep with the change
                indices_batch_color_changes, indices_timestep_color_changes
            ][:, 2:5]
            == task.targets[  # Colors of one timestep before
                indices_batch_color_changes, indices_timestep_color_changes - 1
            ][:, 2:5],
            axis=-1,  # Select channels
        )
    )

    # Split to do relative indexing
    (
        indices_batch_color_changes_random,
        indices_timestep_color_changes_random,
    ) = indices_color_changes_random
    (
        indices_batch_color_changes_bounce,
        indices_timestep_color_changes_bounce,
    ) = indices_color_changes_bounce

    # Random color changes should occur on timesteps where velocity didnt change
    assert np.all(
        task.targets[
            indices_batch_color_changes_random,
            indices_timestep_color_changes_random,
        ][:, -4:-2]
        == 0
    )

    # Bounce color changes should only occur in timesteps where velocity changed
    assert np.all(
        np.any(
            task.targets[
                indices_batch_color_changes_bounce,
                indices_timestep_color_changes_bounce
                - color_change_bounce_delay,
            ][:, -4:-2]
            == 1,
            axis=-1,
        )
    )


@pytest.mark.parametrize("pccnvc", [0.0, 0.01, 0.045])
def test_pccnvc_causes_correct_color_changes(
    pccnvc,
):
    batch_size = 4096
    sequence_length = 500
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        probability_color_change_no_velocity_change=pccnvc,
        probability_color_change_on_velocity_change=0.0,
        return_change=True,
        return_change_mode="feature",
        initial_timestep_is_changepoint=False,
        min_t_color_change=0,
        color_change_bounce_delay=0,
    )

    # Subselect all instances where there was a velocity change
    timesteps_color_changes = task.targets[task.targets[..., -1] == 1]

    # There should be no velocity changes at these locations since pccovc is 0
    assert np.all(timesteps_color_changes[:, -2] == 0)

    # Precompute
    total_color_changes = len(timesteps_color_changes)
    total_timesteps = task.targets[:, :, 0].size
    total_bounces = len(task.targets[task.targets[..., -1] == 1])

    # Percentage should relfect number of random color changes in timesteps where
    # there is no bounce
    assert isclose(
        total_color_changes / (total_timesteps - total_bounces),
        pccnvc,
        rel_tol=0.05,
    )


@pytest.mark.parametrize("pvc", [0.0, 0.001])
@pytest.mark.parametrize("pccovc", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("pccnvc", [0.01, 0.045])
@pytest.mark.parametrize("min_t_color_change", [0, 5, 10, 20])
def test_min_t_color_change_causes_predictable_change_statistics(
    pvc,
    pccovc,
    pccnvc,
    min_t_color_change,
):
    batch_size = 1024
    sequence_length = 500
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        probability_velocity_change=pvc,
        probability_color_change_no_velocity_change=pccnvc,
        probability_color_change_on_velocity_change=pccovc,
        return_change=True,
        return_change_mode="source",
        initial_timestep_is_changepoint=False,
        min_t_color_change=min_t_color_change,
        color_change_bounce_delay=0,
    )

    # Velocity and color changes dont happen on the first timestep
    sequence_length_possible_changes = task.targets.shape[1] - 1
    total_possible_change_timesteps = (
        sequence_length_possible_changes * batch_size
    )

    # Subselect all instances where there was a color change
    timesteps_color_change_bounce = task.targets[task.targets[..., -2] == 1]
    timesteps_color_change_random = task.targets[task.targets[..., -1] == 1]
    timesteps_color_changes = np.concatenate(
        [timesteps_color_change_random, timesteps_color_change_bounce]
    )

    # Subselect all instances where there was a velocity change
    timesteps_velocity_change_bounce = task.targets[task.targets[..., -4] == 1]
    timesteps_velocity_change_random = task.targets[task.targets[..., -3] == 1]
    timesteps_velocity_changes = np.concatenate(
        [timesteps_velocity_change_random, timesteps_velocity_change_bounce]
    )

    # Tally up totals
    total_color_changes = len(timesteps_color_changes)
    total_velocity_changes = len(timesteps_velocity_changes)

    # Check the total number of random bounces matches pvc and is unaffected by
    # min_t_color_change
    fraction_velocity_change_random = len(timesteps_velocity_change_random) / (
        total_possible_change_timesteps - len(timesteps_velocity_change_bounce)
    )
    assert isclose(fraction_velocity_change_random, pvc, rel_tol=0.2)

    # Check that the number of color changes on the bounces matches pccovc and
    # is unaffected by min_t_color_change
    fraction_color_on_velocity_change = len(
        timesteps_color_change_bounce
    ) / len(timesteps_velocity_changes)
    assert isclose(fraction_color_on_velocity_change, pccovc, rel_tol=0.2)

    # Compute an estimate for the fraction of timesteps that have a bounce
    fraction_velocity_change_bounce = (
        len(timesteps_velocity_change_bounce) / total_possible_change_timesteps
    )

    # Count random color changes, which cant happen on bounce timesteps
    total_possible_random_color_change_timesteps = (
        total_possible_change_timesteps - total_velocity_changes
    )

    # What fraction of timesteps didnt have a bounce excluding the first
    fraction_no_velocity_changes = (
        1 - total_velocity_changes / total_possible_change_timesteps
    )

    # The total number of deadzone timesteps is not strictly based on the number
    # of color changes as some of them happen at the end of the sequence where
    # the value should be smaller than min_t_color_change. Compute the fraction
    # of timesteps per sequence at the end
    fraction_end_sequence = (
        min_t_color_change / sequence_length_possible_changes
    )

    # Compute an estimate of probability of color change per timestep
    probability_velocity_change_per_timestep = (
        fraction_velocity_change_bounce
        + pvc * (1 - fraction_velocity_change_bounce)
    )
    probability_color_change_per_timestep = (
        pccnvc * (1 - probability_velocity_change_per_timestep)
        + pccovc * probability_velocity_change_per_timestep
    )

    # Use this estimate to get an estimate of the total number of deadzone
    # timesteps at the end of a single sequence
    end_deadzone_sum = sum(
        [
            (1 - probability_color_change_per_timestep) ** i
            * probability_color_change_per_timestep
            * (min_t_color_change - i)
            for i in range(min_t_color_change + 1)
        ]
    )

    total_deadzone_timesteps = (
        # Each change creates a range of timesteps where the color cannot change
        # but weight according to the percentage of a sequence where this applies
        (1 - fraction_end_sequence) * total_color_changes * min_t_color_change
        # Subtract off the naive estimate for the number of deadzone timesteps
        # for the end of each sequence and then add in the estimate, weighted
        # by the fraction of timesteps that are part of this end deadzone
        + fraction_end_sequence
        * batch_size
        * (end_deadzone_sum - min_t_color_change + 1)
    )

    # Percentage of color changes should be adjusted according to the number of
    # timesteps where color changes were permitted and there were no velocity_changes
    corrected_pccnvc = (
        len(timesteps_color_change_random)
        / (
            total_possible_random_color_change_timesteps
            - total_deadzone_timesteps
        )
    ) * fraction_no_velocity_changes

    assert isclose(corrected_pccnvc, pccnvc, rel_tol=0.05)


@pytest.mark.parametrize("pccovc", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("color_change_bounce_delay", [5, 10, 25])
def test_color_change_bounce_delay_causes_correct_color_changes(
    color_change_bounce_delay, pccovc
):
    batch_size = 1024
    sequence_length = 500
    features = 5
    task = BouncingBallTask(
        sequence_mode="static",
        batch_size=batch_size,
        sequence_length=sequence_length,
        probability_velocity_change=0.0,
        probability_color_change_no_velocity_change=0.0,
        probability_color_change_on_velocity_change=pccovc,
        return_change=True,
        return_change_mode="source",
        initial_timestep_is_changepoint=False,
        color_change_bounce_delay=color_change_bounce_delay,
    )
    # Make sure changing the color change delay does not affect unrelated changes
    assert np.all(task.targets[..., -3] == 0)  # Random vel changes
    assert np.all(task.targets[..., -1] == 0)  # Random color changes

    # Get the indices of the velocity and color changes
    _, indices_velocity_change_bounce = np.where(task.targets[..., -4] == 1)
    _, indices_color_change_bounce = np.where(task.targets[..., -2] == 1)

    # Get the locations for where color changes should be
    predicted_indices_color_change_bounce = (
        indices_velocity_change_bounce + color_change_bounce_delay
    )

    # Filter out the predictions that are longer than the sequence length
    predicted_indices_color_change_bounce_valid = (
        predicted_indices_color_change_bounce[
            predicted_indices_color_change_bounce < sequence_length
        ]
    )

    # Check that all color changes on bounces happened where they're supposed to
    assert np.all(
        np.isin(
            indices_color_change_bounce,
            predicted_indices_color_change_bounce_valid,
        )
    )


# # def test_ccovc_distributional_delay
