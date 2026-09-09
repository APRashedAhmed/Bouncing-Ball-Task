# CLAUDE.md - Bouncing Ball Task

## Overview

This repository contains the Bouncing Ball Task implementation used for computational modeling and human experiments in temporal abstraction and event modeling research.

## Testing Framework

### Requirements
- **Framework**: This project uses `pytest` as the testing framework
- **Location**: All tests must be placed in the `tests/` directory
- **Naming**: Test files should follow the pattern `test_*.py` or `*_test.py`

### Running Tests

```bash
# Run all tests
cd Bouncing-Ball-Task
pytest

# Run specific test file
pytest tests/test_color_controlled_dataset_endproduct.py

# Run with verbose output
pytest -v

# Run specific test function
pytest tests/test_color_controlled_dataset_endproduct.py::TestColorControlledDatasetEndProduct::test_position_consistency_across_variants

# Run tests with specific markers
pytest -m color_controlled_dataset
pytest -m "not slow"
```

### Test Structure

The test suite is organized as follows:
- `tests/conftest.py` - Shared fixtures for all tests
- `tests/test_color_controlled_dataset_endproduct.py` - End-product validation tests
- `tests/test_change_vector_generation.py` - Tests for change vector mechanics
- `tests/test_color_controlled_format_parity.py` - On-disk schema parity against the v3_2_2 snapshot
- `tests/fixtures/v3_2_2_schema.json` - Read-only snapshot of the real dataset's schema
- `tests/color_controlled_variants/` - Variant-specific tests and shared base class
  - `tests/color_controlled_variants/conftest.py` - Variant-specific fixtures
  - `tests/color_controlled_variants/base_variant_tests.py` - Shared base class for variant tests
  - `tests/color_controlled_variants/test_no_change.py` - Tests specific to the no_change variant

### Available Fixtures

The following fixtures are available in `conftest.py`:
- `color_controlled_dataset_small` - Small dataset for fast tests (5 base sequences)
- `color_controlled_dataset_with_videos` - Dataset with video file generation
- `multiple_variants_dataset` - Dataset with multiple variants for testing
- `temp_output_dir` - Temporary directory for test outputs

## Color Controlled Dataset Implementation

The color controlled dataset freezes a trajectory and lets variants author ONLY
its colour change events, giving precise experimental control without ever
letting the change record contradict the motion.

### Key Concepts

1. **Base Sequences**: trajectories integrated from explicit-or-sampled initial
   conditions with `probability_velocity_change = 0`, so a trajectory is a pure
   function of its initial conditions. `control_end=False` keeps the trajectory
   (the trial STARTS at the requested position, `sequence_mode="static"`);
   `control_end=True` reverses it (the trial ENDS there, `sequence_mode="reverse"`).
2. **Change Vectors** (targets channels 5-8): 4 channels, split by authorability
   - Channel 5 `vc_bounce`: wall bounces — DERIVED, read-only ledger
   - Channel 6 `vc_random`: random velocity changes — identically 0 under PVC=0
   - Channel 7 `cc_bounce`: contingent color change — AUTHORED by variants
   - Channel 8 `cc_random`: random color change — AUTHORED by variants
   Only colour is controlled: velocity is downstream of the frozen trajectory,
   so the velocity channels are never exposed for writing.
3. **Variants**: experimental conditions that author the two colour channels
4. **Color Rotations**: Each variant×base combination produces 3 trials (RGB)

### Dataset Generation Formula
```
Total trials = num_base_sequences × num_variants × num_colors (3)
```

### Parameters

The color controlled dataset uses clean parameter classes:

```python
# Core parameters in ColorControlledDatasetParameters
num_base_sequences: int = 10          # Number of base sequences
seed: Optional[int] = None            # Random seed (False = caller seeds)
duration: int = 50                    # Milliseconds PER FRAME
variable_length: bool = False         # Fixed/variable length sequences
video_length_min_s: float = 7.5       # Variable-length knobs: same names and
exp_scale: float = 3.75               # values as HumanDatasetParameters
fixed_video_length: Optional[int] = None
initial_position: Optional[tuple] = None   # (N, 2), None => sampled
initial_velocity: Optional[tuple] = None   # (N, 2), None => sampled
control_end: Optional[tuple] = None        # (N,) bool, None => all False
```

`ColorControlledTaskParameters` subclasses the human task's `TaskParameters`, so
the stimulus is parameter-identical to the real task (`dt=0.1`,
`sequence_length=600`, `color_mask_mode="outer"`, warmup / min_t /
transition_tol) with `probability_velocity_change=0` and
`initial_timestep_is_changepoint=False`. Both are rejected with a `ValueError`
if overridden to anything else.

### `length` / `duration` contract

`length` is a FRAME COUNT (the full `sequence_length`, or the truncated count
under `variable_length`), `duration` is milliseconds PER FRAME, and
`length_ms = length * duration`. Variable-length trials keep their LAST `length`
frames, so they require `control_end=True` for every trial. Their lengths are
drawn by the human task's own sampler (`htaskutils.compute_dataset_size_video_based`,
shared with `model_bouncing_ball`): `frames = rint((Exp(exp_scale s) +
video_length_min_s) / duration)`, draws of 3x the minimum or more rejected;
the task then integrates at `sequence_length = max(lengths)` (HDS:87), so a
caller-supplied `sequence_length` is overridden under `variable_length`.

### On-disk format

`generate_color_controlled_dataset_with_videos()` saves through the real task's
`save_video_dataset`, producing `trial_meta.csv` (indexed `Video ID`),
`dataset_meta.pkl`, and `videos/block_1/video_N/video_N_{samples,parameters,
color_change}.csv` + `.mp4`. The trial dataframe is built by calling the real
task's own helpers (`hds.generate_dataset_metadata`), so its schema matches the
human dataset — see `tests/test_color_controlled_format_parity.py` and the
snapshot `tests/fixtures/v3_2_2_schema.json`. `PCCNVC_adjusted` and `cwl` are
omitted by design; the color-controlled extras (`variant`, `base_sequence_idx`,
`color_rotation_idx`, `control_end`, `Total Changes`) are additive.

### Adding New Variants

To add a new variant:

1. Create a new file: `src/bouncing_ball_task/color_controlled_bouncing_ball/your_variant.py`
2. Implement the variant function — the **color-only** authoring contract:
```python
def generate_your_variant_trials(positions, bounce_frames, params):
    """Your variant description.

    Args:
        positions: (N, T, 2) trajectory positions — read-only copy.
        bounce_frames: (N, T) bool — the derived wall-bounce ledger (targets
            channel 5), read-only copy.
        params: dict with `controlled_dataset_parameters`, `task_parameters`
            (the resolved base task kwargs), `initial_position`,
            `initial_velocity`, `control_end`.

    Returns:
        (N, T, 2) binary array of authored color events:
        [:, :, 0] = contingent color change (channel 7),
        [:, :, 1] = random color change (channel 8).
    """
    color_events = np.zeros(bounce_frames.shape + (2,), dtype=int)
    ...
    return color_events


# Optional: merged into dict_metadata[<variant>]["variant_metadata"]
generate_your_variant_trials.variant_metadata = {'change_strategy': '...'}
```

The runtime enforces the contingency invariant (`ValueError` on violation):
`ch7[t] == 1 => ch5[t] == 1` (contingent color changes only where a real wall
bounce occurred) and `ch7[t] + ch8[t] <= 1` (the real task never fires both).

3. Register in `dataset.py`:
```python
dict_trial_type_generation_funcs['your_variant'] = generate_your_variant_trials
```
4. Create tests in `tests/color_controlled_variants/test_your_variant.py`

> **Note:** Each variant implementation must record its own `hazard_rate` and
> `contingency` in its per-variant metadata sub-dict. These are no longer set as
> hardcoded top-level keys in `controlled_parameters` (they were always `0.0`,
> which is wrong for any non-zero variant).

### Important Implementation Notes

- **No old parameters**: The implementation no longer uses `total_dataset_length` or `num_blocks`
- **Direct generation**: Use `generate_color_controlled_dataset()` directly, not compatibility wrappers
- **Position consistency**: All variants from the same base sequence have identical positions
- **Color-only authoring**: variants author channels 7/8 only; the velocity
  channels are the derived, read-only bounce ledger

## Code Quality

- Run tests before committing: `pytest`
- Follow existing code patterns and conventions
- Document new variants and their behavior
- Ensure reproducibility with fixed seeds

## Commit conventions

This repo now uses [Conventional Commits](https://www.conventionalcommits.org/):
prefix commit subjects with `fix:`, `feat:`, `refactor:`, `test:`, `docs:`, or `chore:`.
