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
pytest tests/test_controlled_dataset_endproduct.py

# Run with verbose output
pytest -v

# Run specific test function
pytest tests/test_controlled_dataset_endproduct.py::TestControlledDatasetEndProduct::test_position_consistency_across_variants

# Run tests with specific markers
pytest -m controlled_dataset
pytest -m "not slow"
```

### Test Structure

The test suite is organized as follows:
- `tests/conftest.py` - Shared fixtures for all tests
- `tests/test_controlled_dataset_endproduct.py` - End-product validation tests
- `tests/test_change_vector_generation.py` - Tests for change vector mechanics
- `tests/controlled_variants/` - Variant-specific tests and shared base class
  - `tests/controlled_variants/conftest.py` - Variant-specific fixtures
  - `tests/controlled_variants/base_variant_tests.py` - Shared base class for variant tests
  - `tests/controlled_variants/test_no_change.py` - Tests specific to the no_change variant

### Available Fixtures

The following fixtures are available in `conftest.py`:
- `controlled_dataset_small` - Small dataset for fast tests (5 base sequences)
- `controlled_dataset_with_videos` - Dataset with video file generation
- `multiple_variants_dataset` - Dataset with multiple variants for testing
- `temp_output_dir` - Temporary directory for test outputs

## Controlled Dataset Implementation

The controlled dataset uses a **change vector manipulation approach** for precise experimental control.

### Key Concepts

1. **Base Sequences**: Initial trajectories generated with standard physics
2. **Change Vectors**: 4-channel arrays tracking velocity and color changes
   - Channel 0: Bounce velocity changes
   - Channel 1: Random velocity changes
   - Channel 2: Color changes on velocity change
   - Channel 3: Random color changes
3. **Variants**: Different experimental conditions that modify change vectors
4. **Color Rotations**: Each variant×base combination produces 3 trials (RGB)

### Dataset Generation Formula
```
Total trials = num_base_sequences × num_variants × num_colors (3)
```

### Parameters

The controlled dataset uses clean parameter classes:

```python
# Core parameters in ControlledDatasetParameters
num_base_sequences: int = 10  # Number of base sequences
seed: Optional[int] = None    # Random seed
duration: int = 1000          # Duration in milliseconds
variable_length: bool = False # Fixed/variable length sequences
```

### Adding New Variants

To add a new variant:

1. Create a new file: `src/bouncing_ball_task/controlled_bouncing_ball/your_variant.py`
2. Implement the variant function:
```python
def generate_your_variant_trials(change_vectors, controlled_dataset_parameters, 
                                task_parameters, base_task):
    """Your variant description."""
    # Modify change_vectors as needed
    # Return (modified_change_vectors, metadata_dict)
    return change_vectors, {'variant_name': 'your_variant', ...}
```
3. Register in `dataset.py`:
```python
dict_trial_type_generation_funcs['your_variant'] = generate_your_variant_trials
```
4. Create tests in `tests/controlled_variants/test_your_variant.py`

> **Note:** Each variant implementation must record its own `hazard_rate` and
> `contingency` in its per-variant metadata sub-dict. These are no longer set as
> hardcoded top-level keys in `controlled_parameters` (they were always `0.0`,
> which is wrong for any non-zero variant).

### Important Implementation Notes

- **No old parameters**: The implementation no longer uses `total_dataset_length` or `num_blocks`
- **Direct generation**: Use `generate_controlled_dataset()` directly, not compatibility wrappers
- **Position consistency**: All variants from the same base sequence have identical positions
- **Change vector approach**: Variants only modify change vectors, not trajectories

## Code Quality

- Run tests before committing: `pytest`
- Follow existing code patterns and conventions
- Document new variants and their behavior
- Ensure reproducibility with fixed seeds

## Commit conventions

This repo now uses [Conventional Commits](https://www.conventionalcommits.org/):
prefix commit subjects with `fix:`, `feat:`, `refactor:`, `test:`, `docs:`, or `chore:`.
