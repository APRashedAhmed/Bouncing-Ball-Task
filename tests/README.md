# Bouncing Ball Task Test Suite

This directory contains the test suite for the Bouncing Ball Task, with specific focus on the controlled dataset generation functionality.

## Running Tests

### Basic Usage

Run all tests:
```bash
pytest
```

Run tests for a specific module:
```bash
pytest tests/controlled_variants/test_no_change.py
```

Run tests with verbose output:
```bash
pytest -v
```

### Using Test Markers

The test suite uses markers to categorize tests:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (e.g., full dataset generation)
- `@pytest.mark.dataset_generation` - Tests that generate datasets
- `@pytest.mark.no_change` - Tests specific to no_change variant
- `@pytest.mark.controlled_dataset` - Tests for controlled dataset generation

Run only unit tests:
```bash
pytest -m unit
```

Run tests excluding slow ones:
```bash
pytest -m "not slow"
```

Run only no_change variant tests:
```bash
pytest -m no_change
```

Run no_change tests but skip slow ones:
```bash
pytest -m "no_change and not slow"
```

### Using the Test Runner Script

A convenience script is provided for running no_change validation tests:

```bash
# Run standard tests (unit + integration, skip slow)
python tests/run_validation.py

# Run only quick unit tests
python tests/run_validation.py --quick

# Run only integration tests
python tests/run_validation.py --integration

# Run all tests including slow ones
python tests/run_validation.py --all

# Generate coverage report
python tests/run_validation.py --coverage

# Generate HTML test report
python tests/run_validation.py --html
```

## Test Organization

### Controlled Dataset Tests

The controlled dataset implementation includes the following tests:

1. **test_controlled_dataset_endproduct.py** - End-product validation tests
2. **test_change_vector_generation.py** - Change vector mechanics
3. **controlled_variants/test_no_change.py** - Tests specific to the no_change variant
4. **controlled_variants/base_variant_tests.py** - Shared base class for variant tests

### Shared Fixtures

The `tests/conftest.py` file provides shared fixtures used across tests:

- `session_temp_dir` - Session-scoped temporary directory
- `temp_output_dir` - Temporary directory for each test
- `controlled_dataset_small` - Small controlled dataset (5 base sequences)
- `controlled_dataset_with_videos` - Controlled dataset with saved video files
- `multiple_variants_dataset` - Dataset generated with multiple variants
- `default_controlled_params` - Default `ControlledDatasetParameters` instance
- `default_task_params` - Default `ControlledTaskParameters` instance
- `reset_random_state` - Autouse fixture that seeds NumPy to 42 before each test

The `tests/controlled_variants/conftest.py` file adds variant-specific fixtures:

- `no_change_dataset` - Dataset generated with only the `no_change` variant

## Writing New Tests

When adding new tests:

1. Use appropriate markers to categorize your tests
2. Use shared fixtures from conftest.py when possible
3. Follow the existing naming convention: `test_<feature>_<aspect>.py`
4. Add docstrings to test methods explaining what they validate

Example:
```python
@pytest.mark.unit
@pytest.mark.controlled_dataset
def test_new_feature(controlled_dataset_small):
    """Test that new feature works correctly."""
    # Your test code here
    assert result == expected
```

## Dependencies

Required packages:
- pytest
- pytest-cov (for coverage reports)
- pytest-html (for HTML reports)

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-html
```
