# Controlled Dataset Variant Tests

This directory contains tests for controlled dataset variants. The structure is designed to be scalable and maintainable as new variants are added.

## Structure

- `base_variant_tests.py` - Base test class containing tests that all variants should pass
- `test_no_change.py` - Tests specific to the no_change variant
- `conftest.py` - Fixtures specific to variant testing

## Running Tests

```bash
# Run all variant tests
pytest tests/controlled_variants/

# Run tests for a specific variant
pytest -m no_change

# Run all variant tests (current and future)
pytest -m variant

# Run specific test file
pytest tests/controlled_variants/test_no_change.py
```

## Adding New Variant Tests

When implementing a new variant (e.g., sudden_change), follow these steps:

1. Create `test_sudden_change.py` in this directory
2. Import and inherit from `BaseVariantTests`
3. Add the appropriate pytest markers (`@pytest.mark.variant`, `@pytest.mark.sudden_change`)
4. Override the `variant_dataset` fixture to provide your variant's dataset
5. Add variant-specific tests as needed

Example structure:
```python
from .base_variant_tests import BaseVariantTests

@pytest.mark.controlled_dataset
@pytest.mark.variant
@pytest.mark.sudden_change
class TestSuddenChangeVariant(BaseVariantTests):
    @pytest.fixture
    def variant_dataset(self, sudden_change_dataset):
        return sudden_change_dataset
    
    # Add variant-specific tests here
```

## Common Tests (from BaseVariantTests)

All variants automatically get these tests:
- `test_position_consistency_across_color_rotations` - Ensures positions are identical across color rotations
- `test_metadata_structure` - Verifies proper metadata structure
- `test_three_color_rotation_pattern` - Checks for correct RGB rotations
- `test_change_vector_shape` - Validates change vector dimensions
- `test_dataframe_completeness` - Ensures all required columns exist
- `test_trial_count_formula` - Verifies correct number of trials

## Benefits

1. **No code duplication** - Common tests are inherited
2. **Easy to add variants** - Just inherit and add specific tests
3. **Clear organization** - Each variant has its own test file
4. **Flexible execution** - Run by variant, all variants, or specific files
5. **Scalable** - Structure supports many variants without clutter
